# Wake Word Engine v12 — Implementation Plan (Two-Stage Cascade + Tier C)

**Audience:** an AI coding agent (build target: a Kaggle notebook).
**Compute target:** Kaggle, **2× T4 GPUs** (16GB each, no NVLink) — use `DistributedDataParallel` (or at minimum `nn.DataParallel` if DDP setup in a notebook is inconvenient), mixed precision (`torch.cuda.amp`), and batch episodes so both GPUs stay fed. Details in Section 6.
**Priorities, in order:** (1) accuracy — specifically, do not confuse a wake phrase with its own truncation/prefix, (2) lightweight enough for continuous on-device inference (Stage 1 in particular), (3) fully on-device — no cloud stage, no speaker verification.
**This is a from-scratch build.** Start directly with the Tier C (truncation-suppression) objective baked into Layer-1 training — do not build the simpler positive-only version first and retrofit.

---

## 1. Architecture (unchanged from prior spec, restated for a standalone build doc)

```
mic ──► Stage 0: AEC (echo cancellation, needs TTS reference signal) ──► VAD gate
                                                                             │
                                                          ┌──────────────────┘
                                                          ▼
                                    STAGE 1 — always-on, low-power
                                    DS-CNN, temporal_head="pool", INT8
                                    high-recall threshold (loose)
                                                          │ rare trigger
                                                          ▼
                                    Utterance endpointing (VAD onset/offset,
                                    variable length, NOT fixed 1s window)
                                                          │
                                                          ▼
                                    STAGE 2 — on-trigger-only, precise
                                    DS-CNN, temporal_head="gru"
                                    Tier-C-trained encoder + hard-negative-
                                    calibrated threshold + duration gate
                                                          │
                                                          ▼
                                    WakeWordTrigger (debounce/hangover)
```

Stage 1 and Stage 2 are trained from **the same Layer-1 training run** (same episodic training loop, same Tier C objective) but exported as two separately-sized checkpoints via the existing `temporal_head` config switch (`"pool"` vs `"gru"`) — no need for two separate architectures or two separate training pipelines. Train once with `temporal_head="gru"` (Stage 2, full accuracy), and additionally either (a) train a second, smaller run with `temporal_head="pool"` and a narrower channel width for Stage 1, or (b) distill Stage 1 from the trained Stage 2 model (recommended — see Section 6.3). Do NOT skip Stage 1 lightweighting to save build time; it's what makes always-on operation viable.

---

## 2. Dataset plan

### 2.1 Primary: Multilingual Spoken Words Corpus (MSWC)
- **What it is:** ~340,000 keywords, 23.4M one-second clips, 50 languages, force-aligned word extractions from Mozilla Common Voice, CC-BY 4.0. This is the direct fix for Google Speech Commands' 35-fixed-word limitation — it gives Layer 1 real phonetic/multi-syllable variety instead of a tiny closed vocabulary.
- **Kaggle availability — be honest with the build agent about this:** MSWC is **not** reliably available as a ready-made Kaggle Dataset (I could not confirm an official or well-maintained community mirror). It's officially distributed via **MLCommons** (`https://mlcommons.org/en/multilingual-spoken-words`) and mirrored on **Hugging Face** as `MLCommons/ml_spoken_words`. On Kaggle, the practical path is:
  - Enable internet access on the Kaggle notebook (Settings → Internet → On).
  - `pip install datasets` and pull directly via `datasets.load_dataset("MLCommons/ml_spoken_words", "en", ...)` (repeat per language you want), **or**
  - Download the language-specific archives directly from the MLCommons URL into `/kaggle/working/` at the start of the notebook.
  - Given Kaggle disk/session limits, don't pull all 50 languages — start with English (+ optionally 2-3 more if the wake phrases you're testing aren't English-only) and the "microset" style subset if you want a faster first pass before scaling up.
- **License:** CC-BY 4.0 — fine for both research and commercial use, attribute per the license.

### 2.2 Supplementary: Google Speech Commands v0.02 — keep, it's cheap and directly on Kaggle
- Confirmed available as ready Kaggle Datasets (e.g. search "Speech Commands v0.02" / "Google Speech Commands V2" in Kaggle Datasets — several mirrors exist). Use one of these directly rather than re-downloading from torchaudio, to save Kaggle session time.
- Role: background-noise folder (`_background_noise_`) for noise augmentation, plus a small, clean, English-only supplementary word set. Not the primary training source anymore — MSWC is.

### 2.3 Supplementary: Mozilla Common Voice — optional, for Section 6.4's realistic-coarticulation data
- Confirmed present on Kaggle as multiple community mirrors of various Common Voice versions (search "Common Voice" in Kaggle Datasets) — quality/version varies by mirror, so spot-check whichever one you pick.
- Role: continuous natural sentence-level speech, useful if you want to derive more realistic truncation examples (Section 6.4) than isolated MSWC word clips provide, via forced alignment. Treat as an optional enhancement, not a v1 requirement — MSWC alone is enough to build Tier C.

### 2.4 Do not use for Layer 1 training
- MELD / MUStARD++ / CREMA-D — fine later as supplementary **Stage 1 impostor/calibration** data (naturalistic conversational speech for recall-threshold tuning), but irrelevant to Layer-1 representation learning and irrelevant to the truncation-suppression objective. Don't spend early build effort on these; add them in Section 7 (calibration) once the core pipeline runs.

---

## 3. Config (extend the existing config cell)

```python
# Existing config, keep as-is unless noted
SR = 16000
N_MELS = 40
EMBED_DIM = 128

# NEW in v12
MAX_CLIP_SEC = 1.8          # endpointing ceiling, not a fixed crop target
MIN_CLIP_SEC = 0.4          # sanity floor - reject implausibly short endpointed segments
STAGE1_TEMPORAL_HEAD = "pool"
STAGE2_TEMPORAL_HEAD = "gru"
STAGE1_CHANNELS = (16, 32, 32, 64)   # narrower than Stage 2 - see Section 6.3
STAGE2_CHANNELS = (32, 64, 64, 128)  # existing DSCNNEncoder widths, unchanged

# Tier C — truncation-suppression objective
TRUNCATION_AUX_WEIGHT = 0.3          # weight of the auxiliary loss relative to the
                                       # main prototypical loss - tune empirically,
                                       # start here and watch val accuracy on BOTH
                                       # objectives (Section 6.2)
TRUNCATION_MARGIN = 0.8              # target embedding-space margin between a clip
                                       # and its truncated self (same units as the
                                       # prototypical loss's distance metric)
TRUNCATION_FRACTION_PER_EPISODE = 0.5  # fraction of episode examples that also get
                                         # a synthetic truncated counterpart constructed

MSWC_LANGUAGES = ["en"]              # start with English, expand later
MSWC_MIN_CLIPS_PER_KEYWORD = 20      # filter out keywords with too few clips (per the
                                       # MSWC paper's own recommended filtering practice)
```

---

## 4. Utterance endpointing (shared function — build this before anything else that consumes audio)

```python
def endpoint_utterance(stream_buffer: np.ndarray, sr: int,
                        onset_db: float = -40, offset_db: float = -45,
                        onset_frames: int = 3, offset_frames: int = 8,
                        pad_sec: float = 0.15,
                        max_clip_sec: float = MAX_CLIP_SEC,
                        min_clip_sec: float = MIN_CLIP_SEC) -> np.ndarray | None:
    """
    Frame the buffer at ~10ms hop, compute per-frame dB. Onset = onset_frames
    consecutive frames above onset_db. Offset = offset_frames consecutive frames
    below offset_db AFTER onset has occurred. Slice [onset - pad, offset + pad].
    Pad to at least min_clip_sec (silence pad), cap at max_clip_sec (crop from
    both ends, NOT center, logging a warning - this should be rare).
    Return None if no complete onset+offset pair is present in the buffer yet
    (i.e. still mid-utterance) - caller keeps buffering.
    """
    ...

# Required unit tests before this is used anywhere else:
def _test_endpoint_basic_utterance(): ...       # silence-speech-silence -> correct slice
def _test_endpoint_rejects_pure_noise(): ...     # no onset -> returns None
def _test_endpoint_caps_at_max_clip_sec(): ...   # long speech -> capped, warning logged
def _test_endpoint_pads_short_utterance(): ...   # short speech -> padded to min_clip_sec
```

Use this identically in: Layer-1 truncation-pair construction (Section 6.4 optional realistic variant), enrollment, hard-negative derivation, calibration, and live Stage-2 scoring.

---

## 5. Tier C — truncation-suppression training (the core accuracy fix, build this into Layer 1 from the start)

### 5.1 Concept
Standard episodic prototypical loss (unchanged, keep it) teaches "cluster same word, separate different words." Add a second loss term, computed on a subset of the same batch, that additionally teaches "a truncated/prefix version of a word must NOT collapse onto the full word's embedding."

### 5.2 Constructing truncated pairs (cheap, synthetic, no extra data collection)
For `TRUNCATION_FRACTION_PER_EPISODE` of the clips already sampled into an episode:
1. Take the clip's waveform.
2. Locate a plausible internal cut point. Simplest robust approach for v1: cut at a random point between 60%-90% of the clip's *speech-active* duration (using the same VAD energy logic as endpointing, applied at finer grain to find internal low-energy dips as preferred cut points when available, falling back to a proportional cut if no clear internal dip exists).
3. Produce `wav_truncated = wav[:cut_point]`, re-pad/window it through the normal feature pipeline exactly like any other training example.
4. Keep the pairing `(wav, wav_truncated)` — both come from the encoder in the same forward pass.

### 5.3 Loss
```python
def truncation_margin_loss(embed_full, embed_truncated, margin=TRUNCATION_MARGIN):
    """
    embed_full, embed_truncated: (batch_pairs, EMBED_DIM), L2-normalized.
    Penalize the truncated embedding for being CLOSER than `margin` to the
    full embedding - encourages separation up to the margin, not unbounded
    separation (unbounded push would fight the main prototypical objective,
    which still wants some general acoustic similarity preserved).
    """
    dist = torch.norm(embed_full - embed_truncated, dim=-1)   # or 1 - cosine_sim, pick one
                                                                 # and use the SAME metric as
                                                                 # the profile scoring downstream
    return F.relu(margin - dist).mean()

def combined_loss(support, query, query_labels, trunc_full, trunc_partial, encoder,
                   n_way, n_support, aux_weight=TRUNCATION_AUX_WEIGHT):
    proto_loss, acc = prototypical_loss(support, query, query_labels, encoder, n_way, n_support)
    if trunc_full is not None:
        e_full = encoder(trunc_full.to(device))
        e_part = encoder(trunc_partial.to(device))
        aux_loss = truncation_margin_loss(e_full, e_part)
    else:
        aux_loss = torch.tensor(0.0, device=device)
    return proto_loss + aux_weight * aux_loss, acc, aux_loss.item()
```

### 5.4 Training loop changes
- Log `proto_loss`, `aux_loss`, and `proto_acc` **separately** to TensorBoard every step — you need to watch that the auxiliary objective isn't degrading the main word-discrimination accuracy. If `proto_acc` stalls or drops after adding the aux term, lower `TRUNCATION_AUX_WEIGHT` before anything else.
- Validation: alongside the existing prototypical val accuracy, add a validation metric that specifically measures `mean(dist(embed_full, embed_truncated))` on held-out clips — this is the number that should be trending up (toward/past `TRUNCATION_MARGIN`) over training. This is your direct proxy for "will this model tell 'karthika' apart from 'karthik'" before you ever get to enrollment.

### 5.5 Checkpointing
Same shape as before (`encoder_state`, `optimizer_state`, `epoch`, `rng_state`, etc.) — add `"aux_loss_curve"` and `"truncation_val_margin"` to the saved payload so downstream sections/build agents can sanity check the checkpoint quality without re-running training.

---

## 6. Stage 1 / Stage 2 split, and multi-GPU (2× T4) notes

### 6.1 One training run, two exported models
Train `temporal_head="gru"` at full width (`STAGE2_CHANNELS`) with Tier C — this becomes Stage 2 after enrollment/calibration. It's the accuracy-priority model; it only runs on rare triggers, so its size/latency budget is generous.

### 6.2 Stage 1: prioritize lightweight over the truncation-margin fully solving accuracy
Stage 1 does **not** need to nail the karthik/karthika distinction — that's Stage 2's job. Stage 1's only requirement is high recall (don't miss real triggers) at minimal compute, since it runs continuously. Options, in increasing effort:
- **(a) Simplest:** train a second, narrower model (`STAGE1_CHANNELS`, `temporal_head="pool"`) with the *same* Tier C data/loss, just smaller — reuse the training loop, change the config.
- **(b) Recommended if time allows:** knowledge-distill Stage 1 from the trained Stage 2 model (Stage 2's embeddings as soft targets for Stage 1's training, on top of Stage 1's own prototypical loss) — typically gives a smaller model better recall than training it standalone at the same size.
- Either way, quantize Stage 1 to INT8 (already in the existing ONNX export/quantization section) — this is the model running continuously, so its per-inference cost matters most.

### 6.3 Multi-GPU (2× T4) instructions for the build agent
- T4s have no NVLink on Kaggle — use `torch.nn.parallel.DistributedDataParallel` with NCCL backend if setting up multi-process DDP in a notebook is feasible (Kaggle supports this via `torch.multiprocessing.spawn` or the `%%writefile` + subprocess pattern for a script launched with `torchrun`); if that's too heavy for a notebook workflow, `nn.DataParallel` is an acceptable fallback for this model size (DS-CNN is small — DP's single-process overhead is less punishing here than on a large transformer).
- Use `torch.cuda.amp.autocast()` + `GradScaler` for mixed precision — T4s have real fp16 tensor core throughput gains and 16GB is the binding constraint, so this also lets you push batch/episode size up.
- Episodic training parallelizes naturally: split `N_WAY` classes' support+query construction across the two GPUs per step (or simply double effective batch size via DDP's per-GPU episode sampling) rather than trying to split a single episode's tiny forward pass across devices.
- Increase `num_workers` in the DataLoader (MSWC at scale is I/O-bound on Kaggle's disk, not compute-bound) and use `pin_memory=True`.
- Train Stage 2 (`gru`, full width) first, using both GPUs; Stage 1 (`pool`, narrow) is small enough that a single GPU run afterward is likely fine, freeing the other GPU for the distillation forward pass (option 6.2b) if you go that route.

### 6.4 Optional enhancement (skip in v1 if time-constrained)
If Section 6's synthetic truncation (5.2, proportional-cut heuristic) proves too crude once you look at real validation curves, switch the truncation-pair construction to use Common Voice (Section 2.3) sentence-level clips with forced alignment (Montreal Forced Aligner) to get real word-boundary cuts instead of a proportional heuristic — more realistic coarticulation, better proxy for real truncated utterances. Treat this as a v1.1 improvement, not a blocker for the first working pipeline.

---

## 7. Enrollment, hard-negative derivation, calibration (Stage 2)

Unchanged from the prior spec in substance — restated briefly since this doc should stand alone:

1. Enrollment: 3-5 user clips → `endpoint_utterance()` → embed with Stage 2 encoder → fit mean/PCA-covariance profile (same shape as before).
2. Auto-derive Tier A hard negatives for free from the *same* enrollment clips (truncate at detected internal boundary, same logic as Section 5.2) — this is now a second line of defense on top of Tier C, not the primary fix; keep it anyway since it's free and catches anything Tier C's general training didn't fully cover for this specific phrase.
3. Record enrollment clip duration mean/std for the Stage-2 duration gate.
4. Calibrate threshold against generic impostor pool (Speech Commands + optionally MELD/MUStARD++/CREMA-D as naturalistic negatives, Section 2.4) AND the Tier A hard negatives, taking the stricter of the two (same `calibrate_stage2_threshold()` logic as before).
5. Stage 1 threshold: calibrate separately for high recall against the generic impostor pool only — no hard-negative awareness needed here, Stage 2 is the precision backstop.

---

## 8. Acceptance checks before calling this done

- Truncation validation margin (Section 5.4) trending toward `TRUNCATION_MARGIN` over training, without `proto_acc` degrading meaningfully from a Tier-C-free baseline run (train a quick baseline without Tier C first if you want a clean before/after comparison).
- End-to-end cascade test on real audio: "hey karthika" (or your actual test phrase) passes; "hey karthik," "karthik," and "karthika" alone are all rejected by Stage 2 (score above threshold and/or duration gate fails) — using only the 3-5 original enrollment clips, no manually recorded hard negatives.
- Stage 1 recall: near-zero missed triggers on held-out genuine clips at its loose threshold (false accepts here are expected and fine — Stage 2 cleans them up).
- Stage 1 model size/latency suitable for continuous on-device inference after INT8 quantization (report both, don't assume — measure).
- ONNX parity checks pass independently for both Stage 1 and Stage 2 exports.
