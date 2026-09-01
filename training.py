import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

from config import (SR, WARMUP_EPOCHS, PRETRAIN_EPOCHS, MIN_LR, GRAD_CLIP_NORM,
                    TRUNCATION_AUX_WEIGHT, WAVLM_DISTILL_WEIGHT, 
                    WHISPER_DISTILL_WEIGHT, CTC_LOSS_WEIGHT, MSWC_SPLIT)
from model import WakeWordModel
from dataset import (MSWCTrainingDataset, collate_fn, load_speech_dataset, 
                     load_background_noise_bank)
from losses import truncation_margin_loss, cosine_distillation_loss, supervised_contrastive_loss

def _fast_cache_collate(batch):
    """Minimal CPU collate — only loads and pads raw audio.
    Whisper mel extraction is done on the GPU inside precompute_teacher_features.
    """
    import numpy as np
    import torch
    import torchaudio
    from audio_utils import pad_or_trim
    from config import SR

    target_samples = int(1.2 * SR)
    wavs = []
    for item in batch:
        audio = item["audio"]
        wav = np.array(audio["array"], dtype=np.float32)
        in_sr = audio["sampling_rate"]
        if in_sr != SR:
            wav_t = torch.from_numpy(wav)
            wav = torchaudio.functional.resample(wav_t, in_sr, SR).numpy()
        wavs.append(pad_or_trim(wav, target_samples))

    return torch.tensor(np.stack(wavs), dtype=torch.float16)


def precompute_teacher_features(ds, wavlm_model_name, whisper_model_name, device, batch_size=512,
                                cache_dir: str = "./teacher_cache"):
    """
    Max-speed 1-pass teacher embedding cache.
    Optimizations:
    1. Disk cache keyed by (num_samples, wavlm_model, whisper_model) — skips recomputation on restart.
    2. CPU workers do ONLY audio loading + resampling (very fast).
    3. Whisper mel extraction happens on the GPU via torchaudio (eliminates CPU bottleneck).
    4. DataParallel scales across all available GPUs (e.g. 2x T4 on Kaggle).
    """
    import time
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio.transforms as T
    from torch.utils.data import DataLoader
    from transformers import WavLMModel, WhisperModel
    from config import SR

    # ── Disk cache: build a deterministic key ───────────────────────────────
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        pass  # E.g. Kaggle /kaggle/input/ is read-only, which is fine if reading an existing cache
    _wlm_tag = wavlm_model_name.replace("/", "_")
    _wh_tag  = whisper_model_name.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"teacher_{len(ds)}_{_wlm_tag}_{_wh_tag}.pt")

    if os.path.exists(cache_path):
        print(f"\n💾 Loading cached teacher embeddings from:\n   {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        print(f"✅ Cache loaded — {len(ds):,} samples  "
              f"(wavlm {tuple(data['wavlm'].shape)}, whisper {tuple(data['whisper'].shape)})")
        return data
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n⚡ Pre-extracting Teacher Embeddings (GPU-accelerated mel, no CPU bottleneck)...")
    print(f"  • WavLM  : {wavlm_model_name}")
    print(f"  • Whisper: {whisper_model_name}")
    print(f"  • Samples: {len(ds):,}   Batch: {batch_size}   GPUs: {torch.cuda.device_count()}")

    wavlm = WavLMModel.from_pretrained(wavlm_model_name).to(device).half()
    wavlm.eval()
    for p in wavlm.parameters(): p.requires_grad = False

    whisper = WhisperModel.from_pretrained(whisper_model_name).encoder.to(device).half()
    whisper.eval()
    for p in whisper.parameters(): p.requires_grad = False

    if torch.cuda.device_count() > 1:
        wavlm = nn.DataParallel(wavlm)
        whisper = nn.DataParallel(whisper)

    # GPU mel-spectrogram transform matching Whisper's preprocessing exactly
    # (n_fft=400, hop=160, n_mels=80, SR=16000 → 3000 frames per 30s of audio)
    WHISPER_N_MELS = 80
    WHISPER_N_FFT  = 400
    WHISPER_HOP    = 160
    WHISPER_FRAMES = 3000   # Whisper encoder expects exactly 3000 mel frames
    WHISPER_AUDIO_LEN = WHISPER_FRAMES * WHISPER_HOP  # 480000 = 30s at 16kHz

    whisper_mel_fn = T.MelSpectrogram(
        sample_rate=SR, n_fft=WHISPER_N_FFT, hop_length=WHISPER_HOP,
        n_mels=WHISPER_N_MELS, power=2.0,
        window_fn=torch.hann_window
    ).to(device)

    workers = min(4, os.cpu_count() or 4)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=_fast_cache_collate,
        num_workers=workers,
        pin_memory=True,
        shuffle=False
    )

    wavlm_targets = []
    whisper_targets = []
    num_samples = len(ds)
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, wavs_t in enumerate(loader):
            wavs_t = wavs_t.to(device, non_blocking=True)  # [B, T_short] float16

            # ── WavLM: runs on short (1.2s) audio ──────────────────────────
            w_out = wavlm(wavs_t)
            if hasattr(w_out, 'last_hidden_state'): w_out = w_out.last_hidden_state
            w_out = w_out.mean(dim=1).cpu()
            wavlm_targets.append(w_out)

            # ── Whisper: pad audio to 30s, compute mel ON GPU ──────────────
            wavs_f = wavs_t.float()   # [B, T_short]
            pad_len = WHISPER_AUDIO_LEN - wavs_f.shape[1]
            if pad_len > 0:
                wavs_f = F.pad(wavs_f, (0, pad_len))
            else:
                wavs_f = wavs_f[:, :WHISPER_AUDIO_LEN]

            mel = whisper_mel_fn(wavs_f)   # [B, 80, T_mel]

            # Normalize exactly like HF's WhisperFeatureExtractor
            log_mel = torch.log10(torch.clamp(mel, min=1e-10))
            # Per-sample max normalization
            B = log_mel.shape[0]
            max_vals = log_mel.reshape(B, -1).max(dim=1).values.reshape(B, 1, 1)
            log_mel = torch.maximum(log_mel, max_vals - 8.0)
            log_mel = (log_mel + 4.0) / 4.0

            # Whisper encoder strictly expects [B, 80, 3000]
            if log_mel.shape[2] < WHISPER_FRAMES:
                log_mel = F.pad(log_mel, (0, WHISPER_FRAMES - log_mel.shape[2]))
            else:
                log_mel = log_mel[:, :, :WHISPER_FRAMES]

            wh_out = whisper(log_mel.half())
            if hasattr(wh_out, 'last_hidden_state'): wh_out = wh_out.last_hidden_state
            wh_out = wh_out.mean(dim=1).cpu()
            whisper_targets.append(wh_out)

            samples_done = min((batch_idx + 1) * batch_size, num_samples)
            elapsed = time.time() - t_start
            rate = samples_done / elapsed
            remaining = (num_samples - samples_done) / rate if rate > 0 else 0
            print(f"  ✓ {samples_done:>7,}/{num_samples:,}  "
                  f"[{elapsed:5.0f}s elapsed | ETA {remaining:5.0f}s | "
                  f"{rate:6.1f} samples/s]")

    # Return to FP32 to match expected model/loss output types
    wavlm_all = torch.cat(wavlm_targets, dim=0).float()
    whisper_all = torch.cat(whisper_targets, dim=0).float()

    mem_mb = (wavlm_all.element_size() * wavlm_all.nelement() +
              whisper_all.element_size() * whisper_all.nelement()) / 1e6
    total_elapsed = time.time() - t_start
    print(f"✅ Caching complete — {num_samples:,} samples in {total_elapsed:.1f}s "
          f"({num_samples/total_elapsed:.0f} samples/s) | RAM: {mem_mb:.1f} MB")

    del wavlm, whisper
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    result = {"wavlm": wavlm_all, "whisper": whisper_all}
    print(f"💾 Saving teacher cache to:\n   {cache_path}  ({mem_mb:.1f} MB)")
    torch.save(result, cache_path)
    print(f"✅ Cache saved — future runs will skip recomputation.")

    return result

from collections import defaultdict
import random
from torch.utils.data import Sampler

class WordBalancedBatchSampler(Sampler):
    def __init__(self, ms_dataset, batch_size, samples_per_word=4):
        self.dataset = ms_dataset
        self.batch_size = batch_size
        self.samples_per_word = samples_per_word
        
        print("🔍 Grouping dataset by words for contrastive sampling...")
        self.word_to_indices = defaultdict(list)
        
        # ms_dataset is MSWCTrainingDataset, ms_dataset.dataset is a Subset
        subset = ms_dataset.dataset
        hf_dataset = subset.dataset
        indices = subset.indices
        
        # Pre-fetch words to avoid slow item-by-item loading if possible
        word_key = None
        if hasattr(hf_dataset, "column_names"):
            for key in ["keyword", "word", "label"]:
                if key in hf_dataset.column_names:
                    word_key = key
                    break
        
        if word_key:
            print(f"⚡ Fast columnar extraction using '{word_key}'...")
            all_words = hf_dataset[word_key]
            for i, actual_idx in enumerate(indices):
                w = str(all_words[actual_idx]) if all_words[actual_idx] else "unknown"
                self.word_to_indices[w].append(i)
        else:
            print("⚠️ Warning: Columnar extraction failed, falling back to slow row-by-row extraction...")
            for i, actual_idx in enumerate(indices):
                item = hf_dataset[actual_idx]
                w = item.get("keyword") or item.get("word") or item.get("label")
                w = str(w) if w else "unknown"
                self.word_to_indices[w].append(i)
                
        self.valid_words = [w for w, idxs in self.word_to_indices.items() if len(idxs) >= self.samples_per_word]
        print(f"✅ Found {len(self.valid_words)} distinct words with >= {self.samples_per_word} samples.")

    def __iter__(self):
        if not self.valid_words:
            # Fallback to random if dataset is completely scattered
            for _ in range(len(self)):
                yield random.sample(range(len(self.dataset)), self.batch_size)
            return
            
        word_pool = list(self.valid_words)
        num_batches = len(self)
        
        for _ in range(num_batches):
            random.shuffle(word_pool)
            batch = []
            
            # Keep pulling words until batch is full
            for word in word_pool:
                idxs = random.sample(self.word_to_indices[word], self.samples_per_word)
                batch.extend(idxs)
                if len(batch) >= self.batch_size:
                    break
                    
            random.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self):
        return len(self.dataset) // self.batch_size

def run_training(
    noise_dir: str = "./noise",
    output_dir: str = "./output",
    dataset_path: str = None,
    resume_path: str = None,
    mswc_split: str = "train[:10000]",
    max_train_samples: int = None,
    pretrain_epochs: int = PRETRAIN_EPOCHS,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    wavlm_model: str = "microsoft/wavlm-large",
    whisper_model: str = "openai/whisper-base",
    cache_teachers: bool = True,
    teacher_cache_dir: str = "./teacher_cache"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    ds = load_speech_dataset(mswc_split, dataset_path=dataset_path)
    noise_bank = load_background_noise_bank(noise_dir)
    
    teacher_targets = None
    if cache_teachers:
        teacher_targets = precompute_teacher_features(ds, wavlm_model, whisper_model, device, batch_size=512,
                                                      cache_dir=teacher_cache_dir)
    
    # Slice dataset AFTER loading the cache (cache key uses full len(ds))
    # This lets you train on a subset while still reusing the full precomputed cache.
    if max_train_samples is not None and max_train_samples < len(ds):
        print(f"📦 Slicing dataset: using first {max_train_samples:,} of {len(ds):,} samples")
        ds = ds.select(range(max_train_samples))
        if teacher_targets is not None:
            teacher_targets = {k: v[:max_train_samples] for k, v in teacher_targets.items()}
    
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    torch.backends.cudnn.benchmark = True
    
    train_dataset = MSWCTrainingDataset(train_ds, noise_clips=noise_bank, teacher_targets=teacher_targets, indices=train_ds.indices if teacher_targets else None)
    train_sampler = WordBalancedBatchSampler(train_dataset, batch_size=batch_size, samples_per_word=4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_dataset = MSWCTrainingDataset(val_ds, noise_clips=noise_bank, teacher_targets=teacher_targets, indices=val_ds.indices if teacher_targets else None)
    val_sampler = WordBalancedBatchSampler(val_dataset, batch_size=batch_size, samples_per_word=4)
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    student = WakeWordModel().to(device)
    
    wavlm, whisper, whisper_extractor = None, None, None
    if not cache_teachers:
        from transformers import WavLMModel, WhisperModel, WhisperFeatureExtractor
        wavlm = WavLMModel.from_pretrained(wavlm_model).to(device)
        wavlm.eval()
        for p in wavlm.parameters(): p.requires_grad = False
        
        whisper_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
        whisper = WhisperModel.from_pretrained(whisper_model).encoder.to(device)
        whisper.eval()
        for p in whisper.parameters(): p.requires_grad = False
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        prog = float(epoch - WARMUP_EPOCHS) / float(max(1, epochs - WARMUP_EPOCHS))
        return (MIN_LR/lr) + (1.0 - (MIN_LR/lr)) * 0.5 * (1.0 + math.cos(math.pi * prog))
        
    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_loss = float("inf")
    patience = 0
    start_epoch = 0

    target_resume = resume_path or os.path.join(output_dir, "best_sota_wakeword_model.pt")
    if os.path.exists(target_resume):
        checkpoint = torch.load(target_resume, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'val_metric' in checkpoint:
            best_val_loss = checkpoint['val_metric']
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Resumed from checkpoint: Epoch {start_epoch-1} (Val Loss: {best_val_loss:.4f})")
    
    # Temperature warm-up schedule for SupCon: high temp early → sharp temp later
    SUPCON_INIT_TEMP  = 0.5   # Start warm (easy gradients, loose clustering)
    SUPCON_FINAL_TEMP = 0.07  # End sharp (tight clustering)
    SUPCON_WEIGHT     = 0.5   # Down-weight so SupCon doesn't drown distillation
    post_pretrain_epochs = max(1, epochs - pretrain_epochs)

    for epoch in range(start_epoch, epochs):
        student.train()
        is_pretrain = (epoch < pretrain_epochs)
        if epoch == pretrain_epochs:
            print("\n🚀 Pre-training complete! Enabling Supervised Contrastive & Truncation losses...\n")
            # Reset early stopping metrics because the loss landscape completely changes
            best_val_loss = float("inf")
            patience = 0
        total_epoch_loss = 0.0

        steps_per_epoch = len(train_loader)
        # Fraction of post-pretrain training completed (0 → 1)
        post_pretrain_step_offset = max(0, epoch - pretrain_epochs) * steps_per_epoch

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            mels = batch["mels"].to(device, non_blocking=True)
            trunc_mels = batch["trunc_mels"].to(device, non_blocking=True)
            wavs_np = np.stack(batch["wavs"], axis=0)
            wavs_t = torch.tensor(wavs_np, dtype=torch.float32, device=device)
            
            energy = mels.mean(dim=-1).squeeze(1)
            attention_mask = energy > -10.0
            
            if "wavlm_targets" in batch:
                wavlm_target = batch["wavlm_targets"].to(device, non_blocking=True)
                whisper_target = batch["whisper_targets"].to(device, non_blocking=True)
            else:
                with torch.no_grad(), autocast('cuda'):
                    wavlm_out = wavlm(wavs_t).last_hidden_state
                    wavlm_target = wavlm_out.mean(dim=1)
                    whisper_inputs = whisper_extractor(
                        [w for w in wavs_np], sampling_rate=SR, return_tensors="pt"
                    ).input_features.to(device, non_blocking=True)
                    whisper_out = whisper(whisper_inputs).last_hidden_state
                    whisper_target = whisper_out.mean(dim=1)
            
            with autocast('cuda'):
                norm_embed, (s_wavlm_proj, s_whisper_proj), ctc_logits = student(mels, mask=attention_mask, return_distill=True)
                trunc_embed = student(trunc_mels, mask=attention_mask)
                
                loss_wavlm = cosine_distillation_loss(s_wavlm_proj, wavlm_target)
                loss_whisper = cosine_distillation_loss(s_whisper_proj, whisper_target)
                
                input_lengths = torch.full((mels.size(0),), ctc_logits.size(1), dtype=torch.long, device=device)
                loss_ctc = ctc_loss_fn(
                    ctc_logits.transpose(0, 1), batch["targets"].to(device), 
                    input_lengths, batch["target_lengths"].to(device)
                )
                
                loss_tier_c = truncation_margin_loss(norm_embed, trunc_embed)

                # Compute warmup_frac for temperature schedule
                warmup_frac = min(1.0, (post_pretrain_step_offset + step) /
                                  (post_pretrain_epochs * steps_per_epoch))
                loss_supcon, p_acc = supervised_contrastive_loss(
                    norm_embed, batch["words"],
                    temperature=SUPCON_FINAL_TEMP,
                    init_temperature=SUPCON_INIT_TEMP,
                    warmup_frac=warmup_frac,
                )
                
                if is_pretrain:
                    loss = (WAVLM_DISTILL_WEIGHT * loss_wavlm) + (WHISPER_DISTILL_WEIGHT * loss_whisper) + (CTC_LOSS_WEIGHT * loss_ctc)
                else:
                    loss = (SUPCON_WEIGHT * loss_supcon +
                            (TRUNCATION_AUX_WEIGHT * loss_tier_c) + 
                            (WAVLM_DISTILL_WEIGHT * loss_wavlm) + 
                            (WHISPER_DISTILL_WEIGHT * loss_whisper) + 
                            (CTC_LOSS_WEIGHT * loss_ctc))
                            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            total_epoch_loss += loss.item()
            
            if step % 25 == 0:
                cur_temp = SUPCON_INIT_TEMP + (SUPCON_FINAL_TEMP - SUPCON_INIT_TEMP) * warmup_frac if not is_pretrain else SUPCON_FINAL_TEMP
                print(f"Step {step:03d} | Loss: {loss.item():.4f} | Distill: {(loss_wavlm+loss_whisper).item():.3f} | TierC: {loss_tier_c.item():.3f} | Proto: {loss_supcon.item():.3f} | Temp: {cur_temp:.3f}")
                
        scheduler.step()
        avg_train_loss = total_epoch_loss / len(train_loader)
        
        student.eval()
        val_trunc_loss = 0.0
        val_supcon_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for v_batch in val_loader:
                v_mels = v_batch["mels"].to(device)
                v_trunc = v_batch["trunc_mels"].to(device)
                v_energy = v_mels.mean(dim=-1).squeeze(1)
                v_mask = v_energy > -10.0
                
                v_embed = student(v_mels, mask=v_mask)
                v_trunc_embed = student(v_trunc, mask=v_mask)
                
                val_trunc_loss += truncation_margin_loss(v_embed, v_trunc_embed).item()
                val_warmup_frac = min(1.0, (max(0, epoch + 1 - pretrain_epochs) * steps_per_epoch) /
                                      max(1, post_pretrain_epochs * steps_per_epoch))
                p_loss, _ = supervised_contrastive_loss(
                    v_embed, v_batch["words"],
                    temperature=SUPCON_FINAL_TEMP,
                    init_temperature=SUPCON_INIT_TEMP,
                    warmup_frac=val_warmup_frac
                )
                val_supcon_loss += p_loss.item()
                val_batches += 1
                
        if val_batches > 0:
            val_trunc_loss /= val_batches
            val_supcon_loss /= val_batches
        
        # Weight the validation metric exactly as they are weighted in training
        val_combined = (TRUNCATION_AUX_WEIGHT * val_trunc_loss) + (SUPCON_WEIGHT * val_supcon_loss)
        
        print(f"Epoch {epoch+1} Done | Train Loss: {avg_train_loss:.4f} | Val Trunc: {val_trunc_loss:.4f} | Val SupCon: {val_supcon_loss:.4f} | Val Combined: {val_combined:.4f}")
        
        if val_combined < best_val_loss:
            best_val_loss = val_combined
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metric': best_val_loss
            }, os.path.join(output_dir, "best_sota_wakeword_model.pt"))
        else:
            if not is_pretrain:
                patience += 1
                if patience >= early_stopping_patience:
                    break
                    
    print("Exporting ONNX...")
    student.load_state_dict(torch.load(os.path.join(output_dir, "best_sota_wakeword_model.pt"), map_location=device)['model_state_dict'])
    student.eval()
    
    dummy_input = torch.randn(1, 1, int(1.2 * SR / 160), 40).to(device)
    onnx_path = os.path.join(output_dir, "wakeword_student.onnx")
    
    try:
        torch.onnx.export(
            student, dummy_input, onnx_path,
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=['input_mel'], output_names=['embedding'],
            dynamic_axes={'input_mel': {0: 'batch_size', 2: 'time'}, 'embedding': {0: 'batch_size'}}
        )
        print(f"ONNX Exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX Export failed: {e}")

    # Export teacher cache to output dir if it exists
    if cache_teachers:
        import shutil
        _wlm_tag = wavlm_model.replace("/", "_")
        _wh_tag  = whisper_model.replace("/", "_")
        cache_file = f"teacher_{len(ds)}_{_wlm_tag}_{_wh_tag}.pt"
        src_cache = os.path.join(teacher_cache_dir, cache_file)
        if os.path.exists(src_cache):
            dst_cache = os.path.join(output_dir, cache_file)
            print(f"Copying teacher cache to output directory: {dst_cache}")
            shutil.copy2(src_cache, dst_cache)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_dir", type=str, default="./noise")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--mswc_split", type=str, default=MSWC_SPLIT)
    parser.add_argument("--pretrain_epochs", type=int, default=PRETRAIN_EPOCHS, help="Number of teacher-only pre-training epochs before enabling SupCon/TierC loss.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--wavlm_model", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-base")
    parser.add_argument("--no_cache_teachers", action="store_true")
    parser.add_argument("--teacher_cache_dir", type=str, default="./teacher_cache",
                        help="Directory to store/load precomputed teacher embeddings on disk.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Train on only the first N samples from the full dataset split, "
                             "while still reusing a larger precomputed teacher cache.")
    
    args = parser.parse_args()
    run_training(
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        resume_path=args.resume_path,
        mswc_split=args.mswc_split,
        max_train_samples=args.max_train_samples,
        pretrain_epochs=args.pretrain_epochs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.early_stopping_patience,
        wavlm_model=args.wavlm_model,
        whisper_model=args.whisper_model,
        cache_teachers=not args.no_cache_teachers,
        teacher_cache_dir=args.teacher_cache_dir,
    )
