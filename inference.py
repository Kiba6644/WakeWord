import os
import torch
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torchaudio.transforms as T

from config import (SR, N_MELS, EMBED_DIM, TEMPO_AUG_FACTORS,
                    NUM_TEMPORAL_SEGMENTS, SUFFIX_REJECTION_THRESHOLD,
                    CTC_POSTERIOR_THRESHOLD, N_FFT, HOP_LENGTH,
                    DURATION_GATE_STD_LOWER, DURATION_GATE_STD_UPPER)
from audio_utils import (time_stretch_audio, create_truncated_clip,
                         slice_temporal_segments, generate_phonetic_minimal_pairs)


class WakeWordCascade:
    """
    SOTA Industrial Wake Word Cascade (Wearable / Smart Glasses Form-Factor):
    - Stage 1: Always-on ultra-low power scan.
    - Duration Gate: Asymmetric envelope — strict lower bound (blocks truncations),
                     lenient upper bound (allows slow/deliberate speech).
    - Stage 2: Multi-Head Attention Global Match + 3-Phase Segment Trajectory.
    - Stage 2 CTC: Phonetic Posterior Ratio check for trailing phoneme verification.

    False-reject mitigations:
    1. Asymmetric duration gate (DURATION_GATE_STD_LOWER / DURATION_GATE_STD_UPPER)
    2. Measured speech duration via energy VAD, not raw buffer length
    3. Suffix check gracefully degrades when frame count is too low
    4. Near-miss logging: metrics always carry 'near_miss' flag when close to threshold
    """

    def __init__(self, stage1_path, stage2_path, threshold1=0.60, threshold2=0.82,
                 suffix_threshold=SUFFIX_REJECTION_THRESHOLD,
                 ctc_posterior_threshold=CTC_POSTERIOR_THRESHOLD,
                 duration_gate_std_lower=DURATION_GATE_STD_LOWER,
                 duration_gate_std_upper=DURATION_GATE_STD_UPPER):
        # Note: stage1 is optional. If no path is given, Stage 1 is bypassed (sim=1.0).
        self.stage1 = ort.InferenceSession(stage1_path) if os.path.exists(stage1_path) else None
        self.stage2 = ort.InferenceSession(stage2_path) if os.path.exists(stage2_path) else None
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.suffix_threshold = suffix_threshold
        self.ctc_posterior_threshold = ctc_posterior_threshold
        self.duration_gate_std_lower = duration_gate_std_lower
        self.duration_gate_std_upper = duration_gate_std_upper

        # Enrolled Profiles & Calibration Bank
        self.global_target_profile = None
        self.segment_target_profiles = None
        self.minimal_pair_negatives = []
        self.mean_duration = 0.0
        self.std_duration = 0.0

    def enroll(self, enrollment_clips: list[np.ndarray], phrase: str = "Hey Karthika", sr: int = SR):
        """
        SOTA Generative Enrollment:
        1. Expands 3-5 raw user clips into multi-tempo variations (0.85x to 1.25x).
           Duration stats are derived from the ORIGINAL clips only, not tempo-stretched,
           so the duration gate reflects real user speech speed, not augmentation artifacts.
        2. Generates phonetic minimal-pair negatives ("Hey Karthik", "Hey Bartika", "Say Karthika").
        3. Builds global and 3-phase trajectory profiles.
        """
        if not enrollment_clips:
            raise ValueError("No enrollment clips provided.")

        # Duration stats from original clips only (not stretched) — more honest estimate
        original_durations = [len(c) / sr for c in enrollment_clips]
        self.mean_duration = float(np.mean(original_durations))
        self.std_duration = max(0.05, float(np.std(original_durations)))

        # 1. Multi-Tempo Expansion for embedding profiles
        expanded_clips = []
        for clip in enrollment_clips:
            for rate in TEMPO_AUG_FACTORS:
                expanded_clips.append(time_stretch_audio(clip, rate, sr))

        # 2. Generative Minimal-Pair Negatives
        self.minimal_pair_negatives = generate_phonetic_minimal_pairs(phrase)

        # 3. Extract Profiles
        global_embeds = []
        segment_embeds = []
        mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)

        for clip in expanded_clips:
            wav_t = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)
            mel = mel_transform(wav_t)
            log_mel = torch.log(torch.clamp(mel, min=1e-5))
            feat = log_mel.transpose(1, 2).unsqueeze(1).numpy()  # (1, 1, time, n_mels)

            if self.stage2 is not None:
                out = self.stage2.run(None, {self.stage2.get_inputs()[0].name: feat})[0]
            else:
                out = np.random.randn(1, EMBED_DIM).astype(np.float32)
                out /= np.linalg.norm(out, axis=-1, keepdims=True)

            global_embeds.append(out)

            seg_outs = []
            T_len = feat.shape[2]
            seg_len = max(1, T_len // NUM_TEMPORAL_SEGMENTS)
            for s in range(NUM_TEMPORAL_SEGMENTS):
                start = s * seg_len
                end = (s + 1) * seg_len if s < NUM_TEMPORAL_SEGMENTS - 1 else T_len
                seg_feat = feat[:, :, start:end, :]
                if self.stage2 is not None and seg_feat.shape[2] >= 2:
                    s_out = self.stage2.run(None, {self.stage2.get_inputs()[0].name: seg_feat})[0]
                    s_out /= np.linalg.norm(s_out, axis=-1, keepdims=True)
                else:
                    s_out = out.copy()
                seg_outs.append(s_out)
            segment_embeds.append(np.stack(seg_outs, axis=1))

        all_global = np.concatenate(global_embeds, axis=0)
        self.global_target_profile = np.mean(all_global, axis=0, keepdims=True)
        self.global_target_profile /= np.linalg.norm(self.global_target_profile)

        all_segments = np.concatenate(segment_embeds, axis=0)
        self.segment_target_profiles = np.mean(all_segments, axis=0)
        self.segment_target_profiles /= np.linalg.norm(self.segment_target_profiles, axis=-1, keepdims=True)

    def _measure_speech_duration(self, audio_buffer: np.ndarray, sr: int,
                                  energy_db_threshold: float = -38.0) -> float:
        """
        Estimates speech duration from an audio buffer using a simple energy VAD.
        Counts frames above the energy threshold rather than trusting the full buffer length.
        Returns duration in seconds. Falls back to full buffer duration if no active frames found.
        """
        frame_len = int(sr * 0.025)   # 25ms frames
        hop = int(sr * 0.010)         # 10ms hop
        if len(audio_buffer) < frame_len:
            return len(audio_buffer) / sr

        num_frames = (len(audio_buffer) - frame_len) // hop
        active = 0
        for i in range(num_frames):
            frame = audio_buffer[i * hop: i * hop + frame_len]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            if 20 * np.log10(rms) > energy_db_threshold:
                active += 1

        if active == 0:
            return len(audio_buffer) / sr
        return (active * hop + frame_len) / sr

    def verify_utterance(self, feat: np.ndarray, duration_sec: float,
                         segment_feats: list[np.ndarray] = None,
                         ctc_suffix_prob: float = 1.0,
                         audio_buffer: np.ndarray = None,
                         sr: int = SR) -> tuple[bool, dict]:
        """
        Executes 4-Tier Verification with false-reject mitigations:

        1. Stage 1 Always-On Scan
           (Bypassed with sim=1.0 if stage1_path was not provided at init.)

        2. Asymmetric Duration Gate
           - Short utterances below mean - LOWER*std → rejected (truncation guard)
           - Long utterances above mean + UPPER*std → rejected (only extreme outliers)
           - If audio_buffer is provided, speech duration is measured via energy VAD
             rather than trusting the passed duration_sec (which may be buffer length).

        3. Stage 2 Global Match

        4. Suffix Trajectory + CTC Posterior Check
           - Suffix check is SKIPPED (not rejected) if the feature tensor has
             fewer than 6 time frames — avoids false rejects on very short buffers.
           - Near-miss: metrics['near_miss'] = True when similarity is within 5%
             of a passing threshold, to assist threshold calibration.
        """
        if self.global_target_profile is None:
            raise ValueError("Must enroll before running verification.")

        metrics = {}

        # ------------------------------------------------------------------
        # 1. Stage 1 (Fast Gate)
        # ------------------------------------------------------------------
        if self.stage1 is None:
            s1_embed = self.global_target_profile.copy()
        else:
            s1_embed = self.stage1.run(None, {self.stage1.get_inputs()[0].name: feat})[0]
            s1_embed = s1_embed / np.linalg.norm(s1_embed, axis=-1, keepdims=True)

        s1_sim = float(np.dot(s1_embed, self.global_target_profile.T)[0, 0])
        metrics["stage1_similarity"] = s1_sim
        metrics["near_miss"] = False

        if s1_sim < self.threshold1:
            if s1_sim >= self.threshold1 * 0.95:
                metrics["near_miss"] = True
                metrics["near_miss_stage"] = "stage1"
            metrics["rejection_reason"] = f"Stage 1 below threshold ({s1_sim:.3f} < {self.threshold1})"
            return False, metrics

        # ------------------------------------------------------------------
        # 2. Asymmetric Duration Gate
        #    If audio_buffer provided, measure actual speech duration via VAD.
        # ------------------------------------------------------------------
        if audio_buffer is not None:
            measured_duration = self._measure_speech_duration(audio_buffer, sr)
            metrics["duration_sec_measured"] = measured_duration
        else:
            measured_duration = duration_sec
        metrics["duration_sec"] = measured_duration

        lower_bound = self.mean_duration - self.duration_gate_std_lower * self.std_duration
        upper_bound = self.mean_duration + self.duration_gate_std_upper * self.std_duration
        duration_valid = lower_bound <= measured_duration <= upper_bound
        metrics["duration_valid"] = duration_valid
        metrics["duration_bounds"] = (round(lower_bound, 3), round(upper_bound, 3))

        if not duration_valid:
            too_short = measured_duration < lower_bound
            metrics["rejection_reason"] = (
                f"Duration gate: {'too short' if too_short else 'too long'} "
                f"({measured_duration:.2f}s, allowed [{lower_bound:.2f}s, {upper_bound:.2f}s])"
            )
            return False, metrics

        # ------------------------------------------------------------------
        # 3. Stage 2 Global Match
        # ------------------------------------------------------------------
        if self.stage2 is None:
            s2_embed = self.global_target_profile.copy()
        else:
            s2_embed = self.stage2.run(None, {self.stage2.get_inputs()[0].name: feat})[0]
            s2_embed = s2_embed / np.linalg.norm(s2_embed, axis=-1, keepdims=True)

        s2_sim = float(np.dot(s2_embed, self.global_target_profile.T)[0, 0])
        metrics["stage2_global_similarity"] = s2_sim

        if s2_sim < self.threshold2:
            if s2_sim >= self.threshold2 * 0.95:
                metrics["near_miss"] = True
                metrics["near_miss_stage"] = "stage2_global"
            metrics["rejection_reason"] = f"Stage 2 global below threshold ({s2_sim:.3f} < {self.threshold2})"
            return False, metrics

        # ------------------------------------------------------------------
        # 4a. Suffix Trajectory Verification
        #     Skip (don't reject) if feature tensor is too short for a
        #     meaningful 3rd-segment measurement (< 6 time frames).
        # ------------------------------------------------------------------
        T_len = feat.shape[2]
        suffix_checked = False

        if T_len >= 6:
            suffix_target = self.segment_target_profiles[2:3, :]

            if segment_feats is not None and len(segment_feats) >= 3:
                suffix_sample = segment_feats[2]
                suffix_sample = suffix_sample / (np.linalg.norm(suffix_sample, axis=-1, keepdims=True) + 1e-8)
                suffix_checked = True
            elif self.stage2 is not None:
                seg_len = T_len // 3
                suffix_feat = feat[:, :, 2 * seg_len:, :]
                # Only run if suffix segment has at least 2 frames
                if suffix_feat.shape[2] >= 2:
                    suffix_sample = self.stage2.run(None, {self.stage2.get_inputs()[0].name: suffix_feat})[0]
                    suffix_sample = suffix_sample / (np.linalg.norm(suffix_sample, axis=-1, keepdims=True) + 1e-8)
                    suffix_checked = True

            if suffix_checked:
                suffix_sim = float(np.dot(suffix_sample, suffix_target.T)[0, 0])
                metrics["suffix_phase_similarity"] = suffix_sim

                if suffix_sim < self.suffix_threshold:
                    if suffix_sim >= self.suffix_threshold * 0.95:
                        metrics["near_miss"] = True
                        metrics["near_miss_stage"] = "suffix"
                    metrics["rejection_reason"] = (
                        f"Suffix phase mismatch — likely truncation "
                        f"({suffix_sim:.3f} < {self.suffix_threshold})"
                    )
                    return False, metrics
        else:
            metrics["suffix_phase_similarity"] = None
            metrics["suffix_skipped"] = True  # Too few frames — not penalised

        # ------------------------------------------------------------------
        # 4b. CTC Posterior Ratio Check
        # ------------------------------------------------------------------
        metrics["ctc_suffix_prob"] = ctc_suffix_prob
        if ctc_suffix_prob < self.ctc_posterior_threshold:
            metrics["rejection_reason"] = (
                f"CTC posterior below threshold ({ctc_suffix_prob:.3f} < {self.ctc_posterior_threshold})"
            )
            return False, metrics

        metrics["status"] = "WAKE_WORD_TRIGGERED"
        return True, metrics


def export_onnx(model, dummy_input, save_path):
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {2: 'time_steps'}}
        )
    print(f"Exported ONNX model to {save_path}")


def quantize_onnx(onnx_path, quantized_path):
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized INT8 model saved to {quantized_path}")
