import os
import torch
import torch.nn.functional as F
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import torchaudio.transforms as T

from .config import (SR, N_MELS, EMBED_DIM, TEMPO_AUG_FACTORS,
                    NUM_TEMPORAL_SEGMENTS, SUFFIX_REJECTION_THRESHOLD,
                    CTC_POSTERIOR_THRESHOLD, N_FFT, HOP_LENGTH,
                    DURATION_GATE_STD_LOWER, DURATION_GATE_STD_UPPER)
from .audio_utils import (time_stretch_audio, create_truncated_clip,
                         generate_phonetic_minimal_pairs)


class WakeWordCascade:
    def __init__(self, stage1_path, stage2_path, threshold1=0.60, threshold2=0.82,
                 suffix_threshold=SUFFIX_REJECTION_THRESHOLD,
                 ctc_posterior_threshold=CTC_POSTERIOR_THRESHOLD,
                 duration_gate_std_lower=DURATION_GATE_STD_LOWER,
                 duration_gate_std_upper=DURATION_GATE_STD_UPPER):
        self.stage1 = ort.InferenceSession(stage1_path) if os.path.exists(stage1_path) else None
        self.stage2 = ort.InferenceSession(stage2_path) if os.path.exists(stage2_path) else None
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.suffix_threshold = suffix_threshold
        self.ctc_posterior_threshold = ctc_posterior_threshold
        self.duration_gate_std_lower = duration_gate_std_lower
        self.duration_gate_std_upper = duration_gate_std_upper

        self.global_target_profile = None
        self.suffix_target_profile = None
        self.minimal_pair_negatives = []
        self.mean_duration = 0.0
        self.std_duration = 0.0

    def enroll(self, enrollment_clips: list[np.ndarray], phrase: str = "Hey Karthika", sr: int = SR):
        if not enrollment_clips:
            raise ValueError("No enrollment clips provided.")

        original_durations = [len(c) / sr for c in enrollment_clips]
        self.mean_duration = float(np.mean(original_durations))
        self.std_duration = max(0.05, float(np.std(original_durations)))

        expanded_clips = []
        for clip in enrollment_clips:
            for rate in TEMPO_AUG_FACTORS:
                expanded_clips.append((rate, time_stretch_audio(clip, rate, sr)))

        self.minimal_pair_negatives = generate_phonetic_minimal_pairs(phrase)

        global_embeds_natural = []
        suffix_embeds_all = []
        mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0)

        for rate, clip in expanded_clips:
            wav_t = torch.tensor(clip, dtype=torch.float32).unsqueeze(0)
            mel = mel_transform(wav_t)
            log_mel = torch.log(torch.clamp(mel, min=1e-5))
            # Per-utterance CMVN
            log_mel = (log_mel - log_mel.mean(dim=-1, keepdim=True)) / (log_mel.std(dim=-1, keepdim=True) + 1e-5)
            feat = log_mel.transpose(1, 2).unsqueeze(1).numpy()  # (1, 1, time, n_mels)

            if self.stage2 is not None:
                outs = self.stage2.run(None, {self.stage2.get_inputs()[0].name: feat})
                g_out = outs[0]
                s_out = outs[1]
            else:
                g_out = np.random.randn(1, EMBED_DIM).astype(np.float32)
                g_out /= np.linalg.norm(g_out, axis=-1, keepdims=True)
                s_out = g_out.copy()

            if rate == 1.0:
                global_embeds_natural.append(g_out)
            suffix_embeds_all.append(s_out)

        all_global = np.concatenate(global_embeds_natural, axis=0)
        self.global_target_profile = np.mean(all_global, axis=0, keepdims=True)
        self.global_target_profile /= np.linalg.norm(self.global_target_profile)

        all_suffixes = np.concatenate(suffix_embeds_all, axis=0)
        self.suffix_target_profile = np.mean(all_suffixes, axis=0, keepdims=True)
        self.suffix_target_profile /= np.linalg.norm(self.suffix_target_profile)

    def _measure_speech_duration(self, audio_buffer: np.ndarray, sr: int,
                                  energy_db_threshold: float = -38.0) -> float:
        frame_len = int(sr * 0.025)
        hop = int(sr * 0.010)
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
                         ctc_suffix_prob: float = 1.0,
                         audio_buffer: np.ndarray = None,
                         sr: int = SR) -> tuple[bool, dict]:
        if self.global_target_profile is None:
            raise ValueError("Must enroll before running verification.")

        metrics = {}

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

        if audio_buffer is not None:
            measured_duration = self._measure_speech_duration(audio_buffer, sr)
            metrics["duration_sec_measured"] = measured_duration
        else:
            measured_duration = duration_sec
        metrics["duration_sec"] = measured_duration

        lower_bound = max(0.25, self.mean_duration - self.duration_gate_std_lower * self.std_duration)
        upper_bound = min(2.00, self.mean_duration + self.duration_gate_std_upper * self.std_duration)
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

        if self.stage2 is None:
            s2_embed = self.global_target_profile.copy()
            suffix_embed = self.suffix_target_profile.copy()
        else:
            outs = self.stage2.run(None, {self.stage2.get_inputs()[0].name: feat})
            s2_embed = outs[0] / np.linalg.norm(outs[0], axis=-1, keepdims=True)
            suffix_embed = outs[1] / np.linalg.norm(outs[1], axis=-1, keepdims=True)

        s2_sim = float(np.dot(s2_embed, self.global_target_profile.T)[0, 0])
        metrics["stage2_global_similarity"] = s2_sim

        if s2_sim < self.threshold2:
            if s2_sim >= self.threshold2 * 0.95:
                metrics["near_miss"] = True
                metrics["near_miss_stage"] = "stage2_global"
            metrics["rejection_reason"] = f"Stage 2 global below threshold ({s2_sim:.3f} < {self.threshold2})"
            return False, metrics

        suffix_sim = float(np.dot(suffix_embed, self.suffix_target_profile.T)[0, 0])
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
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_mel'],
            output_names=['embedding', 'suffix_embed', 'ctc_logits'],
            dynamic_axes={'input_mel': {0: 'batch_size', 2: 'time'}, 
                          'embedding': {0: 'batch_size'},
                          'suffix_embed': {0: 'batch_size'},
                          'ctc_logits': {0: 'batch_size', 1: 'time'}}
        )
    print(f"Exported ONNX model to {save_path}")

def quantize_onnx(onnx_path, quantized_path):
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized INT8 model saved to {quantized_path}")
