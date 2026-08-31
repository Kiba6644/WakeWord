import torch
import numpy as np
import pytest

torch.manual_seed(42)
np.random.seed(42)
from model import WakeWordModel, DSCNNEncoder, MultiHeadAttentionPooling
from train import truncation_margin_loss, cosine_distillation_loss
from inference import WakeWordCascade
from config import SR, NUM_PHONEMES

def test_model_forward_dual_distillation_and_ctc():
    model = WakeWordModel(channels=(32, 64, 64, 128), temporal_head="attention", embed_dim=128)
    dummy_input = torch.randn(2, 1, 100, 40)
    
    # 1. Standard forward pass
    out = model(dummy_input)
    assert out.shape == (2, 128)
    
    # 2. Dual Distillation forward pass
    norm_embed, (wavlm_proj, whisper_proj), ctc_logits = model(dummy_input, return_distill=True)
    assert norm_embed.shape == (2, 128)
    assert wavlm_proj.shape == (2, 1024)
    assert whisper_proj.shape == (2, 512)
    assert ctc_logits.shape[-1] == NUM_PHONEMES
    
    # 3. CTC logits only
    _, ctc_out = model(dummy_input, return_ctc=True)
    assert ctc_out.ndim == 3 # (batch, time, num_phonemes)

def test_model_forward_segments():
    model = WakeWordModel(channels=(32, 64, 64, 128), temporal_head="attention", embed_dim=128)
    dummy_input = torch.randn(2, 1, 100, 40)
    seg_out = model.forward_segments(dummy_input, num_segments=3)
    assert seg_out.shape == (2, 3, 128)

def test_truncation_margin_loss():
    embed_full = torch.randn(4, 128)
    embed_full = torch.nn.functional.normalize(embed_full, p=2, dim=-1)
    loss = truncation_margin_loss(embed_full, embed_full, margin=0.8)
    assert torch.isclose(loss, torch.tensor(0.8), atol=1e-3)

def test_cosine_distillation_loss():
    student_proj = torch.randn(4, 1024)
    loss = cosine_distillation_loss(student_proj, student_proj)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

def test_model_forward_gru():
    model = WakeWordModel(channels=(32, 64, 64, 128), temporal_head="gru", embed_dim=128)
    dummy_input = torch.randn(2, 1, 100, 40)
    out = model(dummy_input)
    assert out.shape == (2, 128)
    seg_out = model.forward_segments(dummy_input, num_segments=3)
    assert seg_out.shape == (2, 3, 128)

class DummyStage2:
    def get_inputs(self):
        class Input:
            name = 'input'
        return [Input()]
    def run(self, *args, **kwargs):
        return [np.ones((1, 128), dtype=np.float32)]

def test_cascade_enrollment_and_verification_with_ctc():
    cascade = WakeWordCascade(stage1_path="", stage2_path="", threshold1=0.5, threshold2=0.5,
                              suffix_threshold=0.62, ctc_posterior_threshold=0.70,
                              duration_gate_std_lower=2.0, duration_gate_std_upper=3.5)
    cascade.stage2 = DummyStage2()

    # 1. Generative Enrollment
    raw_clips = [np.random.randn(int(SR * 1.2)).astype(np.float32) for _ in range(3)]
    cascade.enroll(raw_clips, phrase="Hey Karthika", sr=SR)

    assert cascade.global_target_profile is not None
    assert len(cascade.minimal_pair_negatives) > 0
    assert any("karthik" in p.lower() for p in cascade.minimal_pair_negatives)

    # Duration stats come from original clips (all 1.2s), not stretched clips
    assert abs(cascade.mean_duration - 1.2) < 0.05

    # 2. Genuine matching utterance with audio_buffer for VAD-measured duration
    dummy_feat = np.random.randn(1, 1, 100, 40).astype(np.float32)
    dummy_audio = np.random.randn(int(SR * 1.2)).astype(np.float32) * 0.5  # loud audio = VAD active
    passed, metrics = cascade.verify_utterance(dummy_feat, duration_sec=1.2,
                                               audio_buffer=dummy_audio, sr=SR,
                                               ctc_suffix_prob=0.95)
    assert passed is True
    assert metrics["status"] == "WAKE_WORD_TRIGGERED"
    assert "duration_sec_measured" in metrics  # VAD path was used

    # 3. Truncation rejection via CTC Posterior
    passed_ctc_fail, metrics_ctc = cascade.verify_utterance(
        dummy_feat, duration_sec=1.2, ctc_suffix_prob=0.30
    )
    assert passed_ctc_fail is False
    assert "CTC posterior" in metrics_ctc["rejection_reason"]

    # 4. Duration gate: too-short utterance (0.1s) should be rejected
    cascade.mean_duration = 1.2
    cascade.std_duration = 0.1  # tight std to make gate easy to test
    passed_short, metrics_short = cascade.verify_utterance(
        dummy_feat, duration_sec=0.5, ctc_suffix_prob=0.95
    )
    assert passed_short is False
    assert "too short" in metrics_short["rejection_reason"]

    # 5. Duration gate: slow speech (1.8s with upper_std=3.5) should PASS
    passed_slow, metrics_slow = cascade.verify_utterance(
        dummy_feat, duration_sec=1.55, ctc_suffix_prob=0.95
    )
    assert passed_slow is True, f"Slow speech should pass asymmetric gate. Metrics: {metrics_slow}"
