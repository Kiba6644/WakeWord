import numpy as np
import pytest
from audio_utils import (endpoint_utterance, create_truncated_clip, 
                         time_stretch_audio, phonetic_distance, 
                         slice_temporal_segments, generate_phonetic_minimal_pairs)
from config import SR, MIN_CLIP_SEC, MAX_CLIP_SEC

def test_endpoint_basic_utterance():
    silence = np.random.randn(int(SR * 0.5)) * 1e-4
    speech = np.random.randn(int(SR * 1.0)) * 0.5
    buffer = np.concatenate([silence, speech, silence])
    out = endpoint_utterance(buffer, SR)
    assert out is not None
    assert len(out) > SR * 0.8
    assert len(out) < SR * 1.5

def test_endpoint_rejects_pure_noise():
    noise = np.random.randn(int(SR * 2.0)) * 1e-4
    out = endpoint_utterance(noise, SR)
    assert out is None

def test_endpoint_caps_at_max_clip_sec():
    speech = np.random.randn(int(SR * 3.0)) * 0.5
    silence = np.random.randn(int(SR * 0.5)) * 1e-4
    buffer = np.concatenate([speech, silence])
    out = endpoint_utterance(buffer, SR)
    assert out is not None
    assert len(out) == int(MAX_CLIP_SEC * SR)

def test_endpoint_pads_short_utterance():
    silence = np.random.randn(int(SR * 0.5)) * 1e-4
    speech = np.random.randn(int(SR * 0.05)) * 0.5
    buffer = np.concatenate([silence, speech, silence])
    out = endpoint_utterance(buffer, SR, pad_sec=0.0)
    assert out is not None
    assert len(out) == int(MIN_CLIP_SEC * SR)

def test_create_truncated_clip():
    wav = np.random.randn(int(SR * 1.5))
    trunc = create_truncated_clip(wav, SR)
    assert len(trunc) < len(wav)
    assert len(trunc) >= int(MIN_CLIP_SEC * SR)

def test_time_stretch_audio():
    wav = np.random.randn(int(SR * 1.0)).astype(np.float32)
    stretched_fast = time_stretch_audio(wav, rate=1.25, sr=SR)
    assert len(stretched_fast) < len(wav)
    stretched_slow = time_stretch_audio(wav, rate=0.8, sr=SR)
    assert len(stretched_slow) > len(wav)

def test_phonetic_distance():
    assert phonetic_distance("karthik", "karthik") == 0
    dist = phonetic_distance("karthik", "karthika")
    assert dist <= 2
    dist_diff = phonetic_distance("karthik", "banana")
    assert dist_diff > 2

def test_generate_phonetic_minimal_pairs():
    pairs = generate_phonetic_minimal_pairs("Hey Karthika")
    assert len(pairs) > 5
    # Should include suffix truncation "Hey Karthik"
    assert any("karthik" in p.lower() for p in pairs)
    # Should include substitution "Say Karthika"
    assert any("say" in p.lower() for p in pairs)

def test_slice_temporal_segments():
    wav = np.random.randn(int(SR * 1.5)).astype(np.float32)
    segs = slice_temporal_segments(wav, num_segments=3)
    assert len(segs) == 3
    total_samples = sum(len(s) for s in segs)
    assert total_samples == len(wav)
