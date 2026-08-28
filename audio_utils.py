import numpy as np
import logging
import re
from config import MAX_CLIP_SEC, MIN_CLIP_SEC, NUM_TEMPORAL_SEGMENTS, NUM_PHONEMES

logger = logging.getLogger(__name__)

# Standard Phoneme Dictionary Mapping for CTC Target Alignment
PHONEME_MAP = {
    'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9,
    'DH': 10, 'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17,
    'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25,
    'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33,
    'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39, '<SIL>': 40, '<UNK>': 41
}

def endpoint_utterance(stream_buffer: np.ndarray, sr: int,
                       onset_db: float = -40.0, offset_db: float = -45.0,
                       onset_frames: int = 3, offset_frames: int = 8,
                       pad_sec: float = 0.15,
                       max_clip_sec: float = MAX_CLIP_SEC,
                       min_clip_sec: float = MIN_CLIP_SEC) -> np.ndarray | None:
    """
    Endpoints an utterance based on simple energy VAD.
    """
    if stream_buffer.ndim > 1:
        stream_buffer = stream_buffer.squeeze()

    frame_len = int(sr * 0.03)
    hop_len = int(sr * 0.01)
    
    if len(stream_buffer) < frame_len:
        return None
        
    num_frames = 1 + (len(stream_buffer) - frame_len) // hop_len
    if num_frames <= 0:
        return None
        
    shape = (num_frames, frame_len)
    strides = (stream_buffer.strides[0] * hop_len, stream_buffer.strides[0])
    frames = np.lib.stride_tricks.as_strided(stream_buffer, shape=shape, strides=strides)
    
    rms = np.sqrt(np.mean(frames**2, axis=-1) + 1e-10)
    db = 20 * np.log10(rms)
    
    onset_idx = -1
    offset_idx = -1
    consecutive_onset = 0
    consecutive_offset = 0
    
    for i, e in enumerate(db):
        if onset_idx < 0:
            if e > onset_db:
                consecutive_onset += 1
            else:
                consecutive_onset = 0
                
            if consecutive_onset >= onset_frames:
                onset_idx = i - onset_frames + 1
        else:
            if e < offset_db:
                consecutive_offset += 1
            else:
                consecutive_offset = 0
                
            if consecutive_offset >= offset_frames:
                offset_idx = i - offset_frames + 1
                break
                
    if onset_idx < 0 or offset_idx < 0:
        return None
        
    onset_sample = onset_idx * hop_len
    offset_sample = offset_idx * hop_len + frame_len
    
    pad_samples = int(pad_sec * sr)
    start_sample = max(0, onset_sample - pad_samples)
    end_sample = min(len(stream_buffer), offset_sample + pad_samples)
    
    out_wav = stream_buffer[start_sample:end_sample]
    
    max_samples = int(max_clip_sec * sr)
    if len(out_wav) > max_samples:
        excess = len(out_wav) - max_samples
        crop_start = excess // 2
        out_wav = out_wav[crop_start:crop_start + max_samples]
        
    min_samples = int(min_clip_sec * sr)
    if len(out_wav) < min_samples:
        pad_needed = min_samples - len(out_wav)
        out_wav = np.pad(out_wav, (0, pad_needed), mode='constant')
        
    return out_wav

def create_truncated_clip(wav: np.ndarray, sr: int) -> np.ndarray:
    cut_ratio = np.random.uniform(0.60, 0.90)
    cut_point = max(int(MIN_CLIP_SEC * sr), int(len(wav) * cut_ratio))
    truncated = wav[:cut_point]
    
    min_samples = int(MIN_CLIP_SEC * sr)
    if len(truncated) < min_samples:
        pad_needed = min_samples - len(truncated)
        truncated = np.pad(truncated, (0, pad_needed), mode='constant')
        
    return truncated

def time_stretch_audio(wav: np.ndarray, rate: float, sr: int = 16000) -> np.ndarray:
    if abs(rate - 1.0) < 1e-3:
        return wav.copy()
        
    win_size = int(sr * 0.03)
    hop_orig = win_size // 2
    hop_new = int(hop_orig / rate)
    
    if len(wav) < win_size * 2:
        indices = np.linspace(0, len(wav) - 1, int(len(wav) / rate))
        return np.interp(indices, np.arange(len(wav)), wav).astype(np.float32)
        
    window = np.hanning(win_size)
    num_frames = (len(wav) - win_size) // hop_orig
    out_len = num_frames * hop_new + win_size
    output = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    
    for i in range(num_frames):
        in_pos = i * hop_orig
        out_pos = i * hop_new
        frame = wav[in_pos:in_pos + win_size] * window
        output[out_pos:out_pos + win_size] += frame
        norm[out_pos:out_pos + win_size] += window
        
    mask = norm > 1e-3
    output[mask] /= norm[mask]
    return output

def get_phonetic_code(word: str) -> str:
    word = re.sub(r'[^a-zA-Z]', '', word.lower())
    if not word:
        return ""
        
    mapping = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    
    code = [word[0].upper()]
    for char in word[1:]:
        digit = mapping.get(char, '0')
        if digit != '0' and (not code or digit != code[-1]):
            code.append(digit)
            
    return "".join(code)

def phonetic_distance(word1: str, word2: str) -> int:
    code1 = get_phonetic_code(word1)
    code2 = get_phonetic_code(word2)
    
    m, n = len(code1), len(code2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if code1[i - 1] == code2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                
    return int(dp[m][n])

def generate_phonetic_minimal_pairs(phrase: str) -> list[str]:
    """
    SOTA Generative Voice Factory:
    Algorithmically generates exact phonetic minimal pairs, prefix substitutions,
    and suffix truncations for calibrating Stage 2 against false triggers.
    """
    words = phrase.strip().split()
    if not words:
        return []
        
    minimal_pairs = set()
    target_word = words[-1] # The main keyword (e.g. "Karthika")
    prefix = " ".join(words[:-1]) if len(words) > 1 else ""
    
    # 1. Truncation Hard Negatives (Crucial for "Karthik" vs "Karthika")
    if len(target_word) > 3:
        # Suffix cut 1 letter
        minimal_pairs.add(f"{prefix} {target_word[:-1]}".strip())
        # Suffix cut 2 letters
        minimal_pairs.add(f"{prefix} {target_word[:-2]}".strip())
        
    # 2. Suffix Vowel Substitutions
    vowels = ['a', 'o', 'e', 'i', 'u', 'y']
    for v in vowels:
        if not target_word.lower().endswith(v):
            minimal_pairs.add(f"{prefix} {target_word[:-1]}{v}".strip())
            
    # 3. Leading Consonant Substitutions
    consonants = ['b', 'p', 'd', 't', 'g', 'c', 'm', 'n', 's']
    first_char = target_word[0].lower()
    for c in consonants:
        if c != first_char:
            minimal_pairs.add(f"{prefix} {c}{target_word[1:]}".strip())
            
    # 4. Prefix Substitutions
    if prefix:
        common_prefixes = ["say", "play", "okay", "may", "hi"]
        for p in common_prefixes:
            if p.lower() != prefix.lower():
                minimal_pairs.add(f"{p} {target_word}".strip())
                
    return sorted(list(minimal_pairs))

def slice_temporal_segments(wav: np.ndarray, num_segments: int = NUM_TEMPORAL_SEGMENTS) -> list[np.ndarray]:
    total_len = len(wav)
    seg_len = total_len // num_segments
    segments = []
    
    for i in range(num_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < num_segments - 1 else total_len
        seg = wav[start:end]
        segments.append(seg)
        
    return segments
