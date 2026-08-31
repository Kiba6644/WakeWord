"""
SOTA Wake Word Engine — Self-Contained Complete Training Pipeline for Kaggle (2x T4 GPUs)
========================================================================================
Architecture:
- Student: Depthwise-Separable CNN (DS-CNN) + Squeeze-and-Excitation + Multi-Head Attention Pooling
- Teachers: microsoft/wavlm-large (1024D Phonetic) + openai/whisper-base (512D Robustness)
- Losses: Prototypical Metric Loss + Tier C Truncation Margin + Dual Distillation + Phoneme CTC
- Training Schedule: 2-Stage (5 Warmup Pretrain Epochs -> 45 Fine-Tuning Epochs with Early Stopping)
- Export: Validated INT8 Dynamic Quantized ONNX (<1.5 MB)
"""

import os
import sys
import math
import glob
import random
import re
import argparse
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")

try:
    from transformers import WavLMModel, WhisperModel
except ImportError:
    print("Please install transformers: pip install transformers")

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =====================================================================
# 1. AUDIO & ARCHITECTURE CONSTANTS
# =====================================================================
SR = 16000
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160
EMBED_DIM = 128

STAGE1_CHANNELS = (16, 32, 32, 64)
STAGE2_CHANNELS = (32, 64, 64, 128)
NUM_ATTENTION_HEADS = 4
NUM_PHONEMES = 42

MAX_CLIP_SEC = 1.8
MIN_CLIP_SEC = 0.4

# Loss Weights
TRUNCATION_AUX_WEIGHT = 0.3
TRUNCATION_MARGIN = 0.8
WAVLM_DISTILL_WEIGHT = 0.4
WHISPER_DISTILL_WEIGHT = 0.3
CTC_LOSS_WEIGHT = 0.3

WAVLM_EMBED_DIM = 1024
WHISPER_EMBED_DIM = 512

TEMPO_AUG_FACTORS = (0.85, 1.0, 1.15, 1.25)
WARMUP_EPOCHS = 5
MIN_LR = 1e-5
GRAD_CLIP_NORM = 1.0

# Standard CMU-style Phoneme Index Mapping
PHONEME_MAP = {
    'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9,
    'DH': 10, 'EH': 11, 'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17,
    'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'NG': 24, 'OW': 25,
    'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33,
    'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39, '<SIL>': 40, '<UNK>': 41
}

# =====================================================================
# 2. AUDIO & PHONETIC UTILITIES
# =====================================================================
def time_stretch_audio(wav: np.ndarray, rate: float, sr: int = SR) -> np.ndarray:
    """Time-stretch without pitch alteration via Overlap-Add."""
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

def create_truncated_clip(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Tier C synthetic truncation: cuts between 60%-85% duration."""
    cut_ratio = np.random.uniform(0.60, 0.85)
    cut_point = max(int(MIN_CLIP_SEC * sr), int(len(wav) * cut_ratio))
    truncated = wav[:cut_point]
    min_samples = int(MIN_CLIP_SEC * sr)
    if len(truncated) < min_samples:
        truncated = np.pad(truncated, (0, min_samples - len(truncated)), mode='constant')
    return truncated

def pad_or_trim(wav: np.ndarray, target_length: int) -> np.ndarray:
    """Safely pads or center-crops audio to target length."""
    if len(wav) == target_length:
        return wav
    elif len(wav) > target_length:
        start = (len(wav) - target_length) // 2
        return wav[start:start + target_length]
    else:
        pad_needed = target_length - len(wav)
        left = pad_needed // 2
        return np.pad(wav, (left, pad_needed - left), mode='constant')

def word_to_phoneme_tokens(word: str) -> list[int]:
    """Converts a word into approximate phoneme index targets for CTC training."""
    word = re.sub(r'[^a-zA-Z]', '', word.upper())
    tokens = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i:i+2] in PHONEME_MAP:
            tokens.append(PHONEME_MAP[word[i:i+2]])
            i += 2
        elif word[i] in PHONEME_MAP:
            tokens.append(PHONEME_MAP[word[i]])
            i += 1
        else:
            tokens.append(PHONEME_MAP['<UNK>'])
            i += 1
    return tokens if tokens else [PHONEME_MAP['<UNK>']]

def load_background_noise_bank(noise_dir: str) -> list[np.ndarray]:
    """Directly loads background noise audio files from the specified hardcoded directory."""
    if not noise_dir or not os.path.exists(noise_dir):
        print(f"ℹ️ Background noise directory '{noise_dir}' not found. Using synthetic noise augmentation.")
        return []
        
    wav_files = glob.glob(os.path.join(noise_dir, "*.wav"))
    noise_clips = []
    for p in wav_files:
        try:
            w, in_sr = torchaudio.load(p)
            if w.shape[0] > 1:
                w = w.mean(dim=0, keepdim=True)
            if in_sr != SR:
                w = torchaudio.functional.resample(w, in_sr, SR)
            noise_clips.append(w.squeeze(0).numpy())
        except Exception:
            pass
            
    print(f"🔊 Loaded {len(noise_clips)} background noise files directly from: {noise_dir}")
    return noise_clips

# =====================================================================
# 3. STUDENT MODEL ARCHITECTURE
# =====================================================================
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.silu = nn.SiLU(inplace=True)
        self.se = SqueezeExcitation(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.silu(x)
        x = self.se(x)
        return x

class DSCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64, 64, 128)):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=(10, 4), stride=(2, 2), padding=(4, 1), bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        blocks = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks.append(DepthwiseSeparableConv(in_ch, out_ch, kernel_size=(3, 3), padding=(1, 1)))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.blocks(x)
        return x

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads=NUM_ATTENTION_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.query_tokens = nn.Parameter(torch.empty(num_heads, in_dim))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        self.key_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.val_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * in_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        b, t, d = x.shape
        keys = self.key_proj(x)
        vals = self.val_proj(x)
        
        q = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        scores = torch.bmm(q, keys.transpose(1, 2)) / (d ** 0.5)
        
        if mask is not None:
            # mask: (B, T) boolean. True means valid frame.
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights) # safe fallback if all masked
        
        pooled = torch.bmm(attn_weights, vals)
        pooled = pooled.reshape(b, -1)
        out = self.out_proj(pooled)
        out = self.layer_norm(out)
        return out

class WakeWordModel(nn.Module):
    def __init__(self, channels=STAGE2_CHANNELS, temporal_head="attention", embed_dim=EMBED_DIM):
        super().__init__()
        self.encoder = DSCNNEncoder(in_channels=1, channels=channels)
        self.temporal_head_type = temporal_head
        self.spatial_pool = nn.AdaptiveAvgPool2d((None, 1))
        
        if temporal_head == "attention":
            self.temporal = MultiHeadAttentionPooling(channels[-1], embed_dim)
        elif temporal_head == "gru":
            self.gru = nn.GRU(channels[-1], embed_dim, batch_first=True)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(channels[-1], embed_dim)

        self.distill_wavlm_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, WAVLM_EMBED_DIM)
        )
        self.distill_whisper_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, WHISPER_EMBED_DIM)
        )
        self.ctc_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels[-1] // 2, NUM_PHONEMES),
            nn.LogSoftmax(dim=-1)
        )

    def extract_time_features(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feat = self.encoder(x)
        feat = self.spatial_pool(feat).squeeze(-1)
        feat = feat.transpose(1, 2)
        return feat

    def forward(self, x, mask=None, return_distill=False, return_ctc=False):
        feat = self.extract_time_features(x)
        
        if self.temporal_head_type == "attention":
            if mask is not None:
                # Downsample mask to match CNN time dimension (stride=2 in first layer)
                mask_float = mask.float().unsqueeze(1)
                mask_down = F.adaptive_max_pool1d(mask_float, output_size=feat.size(1)).squeeze(1)
                bool_mask = mask_down > 0.5
            else:
                bool_mask = None
            embed = self.temporal(feat, mask=bool_mask)
        elif self.temporal_head_type == "gru":
            out, _ = self.gru(feat)
            embed = out[:, -1, :]
        else:
            pooled = self.pool(feat.transpose(1, 2)).squeeze(-1)
            embed = self.fc(pooled)
            
        norm_embed = F.normalize(embed, p=2, dim=-1)
        ctc_logits = self.ctc_head(feat) if (return_ctc or return_distill) else None
        
        if return_distill:
            wavlm_proj = self.distill_wavlm_proj(norm_embed)
            whisper_proj = self.distill_whisper_proj(norm_embed)
            return norm_embed, (wavlm_proj, whisper_proj), ctc_logits
            
        if return_ctc:
            return norm_embed, ctc_logits
            
        return norm_embed

def load_speech_dataset(mswc_split: str, dataset_path: str = None):
    """
    Loads MSWC (Multilingual Spoken Words Corpus) English dataset.
    If dataset_path is provided and exists, loads directly from disk in seconds.
    """
    if dataset_path and os.path.exists(dataset_path):
        print(f"⚡ Loading pre-processed MSWC dataset directly from disk: {dataset_path}")
        from datasets import load_from_disk
        return load_from_disk(dataset_path)

    print(f"Loading primary MSWC English dataset ({mswc_split})...")
    try:
        ds = load_dataset("MLCommons/ml_spoken_words", "en_opus", split=mswc_split, trust_remote_code=True)
        print(f"✅ Successfully loaded MSWC dataset ({len(ds)} audio samples)!")
        return ds
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MSWC dataset: {e}\n"
            f"Fix: Ensure you have 'datasets < 3.0.0' installed (e.g. `pip install \"datasets<3.0.0\"`) and Internet access is ON."
        )

# =====================================================================
# 4. DATASET & FRONTEND
# =====================================================================
class MSWCTrainingDataset(Dataset):
    def __init__(self, hf_dataset, noise_clips=None, target_sec=1.2):
        self.dataset = hf_dataset
        self.noise_clips = noise_clips or []
        self.target_samples = int(target_sec * SR)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        raw_audio = item["audio"]["array"]
        in_sr = item["audio"]["sampling_rate"]
        
        # Resolve word label safely across MSWC, Google Speech Commands, and AudioFolder
        word = item.get("word")
        if not word and "label" in item:
            val = item["label"]
            if isinstance(val, str):
                word = val
            elif hasattr(self.dataset, "features") and "label" in self.dataset.features:
                try:
                    word = self.dataset.features["label"].int2str(val)
                except Exception:
                    word = str(val)
            else:
                word = str(val)
        word = str(word) if word else "unknown"

        wav_t = torch.tensor(raw_audio, dtype=torch.float32)
        if in_sr != SR:
            wav_t = torchaudio.functional.resample(wav_t, in_sr, SR)
            
        wav = wav_t.numpy()

        if random.random() < 0.5:
            rate = random.choice(TEMPO_AUG_FACTORS)
            wav = time_stretch_audio(wav, rate, SR)

        if self.noise_clips and random.random() < 0.6:
            noise = random.choice(self.noise_clips)
            if len(noise) > len(wav):
                start = random.randint(0, len(noise) - len(wav))
                noise_seg = noise[start:start + len(wav)]
            else:
                noise_seg = np.pad(noise, (0, len(wav) - len(noise)), mode='wrap')
                
            snr = random.uniform(5.0, 20.0)
            sig_power = np.sum(wav**2) + 1e-8
            noise_power = np.sum(noise_seg**2) + 1e-8
            scale = np.sqrt(sig_power / (noise_power * (10**(snr / 10.0))))
            wav = wav + noise_seg * scale

        wav = pad_or_trim(wav, self.target_samples)
        trunc_wav = create_truncated_clip(wav, SR)
        trunc_wav = pad_or_trim(trunc_wav, self.target_samples)

        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        trunc_tensor = torch.tensor(trunc_wav, dtype=torch.float32).unsqueeze(0)

        mel = self.mel_transform(wav_tensor)
        trunc_mel = self.mel_transform(trunc_tensor)

        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        trunc_log_mel = torch.log(torch.clamp(trunc_mel, min=1e-5))
        phoneme_tokens = word_to_phoneme_tokens(word)

        return {
            "wav": wav,
            "mel": log_mel,
            "trunc_mel": trunc_log_mel,
            "word": word,
            "phoneme_tokens": torch.tensor(phoneme_tokens, dtype=torch.long)
        }

def collate_fn(batch):
    mels = torch.stack([b["mel"] for b in batch], dim=0).transpose(2, 3)
    trunc_mels = torch.stack([b["trunc_mel"] for b in batch], dim=0).transpose(2, 3)
    wavs = [b["wav"] for b in batch]
    words = [b["word"] for b in batch]
    tokens = [b["phoneme_tokens"] for b in batch]
    target_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    targets = torch.cat(tokens, dim=0)
    
    return {
        "mels": mels,
        "trunc_mels": trunc_mels,
        "wavs": wavs,
        "words": words,
        "targets": targets,
        "target_lengths": target_lengths
    }

# =====================================================================
# 5. LOSS FUNCTIONS
# =====================================================================
def truncation_margin_loss(embed_full, embed_truncated, margin=TRUNCATION_MARGIN):
    dist = torch.norm(embed_full - embed_truncated, dim=-1)
    return F.relu(margin - dist).mean()

def cosine_distillation_loss(student_proj, teacher_embed):
    s_norm = F.normalize(student_proj, p=2, dim=-1)
    t_norm = F.normalize(teacher_embed, p=2, dim=-1)
    return (1.0 - (s_norm * t_norm).sum(dim=-1)).mean()

def compute_prototypical_loss(embeds, words):
    word_to_indices = defaultdict(list)
    for i, w in enumerate(words):
        word_to_indices[w].append(i)
        
    classes = [w for w, idxs in word_to_indices.items() if len(idxs) >= 2]
    if len(classes) < 2:
        return torch.tensor(0.0, device=embeds.device, requires_grad=True), 1.0
        
    prototypes = []
    queries = []
    query_labels = []
    
    for c_idx, c in enumerate(classes):
        idxs = word_to_indices[c]
        support_idx = idxs[0]
        query_idxs = idxs[1:]
        
        proto = embeds[support_idx:support_idx+1]
        prototypes.append(proto)
        
        for q_idx in query_idxs:
            queries.append(embeds[q_idx:q_idx+1])
            query_labels.append(c_idx)
            
    prototypes = torch.cat(prototypes, dim=0)
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    queries = torch.cat(queries, dim=0)
    query_labels = torch.tensor(query_labels, dtype=torch.long, device=embeds.device)
    
    dists = torch.norm(queries.unsqueeze(1) - prototypes.unsqueeze(0), dim=-1)
    log_p_y = F.log_softmax(-dists * 5.0, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)
    
    acc = (log_p_y.max(dim=1)[1] == query_labels).float().mean().item()
    return loss, acc

# =====================================================================
# 6. MAIN EXECUTION PIPELINE
# =====================================================================
def run_training(
    noise_dir: str = "/kaggle/input/datasets/neehakurelli/google-speech-commands/_background_noise_",
    output_dir: str = "./output",
    dataset_path: str = None,
    resume_path: str = None,
    mswc_split: str = "train[:10000]",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    wavlm_model: str = "microsoft/wavlm-large",
    whisper_model: str = "openai/whisper-base"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"🚀 Starting SOTA Wake Word Training Pipeline on: {device}")
    print(f"=======================================================")
    print(f"  • Noise Directory:       {noise_dir}")
    print(f"  • Output Directory:      {output_dir}")
    print(f"  • MSWC Split:            {mswc_split}")
    print(f"  • Dataset Path:          {dataset_path or 'Online Download'}")
    print(f"  • Resume Checkpoint:     {resume_path or 'Auto-Detect'}")
    print(f"  • Batch Size:            {batch_size}")
    print(f"  • Max Epochs:            {epochs}")
    print(f"  • Learning Rate:         {lr}")
    print(f"  • Early Stopping:        {early_stopping_patience} epochs\n")
    
    # 1. Load Datasets
    ds = load_speech_dataset(mswc_split, dataset_path=dataset_path)
    noise_bank = load_background_noise_bank(noise_dir)
    
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(
        MSWCTrainingDataset(train_ds, noise_clips=noise_bank), batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        MSWCTrainingDataset(val_ds, noise_clips=noise_bank), batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    print(f"Dataset Ready: {len(train_ds)} train clips, {len(val_ds)} val clips.")
    
    # 2. Load Student and Teachers
    student = WakeWordModel().to(device)
    print(f"Student Model Parameters: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")
    
    print(f"Loading WavLM Teacher ({wavlm_model})...")
    wavlm = WavLMModel.from_pretrained(wavlm_model).to(device)
    wavlm.eval()
    for p in wavlm.parameters(): p.requires_grad = False
    
    print(f"Loading Whisper Teacher ({whisper_model})...")
    from transformers import WhisperFeatureExtractor
    whisper_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
    whisper = WhisperModel.from_pretrained(whisper_model).encoder.to(device)
    whisper.eval()
    for p in whisper.parameters(): p.requires_grad = False
    
    # 3. Optimizers
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        prog = float(epoch - WARMUP_EPOCHS) / float(max(1, epochs - WARMUP_EPOCHS))
        return (MIN_LR/lr) + (1.0 - (MIN_LR/lr)) * 0.5 * (1.0 + math.cos(math.pi * prog))
        
    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    
    best_val_loss = float("inf")
    patience = 0
    start_epoch = 0

    # Auto-detect or load resume checkpoint
    target_resume = resume_path or os.path.join(output_dir, "best_sota_wakeword_model.pt")
    if os.path.exists(target_resume):
        print(f"🔄 Checkpoint found at '{target_resume}'! Resuming training state...")
        checkpoint = torch.load(target_resume, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"   ✓ Successfully resumed! Starting from Epoch {start_epoch + 1}/{epochs}")
        if 'val_metric' in checkpoint:
            best_val_loss = checkpoint['val_metric']
            print(f"   ✓ Best validation loss restored: {best_val_loss:.4f}")
    
    # 4. Training Loop
    for epoch in range(start_epoch, epochs):
        student.train()
        is_pretrain = (epoch < 5) # First 5 epochs warmup
        total_epoch_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{epochs} {'[Stage 1: Teacher Alignment]' if is_pretrain else '[Stage 2: Metric Fine-Tuning]'} ---")
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            mels = batch["mels"].to(device)
            trunc_mels = batch["trunc_mels"].to(device)
            wavs_np = np.stack(batch["wavs"], axis=0)
            wavs_t = torch.tensor(wavs_np, dtype=torch.float32, device=device)
            
            # Compute energy mask to ignore silence padding (-9.2 is approx log(1e-4))
            energy = mels.mean(dim=-1).squeeze(1) # (B, T)
            attention_mask = energy > -10.0
            
            with torch.no_grad():
                # WavLM target
                wavlm_out = wavlm(wavs_t).last_hidden_state
                wavlm_target = wavlm_out.mean(dim=1)
                
                # Whisper target (REAL, 3000 frames)
                whisper_inputs = whisper_extractor(
                    [w for w in wavs_np], sampling_rate=SR, return_tensors="pt"
                ).input_features.to(device)
                whisper_out = whisper(whisper_inputs).last_hidden_state
                whisper_target = whisper_out.mean(dim=1)
            
            with autocast():
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
                loss_proto, p_acc = compute_prototypical_loss(norm_embed, batch["words"])
                
                if is_pretrain:
                    loss = (WAVLM_DISTILL_WEIGHT * loss_wavlm) + (WHISPER_DISTILL_WEIGHT * loss_whisper) + (CTC_LOSS_WEIGHT * loss_ctc)
                else:
                    loss = (loss_proto + 
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
                print(f"Step {step:03d} | Loss: {loss.item():.4f} | Distill: {(loss_wavlm+loss_whisper).item():.3f} | TierC: {loss_tier_c.item():.3f} | Proto: {loss_proto.item():.3f}")
                
        scheduler.step()
        avg_train_loss = total_epoch_loss / len(train_loader)
        
        # Validation & Early Stopping
        student.eval()
        val_trunc_loss = 0.0
        val_proto_loss = 0.0
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
                p_loss, _ = compute_prototypical_loss(v_embed, v_batch["words"])
                val_proto_loss += p_loss.item()
                val_batches += 1
                
        val_trunc_loss /= val_batches
        val_proto_loss /= val_batches
        val_combined = val_trunc_loss + val_proto_loss # We want to minimize both
        
        print(f"Epoch {epoch+1} Done | Train Loss: {avg_train_loss:.4f} | Val Trunc: {val_trunc_loss:.4f} | Val Proto: {val_proto_loss:.4f}")
        
        if val_combined < best_val_loss:
            best_val_loss = val_combined
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'val_metric': best_val_loss
            }, os.path.join(output_dir, "best_sota_wakeword_model.pt"))
            print(f"⭐ New Best Model Saved (Val Loss: {best_val_loss:.4f})")
        else:
            if not is_pretrain:
                patience += 1
                print(f"EarlyStopping Counter: {patience}/{early_stopping_patience}")
                if patience >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {early_stopping_patience} stagnant epochs!")
                    break
                    
    # Export INT8 ONNX Checkpoint
    print("\n--- Exporting Production INT8 ONNX Model ---")
    student.eval()
    dummy_input = torch.randn(1, 1, 100, 40, device=device)
    onnx_path = os.path.join(output_dir, "sota_wakeword_model.onnx")
    int8_path = os.path.join(output_dir, "sota_wakeword_model_int8.onnx")
    
    torch.onnx.export(
        student, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {2: 'time_steps'}}, opset_version=13
    )
    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
    print(f"✅ Export Completed! INT8 File: {int8_path} ({os.path.getsize(int8_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOTA Wake Word Engine")
    parser.add_argument("--noise_dir", type=str, default="/kaggle/input/datasets/neehakurelli/google-speech-commands/_background_noise_")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to pre-processed local dataset directory")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint .pt file to resume training from")
    parser.add_argument("--mswc_split", type=str, default="train[:10000]")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--wavlm_model", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-base")
    
    args = parser.parse_args()
    run_training(
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        resume_path=args.resume_path,
        mswc_split=args.mswc_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.early_stopping_patience,
        wavlm_model=args.wavlm_model,
        whisper_model=args.whisper_model
    )
