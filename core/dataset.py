import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import defaultdict
import glob

from .config import (SR, N_MELS, N_FFT, HOP_LENGTH, MIN_CLIP_SEC, MAX_CLIP_SEC, 
                    TEMPO_AUG_FACTORS, PHONETIC_HARD_NEGATIVE_RATIO, 
                    BATCH_SIZE, MSWC_SPLIT)
from .audio_utils import (create_truncated_clip, time_stretch_audio, phonetic_distance,
                         pad_or_trim, word_to_phoneme_tokens)

def load_background_noise_bank(noise_dir: str) -> list[np.ndarray]:
    if not noise_dir or not os.path.exists(noise_dir):
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
    return noise_clips

def load_speech_dataset(mswc_split: str, dataset_path: str = None):
    if dataset_path and os.path.exists(dataset_path):
        from datasets import load_from_disk
        return load_from_disk(dataset_path)
    from datasets import load_dataset
    try:
        return load_dataset("MLCommons/ml_spoken_words", "en_opus", split=mswc_split, trust_remote_code=True)
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print("\n❌ HuggingFace 'datasets' >= 3.0.0 does not support dataset loading scripts.")
            print("💡 Fix by installing datasets < 3.0.0:\n   pip install \"datasets<3.0.0\"\n")
        raise e

class MSWCTrainingDataset(Dataset):
    def __init__(self, hf_dataset, noise_clips=None, target_sec=1.2, teacher_targets=None, indices=None):
        self.dataset = hf_dataset
        self.noise_clips = noise_clips or []
        self.target_samples = int(target_sec * SR)
        self.teacher_targets = teacher_targets
        self.indices = indices
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
        )
        self.freq_mask = T.FrequencyMasking(freq_mask_param=12)
        self.time_mask = T.TimeMasking(time_mask_param=10)
        
        self.hey_clips = []
        hey_dir = "./hey_clips"
        if not os.path.exists(hey_dir) or not glob.glob(os.path.join(hey_dir, "*.wav")):
            print("📣 'hey_clips' directory missing or empty. Auto-generating synthetic phrase clips...")
            try:
                import subprocess
                subprocess.run([sys.executable, "scripts/generate_hey_clips.py", "--output_dir", hey_dir, "--count", "120"], check=False)
            except Exception as e:
                print(f"⚠️ Could not auto-generate hey clips: {e}")

        if os.path.exists(hey_dir):
            wav_files = glob.glob(os.path.join(hey_dir, "*.wav"))
            for p in wav_files:
                try:
                    w, in_sr = torchaudio.load(p)
                    if w.shape[0] > 1:
                        w = w.mean(dim=0, keepdim=True)
                    if in_sr != SR:
                        w = torchaudio.functional.resample(w, in_sr, SR)
                    self.hey_clips.append(w.squeeze(0).numpy())
                except Exception:
                    pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        raw_audio = item["audio"]["array"]
        in_sr = item["audio"]["sampling_rate"]
        
        word = item.get("keyword") or item.get("word")
        if not word and "label" in item:
            val = item["label"]
            base_ds = self.dataset
            while hasattr(base_ds, "dataset"):
                base_ds = base_ds.dataset
            if isinstance(val, str):
                word = val
            elif hasattr(base_ds, "features") and "label" in base_ds.features:
                try:
                    word = base_ds.features["label"].int2str(val)
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

        if self.hey_clips and random.random() < 0.35:
            hey = random.choice(self.hey_clips)
            wav = np.concatenate([hey, wav])

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

        if random.random() < 0.5:
            gain_db = random.uniform(-6.0, 6.0)
            wav = wav * (10 ** (gain_db / 20.0))

        wav = pad_or_trim(wav, self.target_samples)
        trunc_wav = create_truncated_clip(wav, SR)
        trunc_wav = pad_or_trim(trunc_wav, self.target_samples)

        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        trunc_tensor = torch.tensor(trunc_wav, dtype=torch.float32).unsqueeze(0)

        mel = self.mel_transform(wav_tensor)
        trunc_mel = self.mel_transform(trunc_tensor)

        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        trunc_log_mel = torch.log(torch.clamp(trunc_mel, min=1e-5))

        # Per-utterance CMVN
        log_mel = (log_mel - log_mel.mean(dim=-1, keepdim=True)) / (log_mel.std(dim=-1, keepdim=True) + 1e-5)
        trunc_log_mel = (trunc_log_mel - trunc_log_mel.mean(dim=-1, keepdim=True)) / (trunc_log_mel.std(dim=-1, keepdim=True) + 1e-5)

        # SpecAugment
        if random.random() < 0.5:
            log_mel = self.freq_mask(log_mel)
            trunc_log_mel = self.freq_mask(trunc_log_mel)
        if random.random() < 0.5:
            log_mel = self.freq_mask(log_mel)
            trunc_log_mel = self.freq_mask(trunc_log_mel)
        if random.random() < 0.5:
            log_mel = self.time_mask(log_mel)
            trunc_log_mel = self.time_mask(trunc_log_mel)

        phoneme_tokens = word_to_phoneme_tokens(word)

        actual_idx = self.indices[idx] if self.indices is not None else idx
        res = {
            "wav": wav,
            "mel": log_mel,
            "trunc_mel": trunc_log_mel,
            "word": word,
            "phoneme_tokens": torch.tensor(phoneme_tokens, dtype=torch.long),
            "sample_idx": actual_idx
        }
        
        return res

def collate_fn(batch):
    mels = torch.stack([b["mel"] for b in batch], dim=0).transpose(2, 3)
    trunc_mels = torch.stack([b["trunc_mel"] for b in batch], dim=0).transpose(2, 3)
    wavs = [b["wav"] for b in batch]
    words = [b["word"] for b in batch]
    tokens = [b["phoneme_tokens"] for b in batch]
    target_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    targets = torch.cat(tokens, dim=0)
    sample_idxs = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
    
    res = {
        "mels": mels,
        "trunc_mels": trunc_mels,
        "wavs": wavs,
        "words": words,
        "targets": targets,
        "target_lengths": target_lengths,
        "sample_idxs": sample_idxs
    }
    
    return res

def get_dataloaders(batch_size=BATCH_SIZE, num_workers=4):
    print(f"Loading MSWC dataset split ({MSWC_SPLIT})...")
    dataset_path = os.environ.get("KAGGLE_DATASET_PATH", None)
    ds = load_speech_dataset(MSWC_SPLIT, dataset_path=dataset_path)
    train_dataset = MSWCTrainingDataset(ds)
    train_sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    return train_loader, None
