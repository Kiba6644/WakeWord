import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import defaultdict
import glob

from config import (SR, N_MELS, N_FFT, HOP_LENGTH, MIN_CLIP_SEC, MAX_CLIP_SEC, 
                    TEMPO_AUG_FACTORS, PHONETIC_HARD_NEGATIVE_RATIO, 
                    BATCH_SIZE, MSWC_SPLIT)
from audio_utils import (create_truncated_clip, time_stretch_audio, phonetic_distance,
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
    return load_dataset("MLCommons/ml_spoken_words", "en_opus", split=mswc_split, trust_remote_code=True)

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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        raw_audio = item["audio"]["array"]
        in_sr = item["audio"]["sampling_rate"]
        
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

        res = {
            "wav": wav,
            "mel": log_mel,
            "trunc_mel": trunc_log_mel,
            "word": word,
            "phoneme_tokens": torch.tensor(phoneme_tokens, dtype=torch.long)
        }
        
        if self.teacher_targets is not None:
            actual_idx = self.indices[idx] if self.indices is not None else idx
            res["wavlm_target"] = self.teacher_targets["wavlm"][actual_idx]
            res["whisper_target"] = self.teacher_targets["whisper"][actual_idx]
            
        return res

def collate_fn(batch):
    mels = torch.stack([b["mel"] for b in batch], dim=0).transpose(2, 3)
    trunc_mels = torch.stack([b["trunc_mel"] for b in batch], dim=0).transpose(2, 3)
    wavs = [b["wav"] for b in batch]
    words = [b["word"] for b in batch]
    tokens = [b["phoneme_tokens"] for b in batch]
    target_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    targets = torch.cat(tokens, dim=0)
    
    res = {
        "mels": mels,
        "trunc_mels": trunc_mels,
        "wavs": wavs,
        "words": words,
        "targets": targets,
        "target_lengths": target_lengths
    }
    
    if "wavlm_target" in batch[0]:
        res["wavlm_targets"] = torch.stack([b["wavlm_target"] for b in batch], dim=0)
        res["whisper_targets"] = torch.stack([b["whisper_target"] for b in batch], dim=0)
        
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
