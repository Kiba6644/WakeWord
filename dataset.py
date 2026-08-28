import os
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from collections import defaultdict

from config import (SR, N_MELS, MAX_CLIP_SEC, TRUNCATION_FRACTION_PER_EPISODE,
                    TEMPO_AUG_FACTORS, PHONETIC_HARD_NEGATIVE_RATIO, 
                    BATCH_SIZE, MSWC_SPLIT)
from audio_utils import create_truncated_clip, time_stretch_audio, phonetic_distance

class PhoneticClusterManager:
    """
    Indexes vocabulary by phonetic similarity to supply hard-negative
    minimal pairs during episodic training.
    """
    def __init__(self, words: list[str]):
        self.words = list(set(words))
        self.phonetic_neighbors = defaultdict(list)
        self._build_index()

    def _build_index(self):
        # Index neighboring pairs with Levenshtein phonetic distance <= 2
        for i, w1 in enumerate(self.words):
            for w2 in self.words[i+1:]:
                dist = phonetic_distance(w1, w2)
                if dist <= 2:
                    self.phonetic_neighbors[w1].append(w2)
                    self.phonetic_neighbors[w2].append(w1)

    def get_hard_negatives(self, target_word: str, k: int = 3) -> list[str]:
        neighbors = self.phonetic_neighbors.get(target_word, [])
        if len(neighbors) >= k:
            return random.sample(neighbors, k)
        elif neighbors:
            # Pad with random words
            random_fill = [w for w in self.words if w != target_word and w not in neighbors]
            needed = k - len(neighbors)
            return neighbors + random.sample(random_fill, min(needed, len(random_fill)))
        else:
            random_fill = [w for w in self.words if w != target_word]
            return random.sample(random_fill, min(k, len(random_fill)))

class WakeWordDataset(Dataset):
    def __init__(self, hf_dataset, noise_clips=None, apply_tempo_aug=True):
        self.dataset = hf_dataset
        self.noise_clips = noise_clips
        self.apply_tempo_aug = apply_tempo_aug
        self.mel_spec = T.MelSpectrogram(sample_rate=SR, n_mels=N_MELS, n_fft=400, hop_length=160)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        wav = item["audio"]["array"]
        label = item["label"]
        word = item["word"]

        if item["audio"]["sampling_rate"] != SR:
            wav = torchaudio.functional.resample(
                torch.tensor(wav, dtype=torch.float32), 
                item["audio"]["sampling_rate"], SR
            ).numpy()

        wav = np.array(wav, dtype=np.float32)

        # 1. Dynamic Tempo / Speed Augmentation (Fast vs Slow speech)
        if self.apply_tempo_aug and random.random() < 0.5:
            rate = random.choice(TEMPO_AUG_FACTORS)
            wav = time_stretch_audio(wav, rate, SR)

        # 2. Background Noise Augmentation
        if self.noise_clips and random.random() < 0.5:
            noise = random.choice(self.noise_clips)
            if len(noise) > len(wav):
                start = random.randint(0, len(noise) - len(wav))
                noise = noise[start:start + len(wav)]
            else:
                noise = np.pad(noise, (0, len(wav) - len(noise)), mode='wrap')
                
            snr = random.uniform(5, 20)
            wav_power = np.sum(wav**2)
            noise_power = np.sum(noise**2)
            scale = np.sqrt(wav_power / (noise_power * (10**(snr/10)) + 1e-8))
            wav = wav + noise * scale

        return {"wav": wav, "label": label, "word": word}

def episodic_collate(batch):
    wavs = [item["wav"] for item in batch]
    labels = [item["label"] for item in batch]
    words = [item["word"] for item in batch]
    return wavs, labels, words

def get_dataloaders(batch_size=BATCH_SIZE, num_workers=4):
    try:
        from config import MSWC_SPLIT
        print(f"Loading MSWC dataset split ({MSWC_SPLIT})...")
        ds = load_dataset("MLCommons/ml_spoken_words", "en_opus", split=MSWC_SPLIT, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load online dataset: {e}. Falling back to local/dummy loader.")
        return None, None
        
    train_dataset = WakeWordDataset(ds)
    train_sampler = DistributedSampler(train_dataset) if torch.distributed.is_initialized() else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        shuffle=(train_sampler is None),
        num_workers=num_workers, 
        collate_fn=episodic_collate
    )
                              
    return train_loader, None
