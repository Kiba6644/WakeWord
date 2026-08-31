import os
import torch

# Audio Frontend
SR = 16000
N_MELS = 40
N_FFT = 400
HOP_LENGTH = 160

# Model Architecture
EMBED_DIM = 128
STAGE1_TEMPORAL_HEAD = "pool"
STAGE2_TEMPORAL_HEAD = "attention"
STAGE1_CHANNELS = (16, 32, 32, 64)
STAGE2_CHANNELS = (32, 64, 64, 128)
NUM_ATTENTION_HEADS = 4

# SOTA Phoneme CTC Head
NUM_PHONEMES = 42                              # 39 phonemes + blank + silence + unk
CTC_LOSS_WEIGHT = 0.3                          # Balanced with distillation

# Endpointing
MAX_CLIP_SEC = 1.8
MIN_CLIP_SEC = 0.4

# Tier C (Truncation-Suppression) Objective
TRUNCATION_AUX_WEIGHT = 0.3
TRUNCATION_MARGIN = 0.8
TRUNCATION_FRACTION_PER_EPISODE = 0.5

# SOTA: Dual-Teacher Distillation (WavLM + Whisper)
WAVLM_TEACHER_MODEL = "microsoft/wavlm-large"  # 1024D Phonetic specialist
WAVLM_EMBED_DIM = 1024
WAVLM_DISTILL_WEIGHT = 0.4

WHISPER_TEACHER_MODEL = "openai/whisper-base"  # 512D Noise/Ambient specialist
WHISPER_EMBED_DIM = 512
WHISPER_DISTILL_WEIGHT = 0.3
USE_DUAL_DISTILLATION = True

# Tempo & Generative Minimal-Pair Factory
TEMPO_AUG_FACTORS = (0.85, 1.0, 1.15, 1.25)
MINIMAL_PAIR_NEGATIVE_COUNT = 12

# Verification Thresholds (Stage 2)
NUM_TEMPORAL_SEGMENTS = 3
STAGE1_THRESHOLD = 0.60
STAGE2_GLOBAL_THRESHOLD = 0.82
SUFFIX_REJECTION_THRESHOLD = 0.62          # Relaxed from 0.65 — reduces false rejects on fast speech
CTC_POSTERIOR_THRESHOLD = 0.70
# Duration gate — asymmetric: strict on short (catches truncations), lenient on long (allows slow speech)
DURATION_GATE_STD_LOWER = 2.0              # Reject if duration < mean - 2.0*std  (tight — catches "Hey Karthik")
DURATION_GATE_STD_UPPER = 3.5             # Allow if duration < mean + 3.5*std  (loose — allows slow speech)
DURATION_GATE_STD = 2.5                   # Legacy fallback — kept for backward compat

# Dataset Configuration & Optimal Sizing for Dual-Teacher
MSWC_LANGUAGES = ["en"]
MSWC_SPLIT = "train[:10000]"                  # Exact 10,000 clips (~15-minute proof-of-concept)
MSWC_MIN_CLIPS_PER_KEYWORD = 30                # Filters out noisy/rare words
PHONETIC_HARD_NEGATIVE_RATIO = 0.6

# Training & Optimization Parameters
# Defaulting to 128 to be safe across environments (T4s have 16GB). Adjust downward if OOM on <= 4GB GPUs.
BATCH_SIZE = 128

NUM_WORKERS = 4
LEARNING_RATE = 5e-4                           # Optimal initial LR for AdamW + Distillation
WEIGHT_DECAY = 1e-4                            # L2 regularization
WARMUP_EPOCHS = 5                              # Linear warmup for distillation alignment
PRETRAIN_EPOCHS = 5                            # Epochs 1-5: Teacher + CTC alignment pre-training
EPOCHS = 50                                    # Total maximum epochs
EARLY_STOPPING_PATIENCE = 5                    # Early stopping if val metric doesn't improve for 5 epochs
MIN_LR = 1e-5                                  # Cosine decay floor
GRAD_CLIP_NORM = 1.0                           # Prevents gradient explosion on CTC loss

N_WAY = 10                                     # 10-way metric learning
N_SUPPORT = 5
N_QUERY = 5

# Environment Paths
DATA_DIR = os.environ.get("KAGGLE_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("KAGGLE_WORKING_DIR", "./output")
