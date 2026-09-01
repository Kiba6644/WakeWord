import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from config import (STAGE2_CHANNELS, STAGE2_TEMPORAL_HEAD, EMBED_DIM,
                    WAVLM_TEACHER_MODEL, WHISPER_TEACHER_MODEL, USE_DUAL_DISTILLATION,
                    LEARNING_RATE, WEIGHT_DECAY, WARMUP_EPOCHS, PRETRAIN_EPOCHS, EPOCHS, 
                    EARLY_STOPPING_PATIENCE, MIN_LR, GRAD_CLIP_NORM, BATCH_SIZE, NUM_WORKERS, 
                    OUTPUT_DIR, WAVLM_DISTILL_WEIGHT, WHISPER_DISTILL_WEIGHT, CTC_LOSS_WEIGHT, 
                    TRUNCATION_AUX_WEIGHT)
from model import WakeWordModel
from dataset import get_dataloaders
from losses import truncation_margin_loss, cosine_distillation_loss, supervised_contrastive_loss

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr_ratio=MIN_LR/LEARNING_RATE):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def load_dual_teachers(device):
    teachers = {}
    try:
        from transformers import WavLMModel, WhisperModel, WhisperFeatureExtractor
        print(f"Loading WavLM Teacher ({WAVLM_TEACHER_MODEL})...")
        wavlm = WavLMModel.from_pretrained(WAVLM_TEACHER_MODEL).to(device)
        wavlm.eval()
        for p in wavlm.parameters():
            p.requires_grad = False
        teachers["wavlm"] = wavlm

        print(f"Loading Whisper Teacher ({WHISPER_TEACHER_MODEL})...")
        whisper_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_TEACHER_MODEL)
        whisper = WhisperModel.from_pretrained(WHISPER_TEACHER_MODEL).encoder.to(device)
        whisper.eval()
        for p in whisper.parameters():
            p.requires_grad = False
        teachers["whisper"] = whisper
        teachers["whisper_extractor"] = whisper_extractor
        
    except Exception as e:
        print(f"Teacher models load skipped ({e}). Running student standalone mode.")
    return teachers

def train(rank, world_size):
    setup(rank, world_size)
    
    model = WakeWordModel(
        channels=STAGE2_CHANNELS, 
        temporal_head=STAGE2_TEMPORAL_HEAD, 
        embed_dim=EMBED_DIM
    ).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    teachers = load_dual_teachers(rank) if USE_DUAL_DISTILLATION else {}
    
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_EPOCHS, EPOCHS)
    scaler = GradScaler()
    
    train_loader, _ = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    if train_loader is None:
        print("Dataset loader not initialized. Exiting.")
        cleanup()
        return

    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "logs")) if (rank == 0 and SummaryWriter) else None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            
        is_pretrain_phase = (epoch < PRETRAIN_EPOCHS)
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            with autocast():
                mels = batch["mels"].to(rank, non_blocking=True)
                trunc_mels = batch["trunc_mels"].to(rank, non_blocking=True)
                energy = mels.mean(dim=-1).squeeze(1)
                attention_mask = energy > -10.0
                
                norm_embed, (s_wavlm_proj, s_whisper_proj), ctc_logits = model(mels, mask=attention_mask, return_distill=True)
                trunc_embed = model(trunc_mels, mask=attention_mask)
                
                # Teachers
                wavlm_target = None
                whisper_target = None
                if teachers:
                    import numpy as np
                    wavs_np = np.stack(batch["wavs"], axis=0)
                    wavs_t = torch.tensor(wavs_np, dtype=torch.float32, device=rank)
                    with torch.no_grad():
                        wavlm_target = teachers["wavlm"](wavs_t).last_hidden_state.mean(dim=1)
                        wh_inputs = teachers["whisper_extractor"](
                            [w for w in wavs_np], sampling_rate=16000, return_tensors="pt"
                        ).input_features.to(rank, non_blocking=True)
                        whisper_target = teachers["whisper"](wh_inputs).last_hidden_state.mean(dim=1)

                loss_wavlm = cosine_distillation_loss(s_wavlm_proj, wavlm_target) if wavlm_target is not None else torch.tensor(0.0, device=rank)
                loss_whisper = cosine_distillation_loss(s_whisper_proj, whisper_target) if whisper_target is not None else torch.tensor(0.0, device=rank)
                
                input_lengths = torch.full((mels.size(0),), ctc_logits.size(1), dtype=torch.long, device=rank)
                loss_ctc = ctc_loss_fn(
                    ctc_logits.transpose(0, 1), batch["targets"].to(rank), 
                    input_lengths, batch["target_lengths"].to(rank)
                )
                
                loss_tier_c = truncation_margin_loss(norm_embed, trunc_embed)
                loss_supcon, p_acc = supervised_contrastive_loss(norm_embed, batch["words"])
                
                if is_pretrain_phase:
                    loss = (WAVLM_DISTILL_WEIGHT * loss_wavlm) + (WHISPER_DISTILL_WEIGHT * loss_whisper) + (CTC_LOSS_WEIGHT * loss_ctc)
                else:
                    loss = (loss_supcon + 
                            (TRUNCATION_AUX_WEIGHT * loss_tier_c) + 
                            (WAVLM_DISTILL_WEIGHT * loss_wavlm) + 
                            (WHISPER_DISTILL_WEIGHT * loss_whisper) + 
                            (CTC_LOSS_WEIGHT * loss_ctc))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            if rank == 0 and step % 10 == 0 and writer:
                global_step = epoch * len(train_loader) + step
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/Proto", loss_supcon.item() if not is_pretrain_phase else 0.0, global_step)
                writer.add_scalar("Loss/TierC_Truncation", loss_tier_c.item() if not is_pretrain_phase else 0.0, global_step)
                writer.add_scalar("Loss/WavLM_Distill", loss_wavlm.item(), global_step)
                writer.add_scalar("Loss/Whisper_Distill", loss_whisper.item(), global_step)
                writer.add_scalar("Loss/Phoneme_CTC", loss_ctc.item(), global_step)
                writer.add_scalar("Train/Learning_Rate", current_lr, global_step)
                writer.add_scalar("Accuracy/Proto", p_acc if not is_pretrain_phase else 0.0, global_step)
                
        scheduler.step()
        
        if rank == 0:
            val_loss = loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(OUTPUT_DIR, "best_sota_wakeword_model.pt"))
            else:
                if not is_pretrain_phase:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print("Early stopping triggered!")
                        break
            
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    elif world_size == 1:
        setup(0, 1)
        train(0, 1)
    else:
        print("No GPUs found for DDP training.")
