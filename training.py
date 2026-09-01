import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

from config import (SR, WARMUP_EPOCHS, MIN_LR, GRAD_CLIP_NORM,
                    TRUNCATION_AUX_WEIGHT, WAVLM_DISTILL_WEIGHT, 
                    WHISPER_DISTILL_WEIGHT, CTC_LOSS_WEIGHT)
from model import WakeWordModel
from dataset import (MSWCTrainingDataset, collate_fn, load_speech_dataset, 
                     load_background_noise_bank)
from losses import truncation_margin_loss, cosine_distillation_loss, supervised_contrastive_loss

def _fast_cache_collate(batch):
    import torch
    import numpy as np
    from audio_utils import pad_or_trim
    from config import SR, WHISPER_TEACHER_MODEL
    
    global _worker_whisper_extractor, _worker_resamplers
    if '_worker_whisper_extractor' not in globals():
        from transformers import WhisperFeatureExtractor
        global _worker_whisper_extractor
        _worker_whisper_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_TEACHER_MODEL)
    if '_worker_resamplers' not in globals():
        global _worker_resamplers
        _worker_resamplers = {}
        
    target_samples = int(1.2 * SR)
    wavs = []
    
    for item in batch:
        audio = item["audio"]
        wav_t = torch.tensor(audio["array"], dtype=torch.float32)
        in_sr = audio["sampling_rate"]
        
        if in_sr != SR:
            import torchaudio
            if in_sr not in _worker_resamplers:
                _worker_resamplers[in_sr] = torchaudio.transforms.Resample(in_sr, SR)
            wav_t = _worker_resamplers[in_sr](wav_t)
            
        wavs.append(pad_or_trim(wav_t.numpy(), target_samples))
        
    wavs_np = np.stack(wavs, axis=0)
    wavs_t = torch.tensor(wavs_np, dtype=torch.float16)
    wh_inputs = _worker_whisper_extractor(list(wavs_np), sampling_rate=SR, return_tensors="pt").input_features.half()
    
    return wavs_t, wh_inputs


def precompute_teacher_features(ds, wavlm_model_name, whisper_model_name, device, batch_size=1024,
                                cache_dir: str = "./teacher_cache"):
    """
    Max-speed 1-pass teacher embedding cache.
    Optimizations:
    1. Disk cache keyed by (num_samples, wavlm_model, whisper_model) — skips recomputation on restart.
    2. DataLoader extracts Whisper features in parallel CPU workers
    3. DataParallel scales across all available GPUs (e.g. 2x T4 on Kaggle)
    """
    import time
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import WavLMModel, WhisperModel, WhisperFeatureExtractor
    from config import SR

    # ── Disk cache: build a deterministic key ───────────────────────────────
    os.makedirs(cache_dir, exist_ok=True)
    _wlm_tag = wavlm_model_name.replace("/", "_")
    _wh_tag  = whisper_model_name.replace("/", "_")
    cache_path = os.path.join(cache_dir, f"teacher_{len(ds)}_{_wlm_tag}_{_wh_tag}.pt")

    if os.path.exists(cache_path):
        print(f"\n💾 Loading cached teacher embeddings from:\n   {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        print(f"✅ Cache loaded — {len(ds):,} samples  "
              f"(wavlm {tuple(data['wavlm'].shape)}, whisper {tuple(data['whisper'].shape)})")
        return data
    # ────────────────────────────────────────────────────────────────────────

    print(f"\n⚡ Pre-extracting Teacher Embeddings (Max Speed: Multi-GPU + Multi-Core)...")
    print(f"  • WavLM  : {wavlm_model_name}")
    print(f"  • Whisper: {whisper_model_name}")
    print(f"  • Samples: {len(ds):,}   Batch: {batch_size}   GPUs: {torch.cuda.device_count()}")

    wavlm = WavLMModel.from_pretrained(wavlm_model_name).to(device).half()
    wavlm.eval()
    for p in wavlm.parameters(): p.requires_grad = False

    whisper = WhisperModel.from_pretrained(whisper_model_name).encoder.to(device).half()
    whisper.eval()
    for p in whisper.parameters(): p.requires_grad = False

    if torch.cuda.device_count() > 1:
        wavlm = nn.DataParallel(wavlm)
        whisper = nn.DataParallel(whisper)

    workers = min(8, os.cpu_count() or 8)
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=_fast_cache_collate, 
        num_workers=workers, 
        pin_memory=True, 
        shuffle=False
    )

    wavlm_targets = []
    whisper_targets = []
    num_samples = len(ds)
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (wavs_t, wh_inputs) in enumerate(loader):
            wavs_t = wavs_t.to(device, non_blocking=True)
            wh_inputs = wh_inputs.to(device, non_blocking=True)
            
            # WavLM Inference
            w_out = wavlm(wavs_t)
            if hasattr(w_out, 'last_hidden_state'): w_out = w_out.last_hidden_state
            w_out = w_out.mean(dim=1).cpu()
            wavlm_targets.append(w_out)
            
            # Whisper Inference
            wh_out = whisper(wh_inputs)
            if hasattr(wh_out, 'last_hidden_state'): wh_out = wh_out.last_hidden_state
            wh_out = wh_out.mean(dim=1).cpu()
            whisper_targets.append(wh_out)
            
            samples_done = min((batch_idx + 1) * batch_size, num_samples)
            if (batch_idx + 1) % 1 == 0 or samples_done == num_samples:
                elapsed = time.time() - t_start
                rate = samples_done / elapsed
                remaining = (num_samples - samples_done) / rate if rate > 0 else 0
                print(f"  ✓ {samples_done:>7,}/{num_samples:,}  "
                      f"[{elapsed:5.0f}s elapsed | ETA {remaining:5.0f}s | "
                      f"{rate:6.1f} samples/s]")

    # Return to FP32 to match expected model/loss output types
    wavlm_all = torch.cat(wavlm_targets, dim=0).float()
    whisper_all = torch.cat(whisper_targets, dim=0).float()

    mem_mb = (wavlm_all.element_size() * wavlm_all.nelement() +
              whisper_all.element_size() * whisper_all.nelement()) / 1e6
    total_elapsed = time.time() - t_start
    print(f"✅ Caching complete — {num_samples:,} samples in {total_elapsed:.1f}s "
          f"({num_samples/total_elapsed:.0f} samples/s) | RAM: {mem_mb:.1f} MB")

    del wavlm, whisper
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    result = {"wavlm": wavlm_all, "whisper": whisper_all}
    print(f"💾 Saving teacher cache to:\n   {cache_path}  ({mem_mb:.1f} MB)")
    torch.save(result, cache_path)
    print(f"✅ Cache saved — future runs will skip recomputation.")

    return result

def run_training(
    noise_dir: str = "./noise",
    output_dir: str = "./output",
    dataset_path: str = None,
    resume_path: str = None,
    mswc_split: str = "train[:10000]",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 5,
    wavlm_model: str = "microsoft/wavlm-base-plus",
    whisper_model: str = "openai/whisper-base",
    cache_teachers: bool = True,
    teacher_cache_dir: str = "./teacher_cache"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    ds = load_speech_dataset(mswc_split, dataset_path=dataset_path)
    noise_bank = load_background_noise_bank(noise_dir)
    
    teacher_targets = None
    if cache_teachers:
        teacher_targets = precompute_teacher_features(ds, wavlm_model, whisper_model, device, batch_size=1024,
                                                      cache_dir=teacher_cache_dir)
    
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    
    torch.backends.cudnn.benchmark = True
    
    train_loader = DataLoader(
        MSWCTrainingDataset(train_ds, noise_clips=noise_bank, teacher_targets=teacher_targets, indices=train_ds.indices if teacher_targets else None),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        MSWCTrainingDataset(val_ds, noise_clips=noise_bank, teacher_targets=teacher_targets, indices=val_ds.indices if teacher_targets else None),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    student = WakeWordModel().to(device)
    
    wavlm, whisper, whisper_extractor = None, None, None
    if not cache_teachers:
        from transformers import WavLMModel, WhisperModel, WhisperFeatureExtractor
        wavlm = WavLMModel.from_pretrained(wavlm_model).to(device)
        wavlm.eval()
        for p in wavlm.parameters(): p.requires_grad = False
        
        whisper_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
        whisper = WhisperModel.from_pretrained(whisper_model).encoder.to(device)
        whisper.eval()
        for p in whisper.parameters(): p.requires_grad = False
    
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

    target_resume = resume_path or os.path.join(output_dir, "best_sota_wakeword_model.pt")
    if os.path.exists(target_resume):
        checkpoint = torch.load(target_resume, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'val_metric' in checkpoint:
            best_val_loss = checkpoint['val_metric']
    
    for epoch in range(start_epoch, epochs):
        student.train()
        is_pretrain = (epoch < 5)
        total_epoch_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            mels = batch["mels"].to(device, non_blocking=True)
            trunc_mels = batch["trunc_mels"].to(device, non_blocking=True)
            wavs_np = np.stack(batch["wavs"], axis=0)
            wavs_t = torch.tensor(wavs_np, dtype=torch.float32, device=device)
            
            energy = mels.mean(dim=-1).squeeze(1)
            attention_mask = energy > -10.0
            
            if "wavlm_targets" in batch:
                wavlm_target = batch["wavlm_targets"].to(device, non_blocking=True)
                whisper_target = batch["whisper_targets"].to(device, non_blocking=True)
            else:
                with torch.no_grad(), autocast():
                    wavlm_out = wavlm(wavs_t).last_hidden_state
                    wavlm_target = wavlm_out.mean(dim=1)
                    whisper_inputs = whisper_extractor(
                        [w for w in wavs_np], sampling_rate=SR, return_tensors="pt"
                    ).input_features.to(device, non_blocking=True)
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
                loss_supcon, p_acc = supervised_contrastive_loss(norm_embed, batch["words"])
                
                if is_pretrain:
                    loss = (WAVLM_DISTILL_WEIGHT * loss_wavlm) + (WHISPER_DISTILL_WEIGHT * loss_whisper) + (CTC_LOSS_WEIGHT * loss_ctc)
                else:
                    loss = (loss_supcon + 
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
                print(f"Step {step:03d} | Loss: {loss.item():.4f} | Distill: {(loss_wavlm+loss_whisper).item():.3f} | TierC: {loss_tier_c.item():.3f} | Proto: {loss_supcon.item():.3f}")
                
        scheduler.step()
        avg_train_loss = total_epoch_loss / len(train_loader)
        
        student.eval()
        val_trunc_loss = 0.0
        val_supcon_loss = 0.0
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
                p_loss, _ = supervised_contrastive_loss(v_embed, v_batch["words"])
                val_supcon_loss += p_loss.item()
                val_batches += 1
                
        if val_batches > 0:
            val_trunc_loss /= val_batches
            val_supcon_loss /= val_batches
        val_combined = val_trunc_loss + val_supcon_loss
        
        print(f"Epoch {epoch+1} Done | Train Loss: {avg_train_loss:.4f} | Val Trunc: {val_trunc_loss:.4f} | Val SupCon: {val_supcon_loss:.4f}")
        
        if val_combined < best_val_loss:
            best_val_loss = val_combined
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'val_metric': best_val_loss
            }, os.path.join(output_dir, "best_sota_wakeword_model.pt"))
        else:
            if not is_pretrain:
                patience += 1
                if patience >= early_stopping_patience:
                    break
                    
    print("Exporting ONNX...")
    student.load_state_dict(torch.load(os.path.join(output_dir, "best_sota_wakeword_model.pt"), map_location=device)['model_state_dict'])
    student.eval()
    
    dummy_input = torch.randn(1, 1, int(1.2 * SR / 160), 40).to(device)
    onnx_path = os.path.join(output_dir, "wakeword_student.onnx")
    
    try:
        torch.onnx.export(
            student, dummy_input, onnx_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['input_mel'], output_names=['embedding'],
            dynamic_axes={'input_mel': {0: 'batch_size', 2: 'time'}, 'embedding': {0: 'batch_size'}}
        )
        print(f"ONNX Exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX Export failed: {e}")

    # Export teacher cache to output dir if it exists
    if cache_teachers:
        import shutil
        _wlm_tag = wavlm_model.replace("/", "_")
        _wh_tag  = whisper_model.replace("/", "_")
        cache_file = f"teacher_{len(ds)}_{_wlm_tag}_{_wh_tag}.pt"
        src_cache = os.path.join(teacher_cache_dir, cache_file)
        if os.path.exists(src_cache):
            dst_cache = os.path.join(output_dir, cache_file)
            print(f"Copying teacher cache to output directory: {dst_cache}")
            shutil.copy2(src_cache, dst_cache)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_dir", type=str, default="./noise")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--mswc_split", type=str, default="train[:10000]")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--wavlm_model", type=str, default="microsoft/wavlm-base-plus")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-base")
    parser.add_argument("--no_cache_teachers", action="store_true")
    parser.add_argument("--teacher_cache_dir", type=str, default="./teacher_cache",
                        help="Directory to store/load precomputed teacher embeddings on disk.")
    
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
        whisper_model=args.whisper_model,
        cache_teachers=not args.no_cache_teachers,
        teacher_cache_dir=args.teacher_cache_dir,
    )
