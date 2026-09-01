import torch
import torch.nn.functional as F
import numpy as np

def truncation_margin_loss(embed_full, embed_truncated, margin=0.8):
    diff = embed_full - embed_truncated
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
    return F.relu(margin - dist).mean()

def cosine_distillation_loss(student_proj, teacher_embed):
    s_norm = F.normalize(student_proj, p=2, dim=-1)
    t_norm = F.normalize(teacher_embed, p=2, dim=-1)
    return (1.0 - (s_norm * t_norm).sum(dim=-1)).mean()

def supervised_contrastive_loss(embeds, words, temperature=0.1):
    device = embeds.device
    B = embeds.shape[0]
    if B < 2:
        return (embeds.sum() * 0.0), 0.0
        
    embeds = F.normalize(embeds, p=2, dim=-1)
    sim_matrix = torch.matmul(embeds, embeds.T) / temperature
    sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - sim_max.detach()
    labels = np.array(words)
    mask = torch.tensor(labels[:, None] == labels[None, :], dtype=torch.float32, device=device)
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(B, device=device).view(-1, 1), 0)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mask_sum = mask.sum(1)
    valid_mask = mask_sum > 0
    if not valid_mask.any():
        sq_dist = torch.pdist(embeds, p=2).pow(2)
        return torch.log(torch.mean(torch.exp(-2.0 * sq_dist))), 0.0
    mean_log_prob_pos = (mask * log_prob).sum(1)[valid_mask] / mask_sum[valid_mask]
    loss = -mean_log_prob_pos.mean()
    sim_matrix.fill_diagonal_(-1e4)
    top1_idx = sim_matrix.argmax(dim=1)
    top1_is_pos = mask[torch.arange(B), top1_idx]
    acc = top1_is_pos[valid_mask].mean().item()
    return loss, acc
