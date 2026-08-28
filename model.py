import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (EMBED_DIM, STAGE2_CHANNELS, NUM_ATTENTION_HEADS, 
                    WAVLM_EMBED_DIM, WHISPER_EMBED_DIM, NUM_PHONEMES)

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
        self.query_tokens = nn.Parameter(torch.randn(num_heads, in_dim))
        self.key_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.val_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * in_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        b, t, d = x.shape
        keys = self.key_proj(x)
        vals = self.val_proj(x)
        
        q = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        scores = torch.bmm(q, keys.transpose(1, 2)) / (d ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        pooled = torch.bmm(attn_weights, vals)
        pooled = pooled.reshape(b, -1)
        out = self.out_proj(pooled)
        out = self.layer_norm(out)
        return out

class WakeWordModel(nn.Module):
    """
    SOTA Wake Word Engine Student Model:
    - Dual Distillation Projections: WavLM-Large (1024D) + Whisper-Base (512D)
    - Phoneme CTC Frame Classification Head (42 classes)
    - Temporal Multi-Head Attention Head (128D)
    """
    def __init__(self, channels=STAGE2_CHANNELS, temporal_head="attention", embed_dim=EMBED_DIM,
                 wavlm_embed_dim=WAVLM_EMBED_DIM, whisper_embed_dim=WHISPER_EMBED_DIM,
                 num_phonemes=NUM_PHONEMES):
        super().__init__()
        self.encoder = DSCNNEncoder(in_channels=1, channels=channels)
        self.temporal_head_type = temporal_head
        self.channels = channels
        self.embed_dim = embed_dim
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((None, 1))
        
        if temporal_head == "attention":
            self.temporal = MultiHeadAttentionPooling(channels[-1], embed_dim)
        elif temporal_head == "gru":
            self.gru = nn.GRU(channels[-1], embed_dim, batch_first=True)
        elif temporal_head == "pool":
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(channels[-1], embed_dim)
        else:
            raise ValueError(f"Unknown temporal_head: {temporal_head}")

        # Dual Distillation Heads (Used during multi-task training)
        self.distill_wavlm_proj = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, wavlm_embed_dim)
        )
        self.distill_whisper_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, whisper_embed_dim)
        )

        # SOTA Phoneme CTC Prediction Head
        self.ctc_head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels[-1] // 2, num_phonemes),
            nn.LogSoftmax(dim=-1)
        )

    def extract_time_features(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feat = self.encoder(x)
        feat = self.spatial_pool(feat).squeeze(-1)
        feat = feat.transpose(1, 2) # (batch, time, channels)
        return feat

    def forward(self, x, return_distill=False, return_ctc=False):
        feat = self.extract_time_features(x)
        
        if self.temporal_head_type == "attention":
            embed = self.temporal(feat)
        elif self.temporal_head_type == "gru":
            output, _ = self.gru(feat)
            embed = output[:, -1, :]
        else: # pool
            feat_t = feat.transpose(1, 2)
            pooled = self.pool(feat_t).squeeze(-1)
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

    def forward_segments(self, x, num_segments=3):
        feat = self.extract_time_features(x)
        b, t, c = feat.shape
        seg_len = t // num_segments
        seg_embeds = []
        
        for i in range(num_segments):
            start = i * seg_len
            end = (i + 1) * seg_len if i < num_segments - 1 else t
            seg_feat = feat[:, start:end, :]
            
            if self.temporal_head_type == "attention":
                seg_out = self.temporal(seg_feat)
            elif self.temporal_head_type == "gru":
                _, h = self.gru(seg_feat)
                seg_out = h.squeeze(0)
            else:
                seg_pooled = self.pool(seg_feat.transpose(1, 2)).squeeze(-1)
                seg_out = self.fc(seg_pooled)
                
            seg_embeds.append(F.normalize(seg_out, p=2, dim=-1).unsqueeze(1))
            
        return torch.cat(seg_embeds, dim=1)
