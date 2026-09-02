import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (EMBED_DIM, STAGE2_CHANNELS, NUM_ATTENTION_HEADS, 
                    WAVLM_EMBED_DIM, WHISPER_EMBED_DIM, NUM_PHONEMES, SUFFIX_SPLIT)

class SinusoidalPositionalEncoding(nn.Module):
    def forward(self, x):  # x: [B, T, C]
        T, C = x.shape[1], x.shape[2]
        pos = torch.arange(T, device=x.device).unsqueeze(1)
        dim = torch.arange(0, C, 2, device=x.device).float()
        pe = torch.zeros(T, C, device=x.device)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / C)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / C)))
        return x + pe.unsqueeze(0)

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=False),
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
        self.silu = nn.SiLU(inplace=False)
        self.se = SqueezeExcitation(out_ch)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.silu(x)
        x = self.se(x)
        return x + res

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
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1)
            scores = scores.masked_fill(~mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)
        
        pooled = torch.bmm(attn_weights, vals)
        pooled = pooled.reshape(b, -1)
        out = self.out_proj(pooled)
        out = self.layer_norm(out)
        return out

class WakeWordModel(nn.Module):
    def __init__(self, channels=STAGE2_CHANNELS, temporal_head="attention", embed_dim=EMBED_DIM):
        super().__init__()
        self.encoder = DSCNNEncoder(in_channels=1, channels=channels)
        self.pos_encoder = SinusoidalPositionalEncoding()
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels[-1], nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.temporal_head_type = temporal_head
        
        if temporal_head == "attention":
            self.global_pool = MultiHeadAttentionPooling(channels[-1], embed_dim)
            self.suffix_pool = MultiHeadAttentionPooling(channels[-1], embed_dim)
        elif temporal_head == "gru":
            self.gru = nn.GRU(channels[-1], embed_dim, batch_first=True)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(channels[-1], embed_dim)

        self.distill_wavlm_proj = nn.Sequential(
            nn.Linear(channels[-1], 512, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(512, WAVLM_EMBED_DIM, bias=False)
        )
        self.distill_whisper_proj = nn.Sequential(
            nn.Linear(channels[-1], 256, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(256, WHISPER_EMBED_DIM, bias=False)
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
        feat = feat.mean(dim=-2)
        feat = feat.transpose(1, 2)
        feat = self.pos_encoder(feat)
        feat = self.transformer(feat)
        return feat

    def forward(self, x, mask=None, return_distill=False, return_ctc=False, return_suffix=False):
        feat = self.extract_time_features(x)
        
        if self.temporal_head_type == "attention":
            if mask is not None:
                mask_float = mask.float().unsqueeze(1)
                mask_down = F.adaptive_max_pool1d(mask_float, output_size=feat.size(1)).squeeze(1)
                bool_mask = mask_down > 0.5
            else:
                bool_mask = None
                
            global_out = self.global_pool(feat, mask=bool_mask)
            
            T = feat.shape[1]
            suffix_start = int(T * SUFFIX_SPLIT)
            suffix_feat = feat[:, suffix_start:, :]
            suffix_mask = bool_mask[:, suffix_start:] if bool_mask is not None else None
            suffix_out = self.suffix_pool(suffix_feat, mask=suffix_mask)
            
            global_embed = F.normalize(global_out, p=2, dim=-1)
            suffix_embed = F.normalize(suffix_out, p=2, dim=-1)
        elif self.temporal_head_type == "gru":
            out, _ = self.gru(feat)
            global_embed = F.normalize(out[:, -1, :], p=2, dim=-1)
            suffix_embed = global_embed
        else:
            pooled = self.pool(feat.transpose(1, 2)).squeeze(-1)
            global_embed = F.normalize(self.fc(pooled), p=2, dim=-1)
            suffix_embed = global_embed
            
        ctc_logits = self.ctc_head(feat) if (return_ctc or return_distill or return_suffix) else None
        
        res = []
        if return_distill:
            pooled_feat = feat.mean(dim=1)
            wavlm_proj = self.distill_wavlm_proj(pooled_feat)
            whisper_proj = self.distill_whisper_proj(pooled_feat)
            if return_suffix:
                return global_embed, suffix_embed, (wavlm_proj, whisper_proj), ctc_logits
            return global_embed, (wavlm_proj, whisper_proj), ctc_logits
            
        if return_suffix:
            if return_ctc:
                return global_embed, suffix_embed, ctc_logits
            return global_embed, suffix_embed
            
        if return_ctc:
            return global_embed, ctc_logits
            
        # Standard ONNX export mode: returns global_embed, ctc_logits
        # We need a stable output signature for ONNX that includes CTC.
        # If this is called without special flags but is tracing, we might want ctc_logits.
        # To avoid breaking existing things, default just returns global_embed,
        # but the ONNX export script will use return_ctc=True explicitly.
        return global_embed

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
                seg_out = self.global_pool(seg_feat)
            elif self.temporal_head_type == "gru":
                _, h = self.gru(seg_feat)
                seg_out = h.squeeze(0)
            else:
                seg_pooled = self.pool(seg_feat.transpose(1, 2)).squeeze(-1)
                seg_out = self.fc(seg_pooled)
                
            seg_embeds.append(F.normalize(seg_out, p=2, dim=-1).unsqueeze(1))
            
        return torch.cat(seg_embeds, dim=1)
