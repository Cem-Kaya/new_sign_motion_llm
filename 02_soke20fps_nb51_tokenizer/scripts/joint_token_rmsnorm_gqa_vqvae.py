from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scalar_token_spatiotemporal_vqvae import PerChannelUniformQuantizer
from local_quantize_ema_reset import QuantizeEMAReset
from joint_token_spatiotemporal_vqvae import (
    build_default_token_groups,
    count_parameters,
    default_group_indices_and_mask,
    group_scalar_token_ids,
    grouped_logits_to_expected_values,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class FeedForward(nn.Module):
    def __init__(self, width: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(round(width * float(mlp_ratio)))
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.width // self.num_heads
        self.dropout = float(dropout)
        self.kv_repeat = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(width, self.num_kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        return q.permute(0, 2, 1, 3)

    def _shape_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)
        return k, v

    def forward(
        self,
        query: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        *,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_value = query if key_value is None else key_value
        q = self._shape_q(query)
        k, v = self._shape_kv(key_value)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = scores.softmax(dim=-1)
        if self.training and self.dropout > 0.0:
            attn_drop = F.dropout(attn, p=self.dropout)
        else:
            attn_drop = attn
        out = torch.matmul(attn_drop, v)
        out = out.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], self.width)
        out = self.out_proj(out)
        return out, (attn if return_attn else None)


class GQATransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attn = GroupedQueryAttention(width, num_heads, num_kv_heads, dropout)
        self.norm2 = RMSNorm(width)
        self.ff = FeedForward(width, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x, attn


class GQACrossAttentionResampler(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        num_kv_heads: int,
        num_queries: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, width) * 0.02)
        self.query_norm = RMSNorm(width)
        self.source_norm = RMSNorm(width)
        self.attn = GroupedQueryAttention(width, num_heads, num_kv_heads, dropout)
        self.ff_norm = RMSNorm(width)
        self.ff = FeedForward(width, mlp_ratio, dropout)

    def forward(
        self,
        source: torch.Tensor,
        *,
        query_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch = source.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(batch, -1, -1)
        if query_bias is not None:
            queries = queries + query_bias
        out, attn = self.attn(self.query_norm(queries), self.source_norm(source), return_attn=return_attn)
        x = queries + out
        x = x + self.ff(self.ff_norm(x))
        return x, attn


@dataclass
class ModelConfig:
    num_scalar_features: int = 277
    num_tokens_per_frame: int = 46
    num_bins: int = 128
    max_group_dim: int = 10
    window_size: int = 64
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_spatial_latents: int = 32
    num_temporal_latents: int = 16
    code_dim: int = 512
    code_num: int = 512
    vq_mu: float = 0.99
    spatial_blocks: int = 2
    temporal_blocks: int = 2


class JointTokenRMSGQAVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)

        groups = build_default_token_groups()
        if len(groups) != cfg.num_tokens_per_frame:
            raise ValueError(
                f"Config num_tokens_per_frame={cfg.num_tokens_per_frame} does not match default groups={len(groups)}"
            )
        group_indices_np, group_valid_mask_np = default_group_indices_and_mask(cfg.max_group_dim, groups)
        self.group_names = [g["group_name"] for g in groups]
        self.group_kinds = [g["kind"] for g in groups]
        self.register_buffer("group_indices", torch.tensor(group_indices_np, dtype=torch.long), persistent=False)
        self.register_buffer("group_valid_mask", torch.tensor(group_valid_mask_np, dtype=torch.bool), persistent=False)

        self.value_embed = nn.Embedding(cfg.num_bins, d)
        self.within_group_pos_embed = nn.Parameter(torch.randn(cfg.max_group_dim, d) * 0.02)
        self.token_embed = nn.Parameter(torch.randn(cfg.num_tokens_per_frame, d) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(cfg.window_size, d) * 0.02)
        self.group_in_proj = nn.Linear(cfg.max_group_dim * d, d, bias=False)
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.spatial_encoder = nn.ModuleList(
            [
                GQATransformerBlock(d, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.spatial_blocks)
            ]
        )
        self.spatial_pool = GQACrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.num_spatial_latents,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.temporal_encoder = nn.ModuleList(
            [
                GQATransformerBlock(d, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.temporal_blocks)
            ]
        )
        self.temporal_pool = GQACrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.num_temporal_latents,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.to_code = nn.Linear(d, cfg.code_dim)
        self.quantizer = QuantizeEMAReset(cfg.code_num, cfg.code_dim, mu=cfg.vq_mu)
        self.from_code = nn.Linear(cfg.code_dim, d)

        self.temporal_decoder = nn.ModuleList(
            [
                GQATransformerBlock(d, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.temporal_blocks)
            ]
        )
        self.temporal_upsample = GQACrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.spatial_decoder = nn.ModuleList(
            [
                GQATransformerBlock(d, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout)
                for _ in range(cfg.spatial_blocks)
            ]
        )
        self.spatial_upsample = GQACrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.num_tokens_per_frame,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.out_norm = RMSNorm(d)
        self.logit_head = nn.Linear(d, cfg.max_group_dim * cfg.num_bins)

    def _run_blocks(
        self,
        blocks: nn.ModuleList,
        x: torch.Tensor,
        *,
        return_attn: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_map = None
        for idx, block in enumerate(blocks):
            need = return_attn and idx == (len(blocks) - 1)
            x, attn_map = block(x, return_attn=need)
        return x, attn_map

    def _embed_group_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.value_embed(token_ids)
        x = x + self.within_group_pos_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = self.group_valid_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        x = x * mask
        x = x.reshape(
            token_ids.shape[0],
            token_ids.shape[1],
            token_ids.shape[2],
            self.cfg.max_group_dim * self.cfg.d_model,
        )
        x = self.group_in_proj(x)
        x = x + self.token_embed.unsqueeze(0).unsqueeze(0)
        x = x + self.time_embed.unsqueeze(0).unsqueeze(2)
        return self.embed_drop(x)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        capture_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if token_ids.ndim != 4:
            raise ValueError(f"Expected [B,T,G,D] token ids, got {tuple(token_ids.shape)}")
        bsz, time_len, num_tokens, group_dim = token_ids.shape
        if time_len != self.cfg.window_size or num_tokens != self.cfg.num_tokens_per_frame or group_dim != self.cfg.max_group_dim:
            raise ValueError(
                "Expected "
                f"[B,{self.cfg.window_size},{self.cfg.num_tokens_per_frame},{self.cfg.max_group_dim}] token ids, "
                f"got {tuple(token_ids.shape)}"
            )

        x = self._embed_group_tokens(token_ids)
        x_spatial = x.reshape(bsz * time_len, num_tokens, self.cfg.d_model)
        x_spatial, spatial_attn = self._run_blocks(self.spatial_encoder, x_spatial, return_attn=capture_attention)
        x = x_spatial.reshape(bsz, time_len, num_tokens, self.cfg.d_model)

        x_pool_in = x.reshape(bsz * time_len, num_tokens, self.cfg.d_model)
        x_pool, spatial_pool_attn = self.spatial_pool(x_pool_in, return_attn=capture_attention)
        x = x_pool.reshape(bsz, time_len, self.cfg.num_spatial_latents, self.cfg.d_model)

        x_temporal = x.permute(0, 2, 1, 3).reshape(bsz * self.cfg.num_spatial_latents, time_len, self.cfg.d_model)
        x_temporal, temporal_attn = self._run_blocks(self.temporal_encoder, x_temporal, return_attn=capture_attention)
        x = x_temporal.reshape(bsz, self.cfg.num_spatial_latents, time_len, self.cfg.d_model)

        x_time_pool, temporal_pool_attn = self.temporal_pool(
            x.reshape(bsz * self.cfg.num_spatial_latents, time_len, self.cfg.d_model),
            return_attn=capture_attention,
        )
        x = x_time_pool.reshape(
            bsz,
            self.cfg.num_spatial_latents,
            self.cfg.num_temporal_latents,
            self.cfg.d_model,
        ).permute(0, 2, 1, 3)

        z = self.to_code(x)
        z_vq_in = z.permute(0, 3, 1, 2).reshape(
            bsz,
            self.cfg.code_dim,
            self.cfg.num_temporal_latents * self.cfg.num_spatial_latents,
        )
        z_vq_out, commit_loss, perplexity = self.quantizer(z_vq_in)
        code_indices = self.quantizer.quantize(self.quantizer.preprocess(z_vq_in)).reshape(
            bsz,
            self.cfg.num_temporal_latents,
            self.cfg.num_spatial_latents,
        )

        z = z_vq_out.reshape(
            bsz,
            self.cfg.code_dim,
            self.cfg.num_temporal_latents,
            self.cfg.num_spatial_latents,
        ).permute(0, 2, 3, 1)
        x = self.from_code(z)

        x_tdec = x.permute(0, 2, 1, 3).reshape(
            bsz * self.cfg.num_spatial_latents,
            self.cfg.num_temporal_latents,
            self.cfg.d_model,
        )
        x_tdec, temporal_dec_attn = self._run_blocks(self.temporal_decoder, x_tdec, return_attn=capture_attention)
        x = x_tdec.reshape(
            bsz,
            self.cfg.num_spatial_latents,
            self.cfg.num_temporal_latents,
            self.cfg.d_model,
        )

        time_bias = self.time_embed.unsqueeze(0).expand(bsz * self.cfg.num_spatial_latents, -1, -1)
        x_time_up, temporal_up_attn = self.temporal_upsample(
            x.reshape(bsz * self.cfg.num_spatial_latents, self.cfg.num_temporal_latents, self.cfg.d_model),
            query_bias=time_bias,
            return_attn=capture_attention,
        )
        x = x_time_up.reshape(
            bsz,
            self.cfg.num_spatial_latents,
            self.cfg.window_size,
            self.cfg.d_model,
        ).permute(0, 2, 1, 3)

        x_sdec = x.reshape(bsz * self.cfg.window_size, self.cfg.num_spatial_latents, self.cfg.d_model)
        x_sdec, spatial_dec_attn = self._run_blocks(self.spatial_decoder, x_sdec, return_attn=capture_attention)
        x = x_sdec.reshape(bsz, self.cfg.window_size, self.cfg.num_spatial_latents, self.cfg.d_model)

        token_bias = self.token_embed.unsqueeze(0).expand(bsz * self.cfg.window_size, -1, -1)
        x_spatial_up, spatial_up_attn = self.spatial_upsample(
            x.reshape(bsz * self.cfg.window_size, self.cfg.num_spatial_latents, self.cfg.d_model),
            query_bias=token_bias,
            return_attn=capture_attention,
        )
        x = x_spatial_up.reshape(bsz, self.cfg.window_size, self.cfg.num_tokens_per_frame, self.cfg.d_model)
        logits = self.logit_head(self.out_norm(x)).reshape(
            bsz,
            self.cfg.window_size,
            self.cfg.num_tokens_per_frame,
            self.cfg.max_group_dim,
            self.cfg.num_bins,
        )

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "commit_loss": commit_loss,
            "perplexity": perplexity,
            "code_indices": code_indices,
        }
        if capture_attention:
            out.update(
                {
                    "spatial_attn": spatial_attn.reshape(
                        bsz,
                        time_len,
                        spatial_attn.shape[1],
                        spatial_attn.shape[2],
                        spatial_attn.shape[3],
                    ),
                    "spatial_pool_attn": spatial_pool_attn.reshape(
                        bsz,
                        time_len,
                        spatial_pool_attn.shape[1],
                        spatial_pool_attn.shape[2],
                        spatial_pool_attn.shape[3],
                    ),
                    "temporal_attn": temporal_attn.reshape(
                        bsz,
                        self.cfg.num_spatial_latents,
                        temporal_attn.shape[1],
                        temporal_attn.shape[2],
                        temporal_attn.shape[3],
                    ),
                    "temporal_pool_attn": temporal_pool_attn.reshape(
                        bsz,
                        self.cfg.num_spatial_latents,
                        temporal_pool_attn.shape[1],
                        temporal_pool_attn.shape[2],
                        temporal_pool_attn.shape[3],
                    ),
                    "temporal_dec_attn": temporal_dec_attn.reshape(
                        bsz,
                        self.cfg.num_spatial_latents,
                        temporal_dec_attn.shape[1],
                        temporal_dec_attn.shape[2],
                        temporal_dec_attn.shape[3],
                    ),
                    "temporal_up_attn": temporal_up_attn.reshape(
                        bsz,
                        self.cfg.num_spatial_latents,
                        temporal_up_attn.shape[1],
                        temporal_up_attn.shape[2],
                        temporal_up_attn.shape[3],
                    ),
                    "spatial_dec_attn": spatial_dec_attn.reshape(
                        bsz,
                        time_len,
                        spatial_dec_attn.shape[1],
                        spatial_dec_attn.shape[2],
                        spatial_dec_attn.shape[3],
                    ),
                    "spatial_up_attn": spatial_up_attn.reshape(
                        bsz,
                        time_len,
                        spatial_up_attn.shape[1],
                        spatial_up_attn.shape[2],
                        spatial_up_attn.shape[3],
                    ),
                }
            )
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ModelConfig",
    "JointTokenRMSGQAVQVAE",
    "PerChannelUniformQuantizer",
    "build_default_token_groups",
    "default_group_indices_and_mask",
    "group_scalar_token_ids",
    "grouped_logits_to_expected_values",
    "count_parameters",
]
