from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from local_quantize_ema_reset import QuantizeEMAReset


class PerChannelUniformQuantizer:
    def __init__(
        self,
        num_bins: int = 128,
        clip_low_pct: float = 0.5,
        clip_high_pct: float = 99.5,
    ) -> None:
        self.num_bins = int(num_bins)
        self.clip_low_pct = float(clip_low_pct)
        self.clip_high_pct = float(clip_high_pct)
        self.low: Optional[np.ndarray] = None
        self.high: Optional[np.ndarray] = None
        self.step: Optional[np.ndarray] = None
        self.centers: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "PerChannelUniformQuantizer":
        x = np.asarray(values, dtype=np.float32)
        if x.ndim < 2:
            raise ValueError(f"Expected at least 2 dims, got {x.shape}")
        x = x.reshape(-1, x.shape[-1])
        low = np.percentile(x, self.clip_low_pct, axis=0).astype(np.float32)
        high = np.percentile(x, self.clip_high_pct, axis=0).astype(np.float32)
        span = np.maximum(high - low, 1e-6).astype(np.float32)
        high = (low + span).astype(np.float32)
        step = (span / float(self.num_bins)).astype(np.float32)
        centers = low[None, :] + (np.arange(self.num_bins, dtype=np.float32)[:, None] + 0.5) * step[None, :]

        self.low = low
        self.high = high
        self.step = step
        self.centers = centers.T.astype(np.float32)
        return self

    def _check_fitted(self) -> None:
        if self.low is None or self.high is None or self.step is None or self.centers is None:
            raise RuntimeError("Quantizer is not fitted")

    def encode(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(values, dtype=np.float32)
        lo = self.low.reshape(*([1] * (x.ndim - 1)), -1)
        hi = self.high.reshape(*([1] * (x.ndim - 1)), -1)
        step = self.step.reshape(*([1] * (x.ndim - 1)), -1)
        clipped = np.clip(x, lo, hi)
        idx = np.floor((clipped - lo) / step).astype(np.int64)
        idx = np.clip(idx, 0, self.num_bins - 1)
        return idx

    def decode(self, indices: np.ndarray) -> np.ndarray:
        self._check_fitted()
        idx = np.asarray(indices, dtype=np.int64)
        out = np.take_along_axis(
            self.centers.reshape(*([1] * (idx.ndim - 1)), *self.centers.shape),
            idx[..., None],
            axis=-1,
        )
        return out[..., 0].astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        self._check_fitted()
        return {
            "num_bins": self.num_bins,
            "clip_low_pct": self.clip_low_pct,
            "clip_high_pct": self.clip_high_pct,
            "low": self.low.astype(np.float32),
            "high": self.high.astype(np.float32),
            "step": self.step.astype(np.float32),
            "centers": self.centers.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "PerChannelUniformQuantizer":
        obj = cls(
            num_bins=int(state["num_bins"]),
            clip_low_pct=float(state["clip_low_pct"]),
            clip_high_pct=float(state["clip_high_pct"]),
        )
        obj.low = np.asarray(state["low"], dtype=np.float32)
        obj.high = np.asarray(state["high"], dtype=np.float32)
        obj.step = np.asarray(state["step"], dtype=np.float32)
        obj.centers = np.asarray(state["centers"], dtype=np.float32)
        return obj


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


class TransformerBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(width)
        self.ff = FeedForward(width, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self.norm1(x)
        out, attn = self.attn(
            q,
            q,
            q,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x, attn


class CrossAttentionResampler(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        num_queries: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, width) * 0.02)
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(width)
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
        q = self.query_norm(queries)
        kv = self.source_norm(source)
        out, attn = self.attn(
            q,
            kv,
            kv,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = queries + out
        x = x + self.ff(self.ff_norm(x))
        return x, attn


@dataclass
class ModelConfig:
    num_features: int = 277
    num_bins: int = 128
    window_size: int = 64
    d_model: int = 256
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    num_spatial_latents: int = 64
    num_temporal_latents: int = 16
    code_dim: int = 512
    code_num: int = 512
    vq_mu: float = 0.99
    spatial_blocks: int = 2
    temporal_blocks: int = 2


class ScalarTokenSpatiotemporalVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_model)

        self.value_embed = nn.Embedding(cfg.num_bins, d)
        self.channel_embed = nn.Parameter(torch.randn(cfg.num_features, d) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(cfg.window_size, d) * 0.02)
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.spatial_encoder = nn.ModuleList(
            [TransformerBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.spatial_blocks)]
        )
        self.spatial_pool = CrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.num_spatial_latents,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.temporal_encoder = nn.ModuleList(
            [TransformerBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.temporal_blocks)]
        )
        self.temporal_pool = CrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.num_temporal_latents,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.to_code = nn.Linear(d, cfg.code_dim)
        self.quantizer = QuantizeEMAReset(cfg.code_num, cfg.code_dim, mu=cfg.vq_mu)
        self.from_code = nn.Linear(cfg.code_dim, d)

        self.temporal_decoder = nn.ModuleList(
            [TransformerBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.temporal_blocks)]
        )
        self.temporal_upsample = CrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.window_size,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.spatial_decoder = nn.ModuleList(
            [TransformerBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.spatial_blocks)]
        )
        self.spatial_upsample = CrossAttentionResampler(
            d,
            cfg.num_heads,
            cfg.num_features,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.out_norm = nn.LayerNorm(d)
        self.logit_head = nn.Linear(d, cfg.num_bins)

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

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        capture_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if token_ids.ndim != 3:
            raise ValueError(f"Expected [B,T,F] token ids, got {tuple(token_ids.shape)}")
        bsz, time_len, num_feat = token_ids.shape
        if time_len != self.cfg.window_size or num_feat != self.cfg.num_features:
            raise ValueError(
                f"Expected [B,{self.cfg.window_size},{self.cfg.num_features}] token ids, got {tuple(token_ids.shape)}"
            )

        x = self.value_embed(token_ids)
        x = x + self.channel_embed.unsqueeze(0).unsqueeze(0)
        x = x + self.time_embed.unsqueeze(0).unsqueeze(2)
        x = self.embed_drop(x)

        x_spatial = x.reshape(bsz * time_len, num_feat, self.cfg.d_model)
        x_spatial, spatial_attn = self._run_blocks(
            self.spatial_encoder,
            x_spatial,
            return_attn=capture_attention,
        )
        x = x_spatial.reshape(bsz, time_len, num_feat, self.cfg.d_model)

        x_pool_in = x.reshape(bsz * time_len, num_feat, self.cfg.d_model)
        x_pool, spatial_pool_attn = self.spatial_pool(
            x_pool_in,
            return_attn=capture_attention,
        )
        x = x_pool.reshape(bsz, time_len, self.cfg.num_spatial_latents, self.cfg.d_model)

        x_temporal = x.permute(0, 2, 1, 3).reshape(
            bsz * self.cfg.num_spatial_latents,
            time_len,
            self.cfg.d_model,
        )
        x_temporal, temporal_attn = self._run_blocks(
            self.temporal_encoder,
            x_temporal,
            return_attn=capture_attention,
        )
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

        flat_z = self.quantizer.preprocess(z_vq_in.detach())
        code_indices = self.quantizer.quantize(flat_z).view(
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
        x_tdec, temporal_dec_attn = self._run_blocks(
            self.temporal_decoder,
            x_tdec,
            return_attn=capture_attention,
        )
        x = x_tdec.reshape(
            bsz,
            self.cfg.num_spatial_latents,
            self.cfg.num_temporal_latents,
            self.cfg.d_model,
        )

        time_bias = self.time_embed.unsqueeze(0).expand(
            bsz * self.cfg.num_spatial_latents,
            -1,
            -1,
        )
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
        x_sdec, spatial_dec_attn = self._run_blocks(
            self.spatial_decoder,
            x_sdec,
            return_attn=capture_attention,
        )
        x = x_sdec.reshape(bsz, self.cfg.window_size, self.cfg.num_spatial_latents, self.cfg.d_model)

        channel_bias = self.channel_embed.unsqueeze(0).expand(
            bsz * self.cfg.window_size,
            -1,
            -1,
        )
        x_spatial_up, spatial_up_attn = self.spatial_upsample(
            x.reshape(bsz * self.cfg.window_size, self.cfg.num_spatial_latents, self.cfg.d_model),
            query_bias=channel_bias,
            return_attn=capture_attention,
        )
        x = x_spatial_up.reshape(bsz, self.cfg.window_size, self.cfg.num_features, self.cfg.d_model)
        logits = self.logit_head(self.out_norm(x))

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


def logits_to_expected_values(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected [B,T,F,Bins] logits, got {tuple(logits.shape)}")
    if centers.ndim != 2:
        raise ValueError(f"Expected [F,Bins] centers, got {tuple(centers.shape)}")
    probs = torch.softmax(logits, dim=-1)
    centers_t = centers.to(device=logits.device, dtype=logits.dtype)
    return (probs * centers_t.unsqueeze(0).unsqueeze(0)).sum(dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
