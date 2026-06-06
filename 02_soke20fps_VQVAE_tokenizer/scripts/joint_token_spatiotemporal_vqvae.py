from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from scalar_token_spatiotemporal_vqvae import (
    CrossAttentionResampler,
    PerChannelUniformQuantizer,
    TransformerBlock,
)
from local_quantize_ema_reset import QuantizeEMAReset


def build_default_token_groups() -> List[Dict[str, Any]]:
    body_names = [
        "pelvis",
        "spine1",
        "spine2",
        "spine3",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]

    groups: List[Dict[str, Any]] = []
    for i, name in enumerate(body_names):
        s = i * 6
        groups.append(
            {
                "group_name": name,
                "kind": "body_joint_6d",
                "indices": tuple(range(s, s + 6)),
            }
        )
    groups.append({"group_name": "jaw", "kind": "jaw_3d", "indices": tuple(range(84, 87))})
    groups.append({"group_name": "expression", "kind": "expression_10d", "indices": tuple(range(87, 97))})
    for i in range(15):
        s = 97 + i * 6
        groups.append(
            {
                "group_name": f"left_hand_joint_{i:02d}",
                "kind": "left_hand_6d",
                "indices": tuple(range(s, s + 6)),
            }
        )
    for i in range(15):
        s = 187 + i * 6
        groups.append(
            {
                "group_name": f"right_hand_joint_{i:02d}",
                "kind": "right_hand_6d",
                "indices": tuple(range(s, s + 6)),
            }
        )
    return groups


def default_group_indices_and_mask(
    max_group_dim: int = 10,
    groups: Optional[Sequence[Dict[str, Any]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    groups = list(groups) if groups is not None else build_default_token_groups()
    indices = np.full((len(groups), max_group_dim), -1, dtype=np.int64)
    mask = np.zeros((len(groups), max_group_dim), dtype=bool)
    for gi, group in enumerate(groups):
        group_idx = np.asarray(group["indices"], dtype=np.int64)
        if group_idx.size > max_group_dim:
            raise ValueError(f"Group {group['group_name']} exceeds max_group_dim={max_group_dim}")
        indices[gi, : group_idx.size] = group_idx
        mask[gi, : group_idx.size] = True
    return indices, mask


def group_scalar_token_ids(
    token_ids: np.ndarray,
    group_indices: Optional[np.ndarray] = None,
    *,
    pad_value: int = 0,
) -> np.ndarray:
    idx = default_group_indices_and_mask()[0] if group_indices is None else np.asarray(group_indices, dtype=np.int64)
    x = np.asarray(token_ids, dtype=np.int64)
    if x.shape[-1] != 277:
        raise ValueError(f"Expected scalar token ids ending in 277 features, got {x.shape}")
    out = np.full((*x.shape[:-1], idx.shape[0], idx.shape[1]), int(pad_value), dtype=np.int64)
    for g in range(idx.shape[0]):
        valid = idx[g] >= 0
        out[..., g, valid] = x[..., idx[g, valid]]
    return out


def grouped_logits_to_expected_values(
    logits: torch.Tensor,
    centers: torch.Tensor,
    group_indices: Optional[torch.Tensor] = None,
    group_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError(f"Expected [B,T,G,D,Bins] logits, got {tuple(logits.shape)}")
    if centers.ndim != 2 or centers.shape[0] != 277:
        raise ValueError(f"Expected [277,Bins] centers, got {tuple(centers.shape)}")

    if group_indices is None or group_valid_mask is None:
        idx_np, mask_np = default_group_indices_and_mask(logits.shape[3])
        group_indices = torch.tensor(idx_np, device=logits.device)
        group_valid_mask = torch.tensor(mask_np, device=logits.device)
    else:
        group_indices = group_indices.to(device=logits.device)
        group_valid_mask = group_valid_mask.to(device=logits.device)

    probs = torch.softmax(logits, dim=-1)
    safe_idx = torch.clamp(group_indices, min=0)
    mask = group_valid_mask.to(dtype=logits.dtype)
    centers_t = centers.to(device=logits.device, dtype=logits.dtype)
    group_centers = centers_t[safe_idx]
    expected_group = (probs * group_centers.unsqueeze(0).unsqueeze(0)).sum(dim=-1) * mask.unsqueeze(0).unsqueeze(0)

    out = logits.new_zeros(logits.shape[0], logits.shape[1], centers.shape[0])
    flat_expected = expected_group.reshape(logits.shape[0], logits.shape[1], -1)
    flat_expected = flat_expected.to(dtype=out.dtype)
    flat_idx = safe_idx.reshape(-1)
    flat_valid = group_valid_mask.reshape(-1)
    out[..., flat_idx[flat_valid]] = flat_expected[..., flat_valid]
    return out


@dataclass
class ModelConfig:
    num_scalar_features: int = 277
    num_tokens_per_frame: int = 46
    num_bins: int = 128
    max_group_dim: int = 10
    window_size: int = 64
    d_model: int = 256
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    num_spatial_latents: int = 32
    num_temporal_latents: int = 16
    code_dim: int = 512
    code_num: int = 512
    vq_mu: float = 0.99
    spatial_blocks: int = 2
    temporal_blocks: int = 2


class JointTokenSpatiotemporalVQVAE(nn.Module):
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
            cfg.num_tokens_per_frame,
            cfg.mlp_ratio,
            cfg.dropout,
        )

        self.out_norm = nn.LayerNorm(d)
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
