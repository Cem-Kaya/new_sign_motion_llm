from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from joint_token_rmsnorm_gqa_vqvae import GQATransformerBlock, RMSNorm
from joint_token_spatiotemporal_vqvae import build_default_token_groups, default_group_indices_and_mask


def build_default_token_spatial_coords(groups: Optional[Sequence[Dict[str, Any]]] = None) -> np.ndarray:
    groups = list(groups) if groups is not None else build_default_token_groups()
    coord_map = {
        "pelvis": (5, 3),
        "spine1": (4, 3),
        "spine2": (3, 3),
        "spine3": (2, 3),
        "neck": (1, 3),
        "left_collar": (1, 2),
        "right_collar": (1, 4),
        "head": (0, 3),
        "left_shoulder": (2, 1),
        "right_shoulder": (2, 5),
        "left_elbow": (3, 1),
        "right_elbow": (3, 5),
        "left_wrist": (4, 1),
        "right_wrist": (4, 5),
        "jaw": (0, 3),
        "expression": (0, 4),
    }
    out = np.zeros((len(groups), 2), dtype=np.int64)
    for gi, group in enumerate(groups):
        name = str(group["group_name"])
        if name in coord_map:
            out[gi] = np.asarray(coord_map[name], dtype=np.int64)
        elif name.startswith("left_hand_joint_"):
            idx = int(name.rsplit("_", 1)[-1])
            out[gi] = np.asarray((5 + idx % 3, idx // 3), dtype=np.int64)
        elif name.startswith("right_hand_joint_"):
            idx = int(name.rsplit("_", 1)[-1])
            out[gi] = np.asarray((5 + idx % 3, 6 + idx // 3), dtype=np.int64)
        else:
            raise KeyError(f"Missing 2D coordinate for token {name}")
    return out


def group_scalar_features_277(
    features: np.ndarray,
    group_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    idx = default_group_indices_and_mask()[0] if group_indices is None else np.asarray(group_indices, dtype=np.int64)
    x = np.asarray(features, dtype=np.float32)
    if x.shape[-1] != 277:
        raise ValueError(f"Expected scalar features ending in 277 dims, got {x.shape}")
    out = np.zeros((*x.shape[:-1], idx.shape[0], idx.shape[1]), dtype=np.float32)
    for g in range(idx.shape[0]):
        valid = idx[g] >= 0
        out[..., g, valid] = x[..., idx[g, valid]]
    return out


def grouped_continuous_to_scalar_features_277(
    group_values: torch.Tensor,
    group_indices: Optional[torch.Tensor] = None,
    group_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if group_values.ndim != 4:
        raise ValueError(f"Expected [B,T,G,D] grouped values, got {tuple(group_values.shape)}")
    if group_indices is None or group_valid_mask is None:
        idx_np, mask_np = default_group_indices_and_mask(group_values.shape[3])
        group_indices = torch.tensor(idx_np, device=group_values.device)
        group_valid_mask = torch.tensor(mask_np, device=group_values.device)
    else:
        group_indices = group_indices.to(device=group_values.device)
        group_valid_mask = group_valid_mask.to(device=group_values.device)

    safe_idx = torch.clamp(group_indices, min=0)
    mask = group_valid_mask.to(dtype=group_values.dtype)
    grouped = group_values * mask.unsqueeze(0).unsqueeze(0)
    out = group_values.new_zeros(group_values.shape[0], group_values.shape[1], 277)
    flat_grouped = grouped.reshape(group_values.shape[0], group_values.shape[1], -1)
    flat_idx = safe_idx.reshape(-1)
    flat_valid = group_valid_mask.reshape(-1)
    out[..., flat_idx[flat_valid]] = flat_grouped[..., flat_valid]
    return out


@dataclass
class ModelConfig:
    num_scalar_features: int = 277
    num_tokens_per_frame: int = 46
    max_group_dim: int = 10
    window_size: int = 64
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    spatial_in_blocks: int = 2
    temporal_blocks: int = 2
    spatial_out_blocks: int = 2
    num_spatial_rows: int = 8
    num_spatial_cols: int = 11


class JointTokenSOKE20ContinuousRefiner(nn.Module):
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
        token_coords_np = build_default_token_spatial_coords(groups)
        self.group_names = [g["group_name"] for g in groups]
        self.register_buffer("group_indices", torch.tensor(group_indices_np, dtype=torch.long), persistent=False)
        self.register_buffer("group_valid_mask", torch.tensor(group_valid_mask_np, dtype=torch.bool), persistent=False)
        self.register_buffer("token_rows", torch.tensor(token_coords_np[:, 0], dtype=torch.long), persistent=False)
        self.register_buffer("token_cols", torch.tensor(token_coords_np[:, 1], dtype=torch.long), persistent=False)

        self.scalar_proj = nn.Linear(1, d)
        self.scalar_input_scale = nn.Parameter(torch.ones(cfg.max_group_dim, d))
        self.within_group_pos_embed = nn.Parameter(torch.randn(cfg.max_group_dim, d) * 0.02)
        self.token_row_embed = nn.Parameter(torch.randn(cfg.num_spatial_rows, d) * 0.02)
        self.token_col_embed = nn.Parameter(torch.randn(cfg.num_spatial_cols, d) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(cfg.window_size, d) * 0.02)
        self.group_in_proj = nn.Linear(cfg.max_group_dim * d, d, bias=False)
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.spatial_in = nn.ModuleList(
            [GQATransformerBlock(d, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.spatial_in_blocks)]
        )
        self.temporal = nn.ModuleList(
            [GQATransformerBlock(d, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.temporal_blocks)]
        )
        self.spatial_out = nn.ModuleList(
            [GQATransformerBlock(d, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.spatial_out_blocks)]
        )
        self.out_norm = RMSNorm(d)
        self.group_out = nn.Linear(d, cfg.max_group_dim)

    def _spatial_token_bias(self) -> torch.Tensor:
        return self.token_row_embed[self.token_rows] + self.token_col_embed[self.token_cols]

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x, _ = block(x, return_attn=False)
        return x

    def _embed_group_features(self, group_values: torch.Tensor) -> torch.Tensor:
        x = self.scalar_proj(group_values.unsqueeze(-1))
        x = x * self.scalar_input_scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        x = x + self.within_group_pos_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = self.group_valid_mask.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        x = x * mask
        x = x.reshape(
            group_values.shape[0],
            group_values.shape[1],
            group_values.shape[2],
            self.cfg.max_group_dim * self.cfg.d_model,
        )
        x = self.group_in_proj(x)
        x = x + self._spatial_token_bias().unsqueeze(0).unsqueeze(0)
        x = x + self.time_embed.unsqueeze(0).unsqueeze(2)
        return self.embed_drop(x)

    def forward(self, group_values: torch.Tensor, scalar_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        if group_values.ndim != 4:
            raise ValueError(f"Expected [B,T,G,D] grouped continuous values, got {tuple(group_values.shape)}")
        if scalar_input.ndim != 3:
            raise ValueError(f"Expected [B,T,F] scalar input, got {tuple(scalar_input.shape)}")

        bsz, time_len, num_tokens, group_dim = group_values.shape
        if time_len != self.cfg.window_size or num_tokens != self.cfg.num_tokens_per_frame or group_dim != self.cfg.max_group_dim:
            raise ValueError(
                "Expected "
                f"[B,{self.cfg.window_size},{self.cfg.num_tokens_per_frame},{self.cfg.max_group_dim}] grouped values, "
                f"got {tuple(group_values.shape)}"
            )
        if scalar_input.shape[-1] != self.cfg.num_scalar_features:
            raise ValueError(f"Expected {self.cfg.num_scalar_features} scalar features, got {tuple(scalar_input.shape)}")

        x = self._embed_group_features(group_values)
        x0 = x

        x_spatial = x.reshape(bsz * time_len, num_tokens, self.cfg.d_model)
        x_spatial = self._run_blocks(self.spatial_in, x_spatial)
        x = x_spatial.reshape(bsz, time_len, num_tokens, self.cfg.d_model)

        x_temporal = x.permute(0, 2, 1, 3).reshape(bsz * num_tokens, time_len, self.cfg.d_model)
        x_temporal = self._run_blocks(self.temporal, x_temporal)
        x = x_temporal.reshape(bsz, num_tokens, time_len, self.cfg.d_model).permute(0, 2, 1, 3)
        x = x + x0

        x_spatial_out = x.reshape(bsz * time_len, num_tokens, self.cfg.d_model)
        x_spatial_out = self._run_blocks(self.spatial_out, x_spatial_out)
        x = x_spatial_out.reshape(bsz, time_len, num_tokens, self.cfg.d_model)

        delta_group = self.group_out(self.out_norm(x))
        mask = self.group_valid_mask.to(device=delta_group.device, dtype=delta_group.dtype).unsqueeze(0).unsqueeze(0)
        delta_group = delta_group * mask
        delta_scalar = grouped_continuous_to_scalar_features_277(delta_group, self.group_indices, self.group_valid_mask)
        refined_feat = scalar_input + delta_scalar
        return {"delta_group": delta_group, "delta_scalar": delta_scalar, "refined_feat": refined_feat}


__all__ = [
    "ModelConfig",
    "JointTokenSOKE20ContinuousRefiner",
    "group_scalar_features_277",
    "grouped_continuous_to_scalar_features_277",
]
