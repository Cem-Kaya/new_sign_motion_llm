from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizeEMAReset(nn.Module):
    """Standalone EMA VQ codebook with dead-code reset.

    This is a local drop-in replacement for the SOKE/MotionGPT-style
    QuantizeEMAReset used by earlier notebooks. The public methods and state
    names intentionally match the old class so existing state_dict checkpoints
    keep loading.
    """

    def __init__(self, nb_code: int, code_dim: int, mu: float) -> None:
        super().__init__()
        self.nb_code = int(nb_code)
        self.code_dim = int(code_dim)
        self.mu = float(mu)
        self.init = False
        self.code_sum: torch.Tensor | None = None
        self.code_count: torch.Tensor | None = None
        self.register_buffer("codebook", torch.zeros(self.nb_code, self.code_dim))

    def reset_codebook(self) -> None:
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.codebook.zero_()

    def _tile(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] >= self.nb_code:
            return x
        repeats = math.ceil(self.nb_code / max(int(x.shape[0]), 1))
        noise_std = 0.01 / math.sqrt(float(self.code_dim))
        out = x.repeat(repeats, 1)
        return out + torch.randn_like(out) * noise_std

    @torch.no_grad()
    def init_codebook(self, x: torch.Tensor) -> None:
        tiled = self._tile(x)
        self.codebook.copy_(tiled[: self.nb_code])
        self.code_sum = self.codebook.detach().clone()
        self.code_count = torch.ones(self.nb_code, device=x.device, dtype=x.dtype)
        self.init = True

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, T] -> [N*T, C]
        return x.permute(0, 2, 1).contiguous().view(-1, x.shape[1])

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        codebook = self.codebook.to(device=x.device, dtype=x.dtype)
        distances = (
            x.square().sum(dim=1, keepdim=True)
            - 2.0 * x @ codebook.t()
            + codebook.square().sum(dim=1).unsqueeze(0)
        )
        return distances.argmin(dim=1)

    def dequantize(self, code_idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(code_idx, self.codebook)

    @torch.no_grad()
    def compute_perplexity(self, code_idx: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(code_idx.view(-1), minlength=self.nb_code).to(dtype=torch.float32)
        probs = counts / counts.sum().clamp_min(1.0)
        return torch.exp(-(probs * torch.log(probs + 1e-7)).sum())

    @torch.no_grad()
    def update_codebook(self, x: torch.Tensor, code_idx: torch.Tensor) -> torch.Tensor:
        if self.code_sum is None or self.code_count is None:
            self.init_codebook(x)

        onehot = F.one_hot(code_idx.view(-1), num_classes=self.nb_code).to(dtype=x.dtype, device=x.device)
        code_sum = onehot.t() @ x
        code_count = onehot.sum(dim=0)

        self.code_sum = self.mu * self.code_sum.to(device=x.device, dtype=x.dtype) + (1.0 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count.to(device=x.device, dtype=x.dtype) + (1.0 - self.mu) * code_count

        replacement = self._tile(x)[: self.nb_code]
        usage = (self.code_count >= 1.0).to(dtype=x.dtype).unsqueeze(1)
        updated = self.code_sum / self.code_count.clamp_min(1e-7).unsqueeze(1)
        self.codebook.copy_((usage * updated + (1.0 - usage) * replacement).to(dtype=self.codebook.dtype))

        probs = code_count / code_count.sum().clamp_min(1.0)
        return torch.exp(-(probs * torch.log(probs + 1e-7)).sum())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [N,C,T], got {tuple(x.shape)}")
        n, _, t = x.shape
        flat = self.preprocess(x)

        if self.training and not self.init:
            self.init_codebook(flat)

        code_idx = self.quantize(flat)
        quantized = self.dequantize(code_idx).to(dtype=flat.dtype)

        if self.training:
            perplexity = self.update_codebook(flat, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx).to(device=x.device)

        commit_loss = F.mse_loss(flat, quantized.detach())
        quantized = flat + (quantized - flat).detach()
        quantized = quantized.view(n, t, -1).permute(0, 2, 1).contiguous()
        return quantized, commit_loss, perplexity


__all__ = ["QuantizeEMAReset"]
