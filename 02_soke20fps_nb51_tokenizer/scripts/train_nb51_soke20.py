from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from joint_token_rmsnorm_gqa_vqvae import (
    JointTokenRMSGQAVQVAE,
    ModelConfig,
    PerChannelUniformQuantizer,
    build_default_token_groups,
    count_parameters,
    default_group_indices_and_mask,
    group_scalar_token_ids,
    grouped_logits_to_expected_values,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(preprocess_root: Path, split: str) -> pd.DataFrame:
    p = preprocess_root / f"{split}_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {split} manifest: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise RuntimeError(f"{split} manifest is empty: {p}")
    return df


def fit_or_load_quantizer(
    train_df: pd.DataFrame,
    out_root: Path,
    *,
    num_bins: int,
    max_frames: int,
    rebuild: bool,
    seed: int,
) -> PerChannelUniformQuantizer:
    stats_path = out_root / "quantizer_stats.npz"
    if stats_path.exists() and not rebuild:
        state = np.load(stats_path, allow_pickle=True)
        print("Loaded quantizer:", stats_path)
        return PerChannelUniformQuantizer.from_state_dict({k: state[k] for k in state.files})

    rng = np.random.default_rng(seed)
    per_clip_cap = max(32, int(np.ceil(max_frames / max(1, len(train_df)))))
    chunks: list[np.ndarray] = []
    total = 0
    for clip_idx, row in enumerate(train_df.itertuples(index=False), start=1):
        if clip_idx == 1 or clip_idx % 500 == 0 or clip_idx == len(train_df):
            print(f"  quantizer sampling: {clip_idx}/{len(train_df)} clips | frames={total}")
        arr = np.load(row.cache_npz)["pred"].astype(np.float32)
        if arr.shape[0] > per_clip_cap:
            idx = np.linspace(0, arr.shape[0] - 1, num=per_clip_cap, dtype=np.int64)
            arr = arr[idx]
        chunks.append(arr)
        total += arr.shape[0]
        if total >= max_frames:
            break
    sample = np.concatenate(chunks, axis=0)
    if sample.shape[0] > max_frames:
        idx = rng.choice(sample.shape[0], size=max_frames, replace=False)
        sample = sample[idx]
    print("Quantizer fit sample:", sample.shape)
    q = PerChannelUniformQuantizer(num_bins=num_bins, clip_low_pct=0.5, clip_high_pct=99.5).fit(sample)
    np.savez_compressed(stats_path, **q.state_dict())
    print("Saved quantizer:", stats_path)
    return q


class CanonicalJointTokenDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        quantizer: PerChannelUniformQuantizer,
        group_index_np: np.ndarray,
        *,
        split: str,
        window_size: int,
        seed: int,
        preload: bool,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.quantizer = quantizer
        self.group_index_np = group_index_np
        self.split = str(split)
        self.window_size = int(window_size)
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 10_000))
        self.preload = bool(preload)
        self._cache: list[dict[str, Any]] = []
        if self.preload:
            print(f"Preloading {len(self.df)} clips for {self.split}")
            for idx, row in enumerate(self.df.itertuples(index=False), start=1):
                if idx == 1 or idx % 500 == 0 or idx == len(self.df):
                    print(f"  preload {self.split}: {idx}/{len(self.df)}")
                self._cache.append(self._load_row(row))

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_row(row: Any) -> dict[str, Any]:
        npz = np.load(row.cache_npz)
        return {
            "pred": np.asarray(npz["pred"], dtype=np.float32),
            "copy": np.asarray(npz["copy"], dtype=np.float32),
            "meta": np.asarray(npz["meta"], dtype=np.float32),
            "dataset": row.dataset,
            "split": row.split,
            "clip_id": row.clip_id,
            "cache_npz": row.cache_npz,
        }

    def _get_entry(self, idx: int) -> dict[str, Any]:
        if self.preload:
            return self._cache[idx]
        return self._load_row(self.df.iloc[idx])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._get_entry(idx)
        feat = entry["pred"]
        copy_feat = entry["copy"]
        meta_vec = entry["meta"]
        t = int(feat.shape[0])
        if t < self.window_size:
            raise RuntimeError(f"Clip shorter than window: {entry['clip_id']} shape={feat.shape}")
        max_start = t - self.window_size
        if self.split == "train":
            start = 0 if max_start <= 0 else int(self.rng.integers(0, max_start + 1))
        else:
            start = 0 if max_start <= 0 else int(max_start // 2)
        feat_win = feat[start : start + self.window_size]
        copy_win = copy_feat[start : start + self.window_size]
        scalar_token_ids = self.quantizer.encode(feat_win)
        group_token_ids = group_scalar_token_ids(scalar_token_ids, self.group_index_np, pad_value=0)
        return {
            "group_token_ids": torch.from_numpy(group_token_ids).long(),
            "feat": torch.from_numpy(feat_win).float(),
            "copy": torch.from_numpy(copy_win).float(),
            "meta": torch.from_numpy(meta_vec).float(),
            "dataset": entry["dataset"],
            "split": entry["split"],
            "clip_id": entry["clip_id"],
            "cache_npz": entry["cache_npz"],
        }


def compute_losses(
    batch: dict[str, Any],
    out: dict[str, torch.Tensor],
    *,
    centers_t: torch.Tensor,
    group_index_t: torch.Tensor,
    group_valid_mask_t: torch.Tensor,
    num_bins: int,
    lambda_ce: float,
    lambda_velocity: float,
    lambda_commit: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    group_token_ids = batch["group_token_ids"].to(device, non_blocking=True)
    feat = batch["feat"].to(device, non_blocking=True)

    logits = out["logits"]
    logits_flat = logits.reshape(logits.shape[0], logits.shape[1], -1, num_bins)
    token_flat = group_token_ids.reshape(group_token_ids.shape[0], group_token_ids.shape[1], -1)
    valid_flat = group_valid_mask_t.reshape(-1)
    valid_logits = logits_flat[:, :, valid_flat, :]
    valid_targets = token_flat[:, :, valid_flat]

    ce = F.cross_entropy(valid_logits.reshape(-1, num_bins), valid_targets.reshape(-1))
    feat_hat = grouped_logits_to_expected_values(logits, centers_t, group_index_t, group_valid_mask_t)
    vel_gt = feat[:, 1:] - feat[:, :-1]
    vel_pr = feat_hat[:, 1:] - feat_hat[:, :-1]
    vel = F.smooth_l1_loss(vel_pr, vel_gt)
    commit = out["commit_loss"]
    total = lambda_ce * ce + lambda_velocity * vel + lambda_commit * commit
    return {
        "total": total,
        "ce": ce.detach(),
        "vel": vel.detach(),
        "commit": commit.detach(),
        "perplexity": out["perplexity"].detach(),
    }


def run_epoch(
    model: JointTokenRMSGQAVQVAE,
    loader: DataLoader,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    centers_t: torch.Tensor,
    group_index_t: torch.Tensor,
    group_valid_mask_t: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> dict[str, float]:
    phase = "train" if train else "val"
    model.train(train)
    rows: list[dict[str, float]] = []
    if train and optimizer is None:
        raise ValueError("optimizer is required for train=True")
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader, start=1):
        group_token_ids = batch["group_token_ids"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(group_token_ids)
            losses = compute_losses(
                batch,
                out,
                centers_t=centers_t,
                group_index_t=group_index_t,
                group_valid_mask_t=group_valid_mask_t,
                num_bins=args.num_bins,
                lambda_ce=args.lambda_ce,
                lambda_velocity=args.lambda_velocity,
                lambda_commit=args.lambda_commit,
                device=device,
            )
            step_loss = losses["total"] / max(1, int(args.grad_accum))
        if batch_idx == 1:
            print(
                f"[{phase}] first batch group_token_ids={tuple(group_token_ids.shape)} "
                f"feat={tuple(batch['feat'].shape)} logits={tuple(out['logits'].shape)} "
                f"codes={tuple(out['code_indices'].shape)} loss={float(losses['total'].detach().cpu()):.6f}"
            )
        if train:
            scaler.scale(step_loss).backward()
            should_step = (batch_idx % int(args.grad_accum) == 0) or (batch_idx == len(loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        rows.append(
            {
                "loss": float(losses["total"].detach().cpu()),
                "ce": float(losses["ce"].cpu()),
                "vel": float(losses["vel"].cpu()),
                "commit": float(losses["commit"].cpu()),
                "perplexity": float(losses["perplexity"].cpu()),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "loss": float(df["loss"].mean()),
        "ce": float(df["ce"].mean()),
        "vel": float(df["vel"].mean()),
        "commit": float(df["commit"].mean()),
        "perplexity": float(df["perplexity"].mean()),
    }


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: JointTokenRMSGQAVQVAE,
    model_cfg: ModelConfig,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    quantizer: PerChannelUniformQuantizer,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_cfg": asdict(model_cfg),
            "train_args": vars(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "quantizer_state": quantizer.state_dict(),
            "history": history,
        },
        path,
    )


def maybe_plot(history_csv: Path, plot_path: Path) -> None:
    if not history_csv.exists():
        return
    hist = pd.read_csv(history_csv)
    if hist.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
    axes[0].set_title("Total Loss (lower better)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(hist["epoch"], hist["train_ce"], label="train_ce")
    axes[1].plot(hist["epoch"], hist["val_ce"], label="val_ce")
    axes[1].set_title("Token Cross-Entropy (lower better)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(hist["epoch"], hist["train_velocity"], label="train_velocity")
    axes[2].plot(hist["epoch"], hist["val_velocity"], label="val_velocity")
    axes[2].set_title("Velocity Loss (lower better)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Smooth-L1")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(hist["epoch"], hist["train_perplexity"], label="train_perplexity")
    axes[3].plot(hist["epoch"], hist["val_perplexity"], label="val_perplexity")
    axes[3].set_title("Codebook Perplexity (higher better)")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Perplexity")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Notebook-51 t08_s32_cd1024 tokenizer on SOKE20 caches.")
    parser.add_argument("--preprocess-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preload", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--rebuild-quantizer", action="store_true")
    parser.add_argument("--quantizer-frame-cap", type=int, default=200_000)
    parser.add_argument("--num-bins", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-temporal-latents", type=int, default=8)
    parser.add_argument("--num-spatial-latents", type=int, default=32)
    parser.add_argument("--code-dim", type=int, default=1024)
    parser.add_argument("--code-num", type=int, default=1024)
    parser.add_argument("--spatial-blocks", type=int, default=2)
    parser.add_argument("--temporal-blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-velocity", type=float, default=0.5)
    parser.add_argument("--lambda-commit", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    return parser.parse_args()


RESUME_COMPAT_KEYS = (
    "num_bins",
    "window_size",
    "d_model",
    "num_heads",
    "num_kv_heads",
    "num_temporal_latents",
    "num_spatial_latents",
    "code_dim",
    "code_num",
    "spatial_blocks",
    "temporal_blocks",
)


def resume_mismatches(ckpt_args: dict[str, Any], args: argparse.Namespace) -> list[str]:
    mismatches: list[str] = []
    for key in RESUME_COMPAT_KEYS:
        if key not in ckpt_args:
            continue
        old = ckpt_args[key]
        new = getattr(args, key)
        if old != new:
            mismatches.append(f"{key}: checkpoint={old!r} current={new!r}")
    return mismatches


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "train_args.requested.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

    train_df = load_manifest(args.preprocess_root, "train")
    val_df = load_manifest(args.preprocess_root, "val")
    print("Train clips:", len(train_df), "| Val clips:", len(val_df))

    quantizer = fit_or_load_quantizer(
        train_df,
        args.run_root,
        num_bins=args.num_bins,
        max_frames=args.quantizer_frame_cap,
        rebuild=bool(args.rebuild_quantizer),
        seed=args.seed,
    )

    token_groups = build_default_token_groups()
    group_index_np, group_valid_mask_np = default_group_indices_and_mask(10, token_groups)
    group_index_t = torch.tensor(group_index_np, dtype=torch.long, device=device)
    group_valid_mask_t = torch.tensor(group_valid_mask_np, dtype=torch.bool, device=device)
    centers_t = torch.tensor(quantizer.centers, dtype=torch.float32, device=device)

    train_ds = CanonicalJointTokenDataset(
        train_df,
        quantizer,
        group_index_np,
        split="train",
        window_size=args.window_size,
        seed=args.seed,
        preload=bool(args.preload),
    )
    val_ds = CanonicalJointTokenDataset(
        val_df,
        quantizer,
        group_index_np,
        split="val",
        window_size=args.window_size,
        seed=args.seed,
        preload=bool(args.preload),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    print("Train batches:", len(train_loader), "| Val batches:", len(val_loader))

    model_cfg = ModelConfig(
        num_scalar_features=277,
        num_tokens_per_frame=46,
        num_bins=args.num_bins,
        max_group_dim=10,
        window_size=args.window_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        num_spatial_latents=args.num_spatial_latents,
        num_temporal_latents=args.num_temporal_latents,
        code_dim=args.code_dim,
        code_num=args.code_num,
        vq_mu=0.99,
        spatial_blocks=args.spatial_blocks,
        temporal_blocks=args.temporal_blocks,
    )
    model = JointTokenRMSGQAVQVAE(model_cfg).to(device)
    print("Model config:", model_cfg)
    print("Parameter count:", f"{count_parameters(model):,}")
    print("Serialized codes / 64 frames:", args.num_temporal_latents * args.num_spatial_latents)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.epochs)),
        eta_min=args.scheduler_eta_min,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_ckpt = args.run_root / "best.pt"
    last_ckpt = args.run_root / "last.pt"
    history_csv = args.run_root / "history.csv"
    summary_csv = args.run_root / "results_summary.csv"
    start_epoch = 1
    best_val = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    history: list[dict[str, Any]] = []

    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        mismatches = resume_mismatches(ckpt_args, args) if isinstance(ckpt_args, dict) else []
        if mismatches:
            mismatch_text = "\n  - ".join(mismatches)
            raise RuntimeError(
                "Refusing to resume incompatible checkpoint.\n"
                f"Checkpoint: {last_ckpt}\n"
                "Mismatched resume settings:\n"
                f"  - {mismatch_text}\n"
                "Use --resume 0 to restart this run root, or choose a new --run-root."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt["epoch"]) + 1
        if history:
            hist_df = pd.DataFrame(history)
            best_idx = hist_df["val_loss"].astype(float).idxmin()
            best_val = float(hist_df.loc[best_idx, "val_loss"])
            best_epoch = int(hist_df.loc[best_idx, "epoch"])
            epochs_since_improve = int(hist_df.iloc[-1].get("epochs_since_improve", 0))
        print("Resumed from:", last_ckpt, "next_epoch=", start_epoch)

    (args.run_root / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        print(f"=== EPOCH {epoch}/{args.epochs} ===")
        t0 = time.time()
        tr = run_epoch(
            model,
            train_loader,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            centers_t=centers_t,
            group_index_t=group_index_t,
            group_valid_mask_t=group_valid_mask_t,
            device=device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        with torch.no_grad():
            va = run_epoch(
                model,
                val_loader,
                train=False,
                optimizer=None,
                scaler=scaler,
                args=args,
                centers_t=centers_t,
                group_index_t=group_index_t,
                group_valid_mask_t=group_valid_mask_t,
                device=device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
        scheduler.step()
        elapsed = time.time() - t0

        improved = va["loss"] < best_val
        if improved:
            best_val = float(va["loss"])
            best_epoch = int(epoch)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        row = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_velocity": tr["vel"],
            "train_commit": tr["commit"],
            "train_perplexity": tr["perplexity"],
            "val_loss": va["loss"],
            "val_ce": va["ce"],
            "val_velocity": va["vel"],
            "val_commit": va["commit"],
            "val_perplexity": va["perplexity"],
            "epoch_sec": float(elapsed),
            "improved": bool(improved),
            "epochs_since_improve": int(epochs_since_improve),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_csv, index=False)
        print(row)

        save_checkpoint(
            last_ckpt,
            epoch=epoch,
            model=model,
            model_cfg=model_cfg,
            args=args,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            quantizer=quantizer,
            history=history,
        )
        if improved:
            save_checkpoint(
                best_ckpt,
                epoch=epoch,
                model=model,
                model_cfg=model_cfg,
                args=args,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                quantizer=quantizer,
                history=history,
            )

        pd.DataFrame(
            [
                {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "stop_epoch": epoch,
                    "stop_val_loss": va["loss"],
                    "early_stopped": epochs_since_improve >= int(args.early_stop_patience),
                    "history_csv": str(history_csv),
                    "best_ckpt": str(best_ckpt),
                }
            ]
        ).to_csv(summary_csv, index=False)
        maybe_plot(history_csv, args.run_root / "plots" / "training_curves.png")

        if epochs_since_improve >= int(args.early_stop_patience):
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} val={best_val:.6f}")
            break

    print("Training done.")
    print("history:", history_csv)
    print("summary:", summary_csv)
    print("best:", best_ckpt)


if __name__ == "__main__":
    main()
