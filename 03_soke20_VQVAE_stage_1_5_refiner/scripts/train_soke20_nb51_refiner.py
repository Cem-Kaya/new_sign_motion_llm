from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parents[2]
NEW_DATA_ROOT = SCRIPT_DIR.parents[1]
STAGE1_SCRIPT_DIR = NEW_DATA_ROOT / "02_soke20fps_nb51_tokenizer" / "scripts"
for p in (SCRIPT_DIR, STAGE1_SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_nb51_soke20_metrics import (  # noqa: E402
    NB51Runner,
    SMPLXJ22Decoder,
    dtw_last_cell,
    load_npz_entry,
    mpjpe_cost_matrix_blocked_torch,
    pa_cost_matrix_blocked_torch,
    reconstruct_full169,
)
from joint_token_spatiotemporal_vqvae import default_group_indices_and_mask  # noqa: E402
from joint_token_soke20_continuous_refiner import (  # noqa: E402
    JointTokenSOKE20ContinuousRefiner,
    ModelConfig as RefinerModelConfig,
    group_scalar_features_277,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_stem(*parts: object, limit: int = 180) -> str:
    raw = "__".join(str(p) for p in parts)
    out = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw).strip("_")
    return (out[:limit] or "clip")


def short_hash(text: object) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]


def load_manifest(preprocess_root: Path, split: str, max_clips: int, seed: int) -> pd.DataFrame:
    path = preprocess_root / f"{split}_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"].copy()
    if max_clips > 0 and len(df) > max_clips:
        chunks = []
        per_dataset = max(1, max_clips // max(1, df["dataset"].nunique()))
        for _, group in df.groupby("dataset", sort=True):
            n = min(len(group), per_dataset)
            chunks.append(group.sample(n=n, random_state=seed))
        df = pd.concat(chunks, ignore_index=True)
        if len(df) < max_clips:
            remaining = max_clips - len(df)
            rest = pd.read_csv(path)
            if "status" in rest.columns:
                rest = rest[rest["status"].astype(str) == "ok"].copy()
            rest = rest[~rest["cache_npz"].astype(str).isin(set(df["cache_npz"].astype(str)))]
            if remaining > 0 and len(rest) > 0:
                df = pd.concat([df, rest.sample(n=min(remaining, len(rest)), random_state=seed + 1)], ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No usable rows in {path}")
    return df


def load_existing_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty and "recon_npz" in df.columns:
            return df
    return pd.DataFrame()


def recon_cache_path(recon_cache_root: Path, split: str, row: Any) -> Path:
    dataset = safe_stem(getattr(row, "dataset"))
    clip_id = safe_stem(getattr(row, "clip_id"))
    h = short_hash(getattr(row, "cache_npz"))
    return recon_cache_root / split / dataset / f"{clip_id}__{h}.npz"


def pad_window(x: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] >= window_size:
        return x
    if x.shape[0] == 0:
        raise ValueError("Cannot pad an empty sequence")
    return np.concatenate([x, np.repeat(x[-1:, :], window_size - x.shape[0], axis=0)], axis=0).astype(np.float32)


def sliding_starts(num_frames: int, window_size: int, stride_size: int) -> list[int]:
    if num_frames <= window_size:
        return [0]
    starts = list(range(0, num_frames - window_size + 1, stride_size))
    final_start = num_frames - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.no_grad()
def build_recon_cache(
    df: pd.DataFrame,
    *,
    split: str,
    runner: NB51Runner,
    recon_cache_root: Path,
    stride_size: int,
    force: bool,
    save_every: int,
) -> pd.DataFrame:
    out_manifest = recon_cache_root / f"{split}_recon_manifest.csv"
    existing = load_existing_manifest(out_manifest)
    done = set(existing["source_cache_npz"].astype(str)) if not existing.empty and not force else set()
    rows = [] if force else existing.to_dict("records")
    started = time.time()
    recon_cache_root.joinpath(split).mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        source_cache = str(getattr(row, "cache_npz"))
        out_npz = recon_cache_path(recon_cache_root, split, row)
        if source_cache in done and out_npz.exists():
            continue
        feat, copy_feat, meta_vec = load_npz_entry(source_cache)
        base_feat = runner.predict(feat, stride_size=stride_size)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            gt_feat=feat.astype(np.float32),
            base_feat=base_feat.astype(np.float32),
            copy=copy_feat.astype(np.float32),
            meta=meta_vec.astype(np.float32),
        )
        rows.append(
            {
                "dataset": str(getattr(row, "dataset")),
                "source_alias": str(getattr(row, "source_alias", "")),
                "split": str(getattr(row, "split")),
                "clip_id": str(getattr(row, "clip_id")),
                "source_cache_npz": source_cache,
                "recon_npz": str(out_npz),
                "num_frames": int(feat.shape[0]),
                "target_duration_sec": float(getattr(row, "target_duration_sec", feat.shape[0] / 20.0)),
            }
        )
        done.add(source_cache)
        if idx == 1 or idx % save_every == 0 or idx == len(df):
            pd.DataFrame(rows).to_csv(out_manifest, index=False)
            elapsed = (time.time() - started) / 60.0
            print(f"recon cache {split}: {idx}/{len(df)} | written={len(rows)} | elapsed={elapsed:.1f}m", flush=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_manifest, index=False)
    print(f"Wrote {split} recon manifest: {out_manifest} rows={len(out_df)}", flush=True)
    return out_df


def fit_or_load_stats(train_recon_df: pd.DataFrame, run_root: Path, *, cap_frames: int, force: bool) -> tuple[np.ndarray, np.ndarray]:
    stats_path = run_root / "refiner_feature_stats.npz"
    if stats_path.exists() and not force:
        z = np.load(stats_path)
        return z["mean"].astype(np.float32), z["std"].astype(np.float32)
    chunks = []
    total = 0
    for row in train_recon_df.itertuples(index=False):
        with np.load(row.recon_npz) as z:
            gt_feat = np.asarray(z["gt_feat"], dtype=np.float32)
        chunks.append(gt_feat)
        total += int(gt_feat.shape[0])
        if total >= cap_frames:
            break
    flat = np.concatenate(chunks, axis=0)
    if flat.shape[0] > cap_frames:
        flat = flat[:cap_frames]
    mean = flat.mean(axis=0).astype(np.float32)
    std = np.clip(flat.std(axis=0).astype(np.float32), 1e-4, None)
    np.savez_compressed(stats_path, mean=mean, std=std)
    print(f"Wrote refiner stats: {stats_path} frames={flat.shape[0]}", flush=True)
    return mean, std


class RefinerWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        split: str,
        window_size: int,
        windows_per_clip: int,
        seed: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.split = str(split)
        self.window_size = int(window_size)
        self.windows_per_clip = int(windows_per_clip)
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 10_000))
        self.group_idx_np = default_group_indices_and_mask()[0]

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.df) * max(1, self.windows_per_clip)
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx % len(self.df)]
        with np.load(row.recon_npz) as z:
            base_feat = np.asarray(z["base_feat"], dtype=np.float32)
            gt_feat = np.asarray(z["gt_feat"], dtype=np.float32)
        max_start = max(0, int(base_feat.shape[0]) - self.window_size)
        if self.split == "train":
            start = 0 if max_start == 0 else int(self.rng.integers(0, max_start + 1))
        else:
            start = max_start // 2
        end = min(start + self.window_size, int(base_feat.shape[0]))
        base = pad_window(base_feat[start:end], self.window_size)
        gt = pad_window(gt_feat[start:end], self.window_size)
        base_norm = (base - self.mean) / self.std
        gt_norm = (gt - self.mean) / self.std
        group_norm = group_scalar_features_277(base_norm, self.group_idx_np)
        return {
            "base_norm": torch.from_numpy(base_norm).float(),
            "gt_norm": torch.from_numpy(gt_norm).float(),
            "group_norm": torch.from_numpy(group_norm).float(),
        }


def compute_losses(gt_norm: torch.Tensor, pred_norm: torch.Tensor) -> dict[str, torch.Tensor]:
    recon = F.smooth_l1_loss(pred_norm, gt_norm)
    vel_gt = gt_norm[:, 1:] - gt_norm[:, :-1]
    vel_pr = pred_norm[:, 1:] - pred_norm[:, :-1]
    velocity = F.smooth_l1_loss(vel_pr, vel_gt)
    return {"recon": recon, "velocity": velocity, "total": recon + 0.5 * velocity}


def make_scaler(amp_enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def run_epoch(
    model: JointTokenSOKE20ContinuousRefiner,
    loader: DataLoader,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
) -> dict[str, float]:
    model.train(train)
    rows = []
    for batch_idx, batch in enumerate(loader, start=1):
        group_norm = batch["group_norm"].to(device, non_blocking=True)
        base_norm = batch["base_norm"].to(device, non_blocking=True)
        gt_norm = batch["gt_norm"].to(device, non_blocking=True)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(group_norm, base_norm)
            losses = compute_losses(gt_norm, out["refined_feat"])
            base_losses = compute_losses(gt_norm, base_norm)
        if train:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        if batch_idx == 1:
            prefix = "train" if train else "val"
            print(
                f"{prefix} first batch shapes: group={tuple(group_norm.shape)} base={tuple(base_norm.shape)} "
                f"gt={tuple(gt_norm.shape)} pred={tuple(out['refined_feat'].shape)} loss={float(losses['total'].detach().cpu()):.6f}",
                flush=True,
            )
        rows.append(
            {
                "loss": float(losses["total"].detach().cpu()),
                "recon": float(losses["recon"].detach().cpu()),
                "velocity": float(losses["velocity"].detach().cpu()),
                "base_loss": float(base_losses["total"].detach().cpu()),
                "base_recon": float(base_losses["recon"].detach().cpu()),
                "base_velocity": float(base_losses["velocity"].detach().cpu()),
            }
        )
    df = pd.DataFrame(rows)
    return {k: float(df[k].mean()) for k in df.columns}


def save_training_plot(history: pd.DataFrame, path: Path) -> None:
    if history.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss (lower better)")
    axes[0].plot(history["epoch"], history["val_loss"], label="val loss (lower better)")
    axes[0].plot(history["epoch"], history["val_base_loss"], label="val stage1 base loss (lower better)", linestyle="--")
    axes[0].set_title("SOKE20 NB51 Stage2 Refiner Loss (lower better)")
    axes[0].set_ylabel("Smooth-L1 + velocity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(history["epoch"], history["lr"], color="tab:green", label="learning rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_ylabel("LR")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_refiner(run_root: Path, device: torch.device) -> tuple[JointTokenSOKE20ContinuousRefiner, np.ndarray, np.ndarray]:
    payload = torch.load(run_root / "best.pt", map_location="cpu", weights_only=False)
    cfg = RefinerModelConfig(**payload["model_cfg"])
    model = JointTokenSOKE20ContinuousRefiner(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, np.asarray(payload["feature_mean"], dtype=np.float32), np.asarray(payload["feature_std"], dtype=np.float32)


@torch.no_grad()
def refine_full_clip_windowed(
    model: JointTokenSOKE20ContinuousRefiner,
    base_feat: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    window_size: int,
    stride_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    base_feat = np.asarray(base_feat, dtype=np.float32)
    acc = np.zeros_like(base_feat, dtype=np.float32)
    cnt = np.zeros((base_feat.shape[0], 1), dtype=np.float32)
    group_idx_np = default_group_indices_and_mask()[0]
    for start in sliding_starts(base_feat.shape[0], window_size, stride_size):
        end = min(start + window_size, base_feat.shape[0])
        win = pad_window(base_feat[start:end], window_size)
        win_norm = (win - mean) / std
        group_norm = group_scalar_features_277(win_norm, group_idx_np)
        xb = torch.from_numpy(win_norm[None]).float().to(device)
        xg = torch.from_numpy(group_norm[None]).float().to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            out = model(xg, xb)
        refined = out["refined_feat"][0].detach().float().cpu().numpy() * std + mean
        keep = end - start
        acc[start:end] += refined[:keep]
        cnt[start:end] += 1.0
    return (acc / np.clip(cnt, 1e-6, None)).astype(np.float32)


def train_refiner(args: argparse.Namespace, train_recon_df: pd.DataFrame, val_recon_df: pd.DataFrame, device: torch.device) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    mean, std = fit_or_load_stats(train_recon_df, args.run_root, cap_frames=args.stats_frame_cap, force=args.rebuild_stats)
    train_ds = RefinerWindowDataset(
        train_recon_df,
        mean,
        std,
        split="train",
        window_size=args.window_size,
        windows_per_clip=args.windows_per_clip,
        seed=args.seed,
    )
    val_ds = RefinerWindowDataset(
        val_recon_df,
        mean,
        std,
        split="val",
        window_size=args.window_size,
        windows_per_clip=1,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    cfg = RefinerModelConfig(
        window_size=args.window_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        spatial_in_blocks=args.spatial_in_blocks,
        temporal_blocks=args.temporal_blocks,
        spatial_out_blocks=args.spatial_out_blocks,
    )
    model = JointTokenSOKE20ContinuousRefiner(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.scheduler_eta_min)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = make_scaler(amp_enabled)

    start_epoch = 1
    best_val = float("inf")
    best_epoch = -1
    history_rows: list[dict[str, Any]] = []
    last_path = args.run_root / "last.pt"
    best_path = args.run_root / "best.pt"
    hist_path = args.run_root / "history.csv"
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        best_val = float(payload.get("best_val_loss", best_val))
        best_epoch = int(payload.get("best_epoch", best_epoch))
        start_epoch = int(payload.get("epoch", 0)) + 1
        if hist_path.exists():
            history_rows = pd.read_csv(hist_path).to_dict("records")
        print(f"Resumed refiner from epoch {start_epoch - 1}", flush=True)

    patience_count = 0
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        print(f"=== REFINER EPOCH {epoch}/{args.epochs} ===", flush=True)
        train_stats = run_epoch(
            model,
            train_loader,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model,
                val_loader,
                train=False,
                optimizer=None,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                grad_clip=args.grad_clip,
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_stats["loss"],
            "train_recon": train_stats["recon"],
            "train_velocity": train_stats["velocity"],
            "train_base_loss": train_stats["base_loss"],
            "val_loss": val_stats["loss"],
            "val_recon": val_stats["recon"],
            "val_velocity": val_stats["velocity"],
            "val_base_loss": val_stats["base_loss"],
            "val_improvement_vs_base": val_stats["base_loss"] - val_stats["loss"],
            "epoch_sec": float(time.time() - started),
        }
        improved = row["val_loss"] < best_val
        row["improved"] = bool(improved)
        if improved:
            best_val = row["val_loss"]
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_cfg": asdict(cfg),
                    "model_state_dict": model.state_dict(),
                    "feature_mean": mean.astype(np.float32),
                    "feature_std": std.astype(np.float32),
                    "train_cfg": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                best_path,
            )
        else:
            patience_count += 1
        row["epochs_since_improve"] = int(patience_count)
        torch.save(
            {
                "epoch": epoch,
                "model_cfg": asdict(cfg),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "feature_mean": mean.astype(np.float32),
                "feature_std": std.astype(np.float32),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "train_cfg": vars(args),
            },
            last_path,
        )
        history_rows.append(row)
        hist_df = pd.DataFrame(history_rows)
        hist_df.to_csv(hist_path, index=False)
        save_training_plot(hist_df, args.run_root / "plots" / "refiner_training_curves.png")
        print(row, flush=True)
        if patience_count >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    hist_df = pd.DataFrame(history_rows)
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "stop_epoch": int(hist_df["epoch"].iloc[-1]) if len(hist_df) else 0,
        "stop_val_loss": float(hist_df["val_loss"].iloc[-1]) if len(hist_df) else float("nan"),
        "best_ckpt": str(best_path),
        "last_ckpt": str(last_path),
        "history_csv": str(hist_path),
    }
    (args.run_root / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(args.run_root / "results_summary.csv", index=False)


def evaluate_motion_metrics(args: argparse.Namespace, val_recon_df: pd.DataFrame, device: torch.device) -> None:
    if not (args.run_root / "best.pt").exists():
        print("Skipping motion metrics: no refiner best.pt yet", flush=True)
        return
    metric_df = val_recon_df
    if args.metric_max_clips > 0 and len(metric_df) > args.metric_max_clips:
        metric_df = metric_df.groupby("dataset", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), max(1, args.metric_max_clips // max(1, metric_df["dataset"].nunique()))), random_state=args.seed)
        )
        metric_df = metric_df.reset_index(drop=True)
    model, mean, std = load_refiner(args.run_root, device)
    decoder = SMPLXJ22Decoder(args.body_model_root, device)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    rows = []
    for idx, row in enumerate(metric_df.itertuples(index=False), start=1):
        with np.load(row.recon_npz) as z:
            gt_feat = np.asarray(z["gt_feat"], dtype=np.float32)
            base_feat = np.asarray(z["base_feat"], dtype=np.float32)
            copy_feat = np.asarray(z["copy"], dtype=np.float32)
            meta_vec = np.asarray(z["meta"], dtype=np.float32)
        refined_feat = refine_full_clip_windowed(
            model,
            base_feat,
            mean,
            std,
            window_size=args.window_size,
            stride_size=args.stride_size,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        gt169 = reconstruct_full169(gt_feat, copy_feat, meta_vec)
        base169 = reconstruct_full169(base_feat, copy_feat, meta_vec)
        ref169 = reconstruct_full169(refined_feat, copy_feat, meta_vec)
        gt_j = decoder(gt169, frame_batch=args.metric_frame_batch)
        base_j = decoder(base169, frame_batch=args.metric_frame_batch)
        ref_j = decoder(ref169, frame_batch=args.metric_frame_batch)
        base_raw_costs = mpjpe_cost_matrix_blocked_torch(base_j, gt_j, device=device, block_rows=args.metric_raw_block_rows)
        ref_raw_costs = mpjpe_cost_matrix_blocked_torch(ref_j, gt_j, device=device, block_rows=args.metric_raw_block_rows)
        base_pa_costs = pa_cost_matrix_blocked_torch(base_j, gt_j, device=device, block_rows=args.metric_pa_block_rows)
        ref_pa_costs = pa_cost_matrix_blocked_torch(ref_j, gt_j, device=device, block_rows=args.metric_pa_block_rows)
        rec = {
            "dataset": row.dataset,
            "split": row.split,
            "clip_id": row.clip_id,
            "num_frames": int(gt_feat.shape[0]),
            "stage1_mpjpe_mm_lower_better": float(np.mean(np.linalg.norm(base_j - gt_j, axis=-1)) * 1000.0),
            "stage1_pa_mpjpe_mm_lower_better": float(np.diag(base_pa_costs).mean() * 1000.0),
            "stage1_dtw_mpjpe_mm_lower_better": float(dtw_last_cell(base_raw_costs.astype(np.float64)) * 1000.0),
            "stage1_dtw_pa_mpjpe_mm_lower_better": float(dtw_last_cell(base_pa_costs.astype(np.float64)) * 1000.0),
            "stage2_mpjpe_mm_lower_better": float(np.mean(np.linalg.norm(ref_j - gt_j, axis=-1)) * 1000.0),
            "stage2_pa_mpjpe_mm_lower_better": float(np.diag(ref_pa_costs).mean() * 1000.0),
            "stage2_dtw_mpjpe_mm_lower_better": float(dtw_last_cell(ref_raw_costs.astype(np.float64)) * 1000.0),
            "stage2_dtw_pa_mpjpe_mm_lower_better": float(dtw_last_cell(ref_pa_costs.astype(np.float64)) * 1000.0),
        }
        rows.append(rec)
        if idx == 1 or idx % 25 == 0 or idx == len(metric_df):
            print(
                f"motion {idx}/{len(metric_df)} {row.dataset} "
                f"stage1_pa={rec['stage1_pa_mpjpe_mm_lower_better']:.3f} "
                f"stage2_pa={rec['stage2_pa_mpjpe_mm_lower_better']:.3f}",
                flush=True,
            )
    out_df = pd.DataFrame(rows)
    out_csv = args.run_root / "val_stage1_vs_stage2_motion_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    summary = []
    for dataset, group in out_df.groupby("dataset"):
        rec = {"dataset": dataset, "n_clips": int(len(group))}
        for col in [c for c in out_df.columns if c.endswith("_lower_better")]:
            rec[f"{col}_mean"] = float(group[col].mean())
        summary.append(rec)
    summary_df = pd.DataFrame(summary)
    summary_csv = args.run_root / "val_stage1_vs_stage2_motion_summary.csv"
    summary_json = args.run_root / "val_stage1_vs_stage2_motion_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps({"metrics_csv": str(out_csv), "per_dataset": summary}, indent=2), encoding="utf-8")
    print("Wrote motion metrics:", out_csv, flush=True)
    print(summary_df.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Notebook-39-style continuous refiner on SOKE20 NB51 tokenizer outputs.")
    parser.add_argument("--preprocess-root", type=Path, required=True)
    parser.add_argument("--stage1-run-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, default=WORK_ROOT / "body_models")
    parser.add_argument("--stage1-checkpoint", type=str, default="best.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--windows-per-clip", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial-in-blocks", type=int, default=2)
    parser.add_argument("--temporal-blocks", type=int, default=2)
    parser.add_argument("--spatial-out-blocks", type=int, default=2)
    parser.add_argument("--stats-frame-cap", type=int, default=200000)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-val-clips", type=int, default=0)
    parser.add_argument("--rebuild-recon-cache", action="store_true")
    parser.add_argument("--rebuild-stats", action="store_true")
    parser.add_argument("--skip-recon-cache", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--run-motion-metrics", type=int, default=0)
    parser.add_argument("--metric-max-clips", type=int, default=0)
    parser.add_argument("--metric-frame-batch", type=int, default=4096)
    parser.add_argument("--metric-pa-block-rows", type=int, default=32)
    parser.add_argument("--metric-raw-block-rows", type=int, default=128)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.run_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()), flush=True)
    train_df = load_manifest(args.preprocess_root, "train", args.max_train_clips, args.seed)
    val_df = load_manifest(args.preprocess_root, "val", args.max_val_clips, args.seed)
    print(
        {
            "train_clips": int(len(train_df)),
            "val_clips": int(len(val_df)),
            "datasets_train": train_df["dataset"].value_counts().to_dict(),
            "datasets_val": val_df["dataset"].value_counts().to_dict(),
            "run_root": str(args.run_root),
        },
        flush=True,
    )
    recon_cache_root = args.run_root / "stage1_fullclip_recon_cache"
    if not args.skip_recon_cache:
        runner = NB51Runner(args.stage1_run_root, args.stage1_checkpoint, device)
        train_recon_df = build_recon_cache(
            train_df,
            split="train",
            runner=runner,
            recon_cache_root=recon_cache_root,
            stride_size=args.stride_size,
            force=args.rebuild_recon_cache,
            save_every=args.save_every,
        )
        val_recon_df = build_recon_cache(
            val_df,
            split="val",
            runner=runner,
            recon_cache_root=recon_cache_root,
            stride_size=args.stride_size,
            force=args.rebuild_recon_cache,
            save_every=args.save_every,
        )
        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        train_recon_df = load_existing_manifest(recon_cache_root / "train_recon_manifest.csv")
        val_recon_df = load_existing_manifest(recon_cache_root / "val_recon_manifest.csv")
        if train_recon_df.empty or val_recon_df.empty:
            raise RuntimeError("skip-recon-cache was set but recon manifests are missing or empty")

    if not args.skip_training:
        train_refiner(args, train_recon_df, val_recon_df, device)
    else:
        print("Training skipped by flag.", flush=True)
    if int(args.run_motion_metrics):
        evaluate_motion_metrics(args, val_recon_df, device)


if __name__ == "__main__":
    main()
