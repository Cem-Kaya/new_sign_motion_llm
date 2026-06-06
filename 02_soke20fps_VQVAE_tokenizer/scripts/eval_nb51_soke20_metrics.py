from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import smplx  # noqa: E402
from joint_token_rmsnorm_gqa_vqvae import (  # noqa: E402
    JointTokenRMSGQAVQVAE,
    ModelConfig,
    PerChannelUniformQuantizer,
    build_default_token_groups,
    default_group_indices_and_mask,
    group_scalar_token_ids,
    grouped_logits_to_expected_values,
)
from soke20_preprocess import (  # noqa: E402
    CANON,
    CANON_BETAS,
    ROT6D_BODY_KEEP,
    ROT6D_LEG,
    ROT6D_LHAND,
    ROT6D_RHAND,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_normalize_body_model_root(path: Path) -> Path:
    p = Path(path).expanduser().resolve()
    if (p / "smplx" / "SMPLX_NEUTRAL.npz").exists() or (p / "smplx" / "SMPLX_NEUTRAL.pkl").exists():
        return p
    if (p / "SMPLX_NEUTRAL.npz").exists() or (p / "SMPLX_NEUTRAL.pkl").exists():
        return p.parent
    return p


def load_npz_entry(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as z:
        pred = np.asarray(z["pred"], dtype=np.float32)
        copy = np.asarray(z["copy"], dtype=np.float32)
        meta = np.asarray(z["meta"], dtype=np.float32)
    return (
        np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(copy, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(meta, nan=0.0, posinf=0.0, neginf=0.0),
    )


def reconstruct_full169(pred_feat: np.ndarray, copy_feat: np.ndarray, meta_vec: np.ndarray) -> np.ndarray:
    pred_feat = np.asarray(pred_feat, dtype=np.float32)
    copy_feat = np.asarray(copy_feat, dtype=np.float32)
    meta_vec = np.asarray(meta_vec, dtype=np.float32)
    feat327 = np.zeros((pred_feat.shape[0], 327), dtype=np.float32)
    feat327[:, :5] = copy_feat[:, :5]
    feat327[:, ROT6D_BODY_KEEP] = pred_feat[:, :84]
    feat327[:, ROT6D_LEG] = copy_feat[:, 5:53]
    feat327[:, ROT6D_LHAND] = pred_feat[:, 97:187]
    feat327[:, ROT6D_RHAND] = pred_feat[:, 187:277]
    feat327[:, CANON_BETAS] = copy_feat[:, 53:63]
    meta = {
        "yaw0": float(meta_vec[0]),
        "floor_y": float(meta_vec[1]),
        "origin_x": float(meta_vec[2]),
        "origin_z": float(meta_vec[3]),
    }
    return np.asarray(CANON.decode(feat327, meta), dtype=np.float32)


class SMPLXJ22Decoder:
    def __init__(self, body_model_root: Path, device: torch.device) -> None:
        self.device = device
        root = maybe_normalize_body_model_root(body_model_root)
        self.model = smplx.create(
            model_path=str(root),
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            num_betas=10,
        ).to(device)
        self.model.eval()
        self._z_cache: dict[int, torch.Tensor] = {}
        self._expr_cache: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, motion169: np.ndarray, frame_batch: int) -> np.ndarray:
        arr = np.asarray(motion169, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 169:
            raise ValueError(f"Expected [T,169] SMPL-X motion, got {arr.shape}")
        out_j = np.empty((arr.shape[0], 22, 3), dtype=np.float32)
        for start in range(0, arr.shape[0], int(frame_batch)):
            end = min(arr.shape[0], start + int(frame_batch))
            chunk = arr[start:end]
            aa = chunk[:, :156].reshape(-1, 52, 3)
            bet = chunk[:, 156:166]
            tr = chunk[:, 166:169]
            bsz = int(chunk.shape[0])
            if bsz not in self._z_cache:
                self._z_cache[bsz] = torch.zeros((bsz, 3), dtype=torch.float32, device=self.device)
                self._expr_cache[bsz] = torch.zeros((bsz, 10), dtype=torch.float32, device=self.device)
            out = self.model(
                global_orient=torch.tensor(aa[:, 0], dtype=torch.float32, device=self.device),
                body_pose=torch.tensor(aa[:, 1:22].reshape(bsz, 63), dtype=torch.float32, device=self.device),
                left_hand_pose=torch.tensor(aa[:, 22:37].reshape(bsz, 45), dtype=torch.float32, device=self.device),
                right_hand_pose=torch.tensor(aa[:, 37:52].reshape(bsz, 45), dtype=torch.float32, device=self.device),
                jaw_pose=self._z_cache[bsz],
                leye_pose=self._z_cache[bsz],
                reye_pose=self._z_cache[bsz],
                expression=self._expr_cache[bsz],
                betas=torch.tensor(bet, dtype=torch.float32, device=self.device),
                transl=torch.tensor(tr, dtype=torch.float32, device=self.device),
            )
            joints = out.joints[:, :22, :].detach().cpu().numpy().astype(np.float32)
            out_j[start:end] = joints - joints[:, [0], :]
        return out_j


class NB51Runner:
    def __init__(self, run_root: Path, checkpoint_name: str, device: torch.device) -> None:
        self.run_root = Path(run_root)
        self.device = device
        payload = torch.load(self.run_root / checkpoint_name, map_location="cpu", weights_only=False)
        self.cfg = ModelConfig(**payload["model_cfg"])
        self.model = JointTokenRMSGQAVQVAE(self.cfg).to(device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.quantizer = PerChannelUniformQuantizer.from_state_dict(payload["quantizer_state"])
        self.centers_t = torch.tensor(self.quantizer.centers, dtype=torch.float32, device=device)
        groups = build_default_token_groups()
        self.group_index_np, self.group_valid_mask_np = default_group_indices_and_mask(self.cfg.max_group_dim, groups)
        self.group_index_t = torch.tensor(self.group_index_np, dtype=torch.long, device=device)
        self.group_valid_mask_t = torch.tensor(self.group_valid_mask_np, dtype=torch.bool, device=device)
        self.amp_enabled = device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.amp_enabled and torch.cuda.is_bf16_supported() else torch.float16

    @staticmethod
    def _starts(num_frames: int, window_size: int, stride_size: int) -> list[int]:
        if num_frames <= window_size:
            return [0]
        starts = list(range(0, num_frames - window_size + 1, stride_size))
        final_start = num_frames - window_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    @staticmethod
    def _pad_window(win: np.ndarray, window_size: int) -> np.ndarray:
        if win.shape[0] >= window_size:
            return win
        return np.concatenate([win, np.repeat(win[-1:, :], window_size - win.shape[0], axis=0)], axis=0)

    @torch.no_grad()
    def predict(self, feat_full: np.ndarray, stride_size: int) -> np.ndarray:
        feat_full = np.asarray(feat_full, dtype=np.float32)
        t, f = feat_full.shape
        accum = np.zeros((t, f), dtype=np.float32)
        counts = np.zeros((t, 1), dtype=np.float32)
        for start in self._starts(t, self.cfg.window_size, stride_size):
            end = min(t, start + self.cfg.window_size)
            win = self._pad_window(feat_full[start:end], self.cfg.window_size)
            scalar_token_ids = self.quantizer.encode(win)
            group_token_ids = group_scalar_token_ids(scalar_token_ids, self.group_index_np, pad_value=0)
            xb = torch.from_numpy(group_token_ids[None]).long().to(self.device)
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                out = self.model(xb)
                pred = grouped_logits_to_expected_values(
                    out["logits"],
                    self.centers_t,
                    self.group_index_t,
                    self.group_valid_mask_t,
                )[0].float().detach().cpu().numpy()
            keep = end - start
            accum[start:end] += pred[:keep]
            counts[start:end] += 1.0
        return (accum / np.clip(counts, 1e-6, None)).astype(np.float32)


def dtw_last_cell(costs: np.ndarray) -> float:
    costs = np.asarray(costs, dtype=np.float64)
    n, m = costs.shape
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        row = costs[i - 1]
        for j in range(1, m + 1):
            curr[j] = row[j - 1] + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / max(n, m))


def mpjpe_cost_matrix_blocked_torch(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    *,
    device: torch.device,
    block_rows: int = 128,
) -> np.ndarray:
    pred = torch.as_tensor(np.asarray(pred_seq, dtype=np.float32), dtype=torch.float32, device=device)
    gt = torch.as_tensor(np.asarray(gt_seq, dtype=np.float32), dtype=torch.float32, device=device)
    n = int(pred.shape[0])
    m = int(gt.shape[0])
    costs = np.empty((n, m), dtype=np.float32)
    gt_base = gt[None, :, :, :]
    for start in range(0, n, int(block_rows)):
        end = min(n, start + int(block_rows))
        cost = torch.linalg.norm(pred[start:end, None, :, :] - gt_base, dim=-1).mean(dim=-1)
        costs[start:end] = cost.detach().cpu().numpy()
    return costs


def pa_cost_matrix_blocked_torch(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    *,
    device: torch.device,
    block_rows: int = 32,
) -> np.ndarray:
    pred = torch.as_tensor(np.asarray(pred_seq, dtype=np.float32), dtype=torch.float32, device=device)
    gt = torch.as_tensor(np.asarray(gt_seq, dtype=np.float32), dtype=torch.float32, device=device)
    n, joints, _ = pred.shape
    m = int(gt.shape[0])
    costs = np.empty((int(n), int(m)), dtype=np.float32)
    gt_base = gt[None, :, :, :]
    eye = torch.eye(3, dtype=torch.float32, device=device)
    for start in range(0, int(n), int(block_rows)):
        end = min(int(n), start + int(block_rows))
        b = end - start
        x = pred[start:end, None, :, :].expand(b, m, joints, 3).reshape(b * m, joints, 3)
        y = gt_base.expand(b, m, joints, 3).reshape(b * m, joints, 3)
        mu_x = x.mean(dim=1)
        mu_y = y.mean(dim=1)
        x0 = x - mu_x[:, None, :]
        y0 = y - mu_y[:, None, :]
        var_x = (x0.square().sum(dim=(1, 2)) / float(joints)).clamp_min(1e-12)
        cov = torch.einsum("pji,pjk->pik", x0, y0) / float(joints)
        u, d, vh = torch.linalg.svd(cov)
        det = torch.det(torch.matmul(u, vh))
        sdiag = torch.ones((b * m, 3), dtype=torch.float32, device=device)
        sdiag[det < 0, -1] = -1.0
        s_mat = eye.unsqueeze(0).expand(b * m, 3, 3).clone()
        s_mat[:, 2, 2] = sdiag[:, 2]
        rot = torch.matmul(torch.matmul(u, s_mat), vh)
        scale = (d * sdiag).sum(dim=1) / var_x
        xr = torch.einsum("pjd,pde->pje", x, rot)
        muxr = torch.einsum("pd,pde->pe", mu_x, rot)
        trans = mu_y - scale[:, None] * muxr
        aligned = scale[:, None, None] * xr + trans[:, None, :]
        cost = torch.linalg.norm(aligned - y, dim=-1).mean(dim=1)
        costs[start:end] = cost.reshape(b, m).detach().cpu().numpy()
    return costs


def summarize_by_dataset(out_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mpjpe_mm_lower_better",
        "jpe_mm_lower_better",
        "pa_mpjpe_mm_lower_better",
        "dtw_mpjpe_mm_lower_better",
        "dtw_pa_mpjpe_mm_lower_better",
    ]
    present = [c for c in metric_cols if c in out_df.columns]
    rows: list[dict[str, Any]] = []
    for (dataset, source_alias), group in out_df.groupby(["dataset", "source_alias"], dropna=False):
        rec: dict[str, Any] = {
            "dataset": dataset,
            "source_alias": source_alias,
            "n_clips": int(len(group)),
            "frames_mean": float(group["num_frames"].mean()),
        }
        for col in present:
            rec[f"{col}_mean"] = float(group[col].mean())
            rec[f"{col}_median"] = float(group[col].median())
            rec[f"{col}_p90"] = float(group[col].quantile(0.90))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["dataset", "source_alias"]).reset_index(drop=True)


def build_summary_display_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        ("dataset", "Dataset"),
        ("source_alias", "Alias"),
        ("n_clips", "Clips"),
        ("frames_mean", "Mean frames"),
        ("mpjpe_mm_lower_better_mean", "MPJPE mean mm ↓"),
        ("mpjpe_mm_lower_better_median", "MPJPE median mm ↓"),
        ("pa_mpjpe_mm_lower_better_mean", "PA-MPJPE mean mm ↓"),
        ("pa_mpjpe_mm_lower_better_median", "PA-MPJPE median mm ↓"),
        ("dtw_mpjpe_mm_lower_better_mean", "DTW-MPJPE mean mm ↓"),
        ("dtw_pa_mpjpe_mm_lower_better_mean", "DTW-PA-MPJPE mean mm ↓"),
    ]
    present = [(src, dst) for src, dst in columns if src in summary_df.columns]
    out = summary_df[[src for src, _ in present]].copy()
    out.columns = [dst for _, dst in present]
    return out


def write_summary_table(summary_table_df: pd.DataFrame, table_png_path: Path) -> None:
    if summary_table_df.empty:
        return
    display_df = summary_table_df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        elif pd.api.types.is_integer_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: "" if pd.isna(x) else str(int(x)))
        else:
            display_df[col] = display_df[col].astype(str)

    col_labels = [c.replace(" ", "\n").replace("↓", "↓") for c in display_df.columns]
    fig_width = max(13.0, 1.25 * len(display_df.columns))
    fig_height = max(3.2, 0.55 * len(display_df) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title("Per-Dataset Validation Motion Metrics Table (lower better)", fontsize=15, fontweight="bold", pad=14)
    table = ax.table(
        cellText=display_df.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.6)

    n_rows, n_cols = display_df.shape
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold", color="#111827")
            cell.set_height(cell.get_height() * 1.45)
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")
        if col_idx < n_cols and display_df.columns[col_idx] in {"Dataset", "Alias"}:
            cell.set_text_props(ha="left")
            cell.set_width(0.14 if display_df.columns[col_idx] == "Dataset" else 0.08)
        else:
            cell.set_width(0.09)

    fig.text(
        0.01,
        0.03,
        "CSV companion contains the same compact table; detailed mean/median/p90 summary remains in the regular summary CSV.",
        fontsize=8.5,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    table_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(table_png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_plot(summary_df: pd.DataFrame, plot_path: Path) -> None:
    # Backward-compatible wrapper: this path is now a table PNG, not a graph.
    if summary_df.empty:
        return
    write_summary_table(build_summary_display_table(summary_df), plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-dataset full-clip validation metrics for the SOKE20 NB51 tokenizer.")
    parser.add_argument("--preprocess-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-clips-per-dataset", type=int, default=0)
    parser.add_argument("--stride-size", type=int, default=32)
    parser.add_argument("--frame-batch", type=int, default=4096)
    parser.add_argument("--pa-block-rows", type=int, default=32)
    parser.add_argument("--raw-block-rows", type=int, default=128)
    parser.add_argument("--compute-dtw-pa", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    manifest_path = args.preprocess_root / f"{args.split}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not (args.run_root / args.checkpoint).exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.run_root / args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

    df = pd.read_csv(manifest_path)
    if args.max_clips_per_dataset > 0:
        chunks = []
        for _, group in df.groupby("dataset", sort=True):
            n = min(int(args.max_clips_per_dataset), len(group))
            chunks.append(group.sample(n=n, random_state=args.seed))
        df = pd.concat(chunks, ignore_index=True).sort_values(["dataset", "clip_id"]).reset_index(drop=True)

    suffix = f"{args.split}_fullclip_per_dataset_motion"
    if args.max_clips_per_dataset > 0:
        suffix += f"_limit{args.max_clips_per_dataset}"
    out_csv = args.run_root / f"{suffix}_metrics.csv"
    summary_csv = args.run_root / f"{suffix}_summary.csv"
    summary_table_csv = args.run_root / f"{suffix}_summary_table.csv"
    summary_json = args.run_root / f"{suffix}_summary.json"
    plot_path = args.run_root / "plots" / f"{suffix}_summary.png"

    existing: pd.DataFrame | None = None
    done: set[str] = set()
    if out_csv.exists() and not args.force:
        existing = pd.read_csv(out_csv)
        if not existing.empty:
            done = set(existing["clip_key"].astype(str))
            print(f"Resuming metrics from {out_csv} with {len(done)} completed clips")

    runner = NB51Runner(args.run_root, args.checkpoint, device)
    decoder = SMPLXJ22Decoder(args.body_model_root, device)
    rows: list[dict[str, Any]] = []
    if existing is not None and not existing.empty:
        rows.extend(existing.to_dict("records"))

    started = time.time()
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        dataset = str(getattr(row, "dataset"))
        split = str(getattr(row, "split"))
        clip_id = str(getattr(row, "clip_id"))
        source_alias = str(getattr(row, "source_alias", dataset))
        clip_key = f"{dataset}/{split}/{clip_id}"
        if clip_key in done:
            continue

        feat, copy_feat, meta_vec = load_npz_entry(getattr(row, "cache_npz"))
        pred_feat = runner.predict(feat, stride_size=args.stride_size)
        gt169 = reconstruct_full169(feat, copy_feat, meta_vec)
        pr169 = reconstruct_full169(pred_feat, copy_feat, meta_vec)
        gt_j = decoder(gt169, frame_batch=args.frame_batch)
        pr_j = decoder(pr169, frame_batch=args.frame_batch)

        raw_costs = mpjpe_cost_matrix_blocked_torch(
            pr_j,
            gt_j,
            device=device,
            block_rows=args.raw_block_rows,
        )
        if int(args.compute_dtw_pa):
            pa_costs = pa_cost_matrix_blocked_torch(
                pr_j,
                gt_j,
                device=device,
                block_rows=args.pa_block_rows,
            )
            dtw_pa_mpjpe_mm = float(dtw_last_cell(pa_costs.astype(np.float64)) * 1000.0)
            pa_mpjpe_mm = float(np.diag(pa_costs).mean() * 1000.0)
        else:
            diag_pa = pa_cost_matrix_blocked_torch(
                pr_j,
                gt_j,
                device=device,
                block_rows=max(1, min(args.pa_block_rows, 16)),
            )
            pa_mpjpe_mm = float(np.diag(diag_pa).mean() * 1000.0)
            dtw_pa_mpjpe_mm = float("nan")

        mpjpe_mm = float(np.mean(np.linalg.norm(pr_j - gt_j, axis=-1)) * 1000.0)
        dtw_mpjpe_mm = float(dtw_last_cell(raw_costs.astype(np.float64)) * 1000.0)
        rec = {
            "clip_key": clip_key,
            "dataset": dataset,
            "source_alias": source_alias,
            "split": split,
            "clip_id": clip_id,
            "cache_npz": str(getattr(row, "cache_npz")),
            "num_frames": int(feat.shape[0]),
            "target_duration_sec": float(getattr(row, "target_duration_sec", feat.shape[0] / 20.0)),
            "mpjpe_mm_lower_better": mpjpe_mm,
            "jpe_mm_lower_better": mpjpe_mm,
            "pa_mpjpe_mm_lower_better": pa_mpjpe_mm,
            "dtw_mpjpe_mm_lower_better": dtw_mpjpe_mm,
            "dtw_pa_mpjpe_mm_lower_better": dtw_pa_mpjpe_mm,
        }
        rows.append(rec)
        done.add(clip_key)

        if idx == 1 or idx % 25 == 0 or idx == len(df):
            elapsed = (time.time() - started) / 60.0
            print(
                f"metrics {idx}/{len(df)} | {dataset} | frames={feat.shape[0]} "
                f"mpjpe={mpjpe_mm:.3f} pa={pa_mpjpe_mm:.3f} dtw={dtw_mpjpe_mm:.3f} "
                f"dtw_pa={dtw_pa_mpjpe_mm:.3f} elapsed={elapsed:.1f}m",
                flush=True,
            )
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    summary_df = summarize_by_dataset(out_df)
    summary_df.to_csv(summary_csv, index=False)
    summary_table_df = build_summary_display_table(summary_df)
    summary_table_df.to_csv(summary_table_csv, index=False)
    write_summary_plot(summary_df, plot_path)

    payload = {
        "run_root": str(args.run_root),
        "preprocess_root": str(args.preprocess_root),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "n_clips": int(len(out_df)),
        "max_clips_per_dataset": int(args.max_clips_per_dataset),
        "compute_dtw_pa": bool(args.compute_dtw_pa),
        "metrics_csv": str(out_csv),
        "summary_csv": str(summary_csv),
        "summary_table_csv": str(summary_table_csv),
        "summary_table_png": str(plot_path),
        "plot_path": str(plot_path),
        "plot_path_note": "This path is a matplotlib table PNG, not a graph.",
        "per_dataset": summary_df.to_dict("records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote:", out_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", summary_table_csv)
    print("Wrote:", summary_json)
    print("Wrote:", plot_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
