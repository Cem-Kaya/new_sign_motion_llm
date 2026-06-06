from __future__ import annotations

import argparse
import collections
import io
import json
import math
import os
import pickle
import random
import sys
import time
import typing
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parents[2]
REF_SOKE = WORKSPACE_DIR / "ref" / "SOKE"
if str(REF_SOKE) not in sys.path:
    sys.path.insert(0, str(REF_SOKE))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_nb51_soke20_metrics import (  # noqa: E402
    SMPLXJ22Decoder,
    dtw_last_cell,
    mpjpe_cost_matrix_blocked_torch,
    pa_cost_matrix_blocked_torch,
)
from reproduction.soke_lm_reproduction import SOKEThreeStreamTokenizer  # noqa: E402
from soke20_preprocess import (  # noqa: E402
    frame_sort_key,
    frame_to_smplx182,
    load_frame,
    sample_frame_paths,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_official_safe_globals() -> None:
    import numpy.core.multiarray as ma
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.dictconfig import DictConfig
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode, BooleanNode, FloatNode, IntegerNode, StringNode

    class WordVectorizer:
        pass

    WordVectorizer.__module__ = "mGPT.data.humanml.utils.word_vectorizer"
    WordVectorizer.__qualname__ = "WordVectorizer"
    np_dtype_classes = []
    for name in ["Float64DType", "Float32DType", "Int64DType", "Int32DType", "BoolDType"]:
        if hasattr(np.dtypes, name):
            np_dtype_classes.append(getattr(np.dtypes, name))
    torch.serialization.add_safe_globals(
        [
            typing.Any,
            list,
            dict,
            tuple,
            set,
            frozenset,
            slice,
            collections.defaultdict,
            int,
            float,
            str,
            bool,
            type(None),
            WordVectorizer,
            ma._reconstruct,
            np.ndarray,
            np.dtype,
            *np_dtype_classes,
            ListConfig,
            DictConfig,
            ContainerMetadata,
            Metadata,
            AnyNode,
            StringNode,
            IntegerNode,
            FloatNode,
            BooleanNode,
        ]
    )


def load_official_model(checkpoint: Path, device: torch.device) -> tuple[SOKEThreeStreamTokenizer, dict[str, Any]]:
    add_official_safe_globals()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = {
        "name": "official_soke_paper_vqvae",
        "quantizer": "ema_reset",
        "code_dim": 512,
        "output_emb_width": 512,
        "down_t": 2,
        "stride_t": 2,
        "width": 512,
        "depth": 3,
        "dilation_growth_rate": 3,
        "activation": "relu",
        "code_num_body": 96,
        "code_num_lhand": 192,
        "code_num_rhand": 192,
    }
    model = SOKEThreeStreamTokenizer(cfg).to(device)
    mapped: dict[str, torch.Tensor] = {}
    for key, val in ckpt["state_dict"].items():
        if key.startswith("vae."):
            mapped["body_vae." + key[4:]] = val
        elif key.startswith("hand_vae."):
            mapped["lhand_vae." + key[len("hand_vae.") :]] = val
        elif key.startswith("rhand_vae."):
            mapped["rhand_vae." + key[len("rhand_vae.") :]] = val
    result = model.load_state_dict(mapped, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"official state mismatch missing={result.missing_keys[:8]} unexpected={result.unexpected_keys[:8]}")
    model.eval()
    meta = {
        "checkpoint": str(checkpoint),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "mapped_tensor_count": len(mapped),
        "model_cfg": cfg,
    }
    return model, meta


def official_mean_std_133() -> tuple[np.ndarray, np.ndarray, str]:
    mean179 = torch.load(WORKSPACE_DIR / "ref" / "data" / "CSL-Daily" / "mean.pt", map_location="cpu").detach().cpu().numpy().astype(np.float32)
    std179 = torch.load(WORKSPACE_DIR / "ref" / "data" / "CSL-Daily" / "std.pt", map_location="cpu").detach().cpu().numpy().astype(np.float32)
    rest_m = mean179[36:]
    rest_s = std179[36:]
    mean = np.concatenate([rest_m[:123], rest_m[133:143]], axis=0).astype(np.float32)
    std = np.concatenate([rest_s[:123], rest_s[133:143]], axis=0).astype(np.float32)
    return mean, std, "ref/data/CSL-Daily/mean.pt + std.pt sliced to SOKE 133D"


def local_mean_std_133() -> tuple[np.ndarray, np.ndarray, str]:
    root = WORKSPACE_DIR / "artifacts" / "20_soke_repo_style_stage1"
    mean = np.load(root / "feature_mean_133.npy").astype(np.float32)
    std = np.load(root / "feature_std_133.npy").astype(np.float32)
    return mean, std, "artifacts/20_soke_repo_style_stage1/feature_mean_133.npy + feature_std_133.npy"


def soke_preprocess_133(smplx182: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(smplx182, dtype=np.float32)
    core179 = x[:, :179]
    transl3 = x[:, 179:182]
    rest = core179[:, 36:]
    feat133 = np.concatenate([rest[:, :123], rest[:, 133:143]], axis=-1)
    aux36 = core179[:, :36]
    betas10 = core179[:, 159:169]
    return (
        np.nan_to_num(feat133).astype(np.float32),
        np.nan_to_num(aux36).astype(np.float32),
        np.nan_to_num(betas10).astype(np.float32),
        np.nan_to_num(transl3).astype(np.float32),
    )


def soke133_to_smplx169(feat133: np.ndarray, aux36: np.ndarray, betas10: np.ndarray, transl3: np.ndarray) -> np.ndarray:
    feat = np.asarray(feat133, dtype=np.float32)
    out = np.zeros((feat.shape[0], 169), dtype=np.float32)
    out[:, :156] = np.concatenate([aux36, feat[:, :120]], axis=-1)
    out[:, 156:166] = betas10
    out[:, 166:169] = transl3
    return np.nan_to_num(out).astype(np.float32)


def pad_to_multiple(x: np.ndarray, multiple: int) -> tuple[np.ndarray, int]:
    t = int(x.shape[0])
    target = int(math.ceil(t / float(multiple)) * multiple)
    if target == t:
        return x, t
    pad = np.repeat(x[-1:, :], target - t, axis=0)
    return np.concatenate([x, pad], axis=0).astype(np.float32), t


def load_clip_features(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose_dir = Path(str(row["pose_dir"]))
    if not pose_dir.exists():
        raise FileNotFoundError(f"Missing pose_dir: {pose_dir}")
    frame_files = sorted(pose_dir.glob("*.pkl"), key=frame_sort_key)
    sampled_files = sample_frame_paths(
        frame_files,
        source_fps=float(row.get("source_fps", 24.0)),
        target_fps=float(row.get("target_fps", 20.0)),
    )
    smplx182 = np.stack([frame_to_smplx182(load_frame(p)) for p in sampled_files], axis=0).astype(np.float32)
    return soke_preprocess_133(smplx182)


@torch.no_grad()
def reconstruct_clip(
    model: SOKEThreeStreamTokenizer,
    feat133: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    padded, original_t = pad_to_multiple(np.asarray(feat133, dtype=np.float32), 4)
    x = ((padded - mean[None, :]) / std[None, :]).astype(np.float32)
    xb = torch.from_numpy(x[None]).to(device)
    parts = model.encode_parts(xb)
    y = model.decode_parts(parts["body"][0:1], parts["lhand"][0:1], parts["rhand"][0:1])
    out = y.detach().float().cpu().numpy()[0] * std[None, :] + mean[None, :]
    return out[:original_t].astype(np.float32)


def summarize_by_dataset(out_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "mpjpe_mm_lower_better",
        "pa_mpjpe_mm_lower_better",
        "dtw_mpjpe_mm_lower_better",
        "dtw_pa_mpjpe_mm_lower_better",
    ]
    rows: list[dict[str, Any]] = []
    for (dataset, source_alias), group in out_df.groupby(["dataset", "source_alias"], dropna=False):
        rec: dict[str, Any] = {
            "dataset": dataset,
            "source_alias": source_alias,
            "n_clips": int(len(group)),
            "frames_mean": float(group["num_frames"].mean()),
        }
        for col in metric_cols:
            rec[f"{col}_mean"] = float(group[col].mean())
            rec[f"{col}_median"] = float(group[col].median())
            rec[f"{col}_p90"] = float(group[col].quantile(0.90))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["dataset", "source_alias"]).reset_index(drop=True)


def summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    present = [(src, dst) for src, dst in cols if src in summary_df.columns]
    out = summary_df[[src for src, _ in present]].copy()
    out.columns = [dst for _, dst in present]
    return out


def write_table_png(table_df: pd.DataFrame, path: Path, title: str) -> None:
    display = table_df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        elif pd.api.types.is_integer_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(int(x)))
        else:
            display[col] = display[col].astype(str)
    labels = [c.replace(" ", "\n") for c in display.columns]
    fig, ax = plt.subplots(figsize=(max(13.0, 1.25 * len(display.columns)), max(3.2, 0.55 * len(display) + 1.6)))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    table = ax.table(cellText=display.values, colLabels=labels, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.6)
    n_cols = display.shape[1]
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold", color="#111827")
            cell.set_height(cell.get_height() * 1.45)
        elif r % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")
        if c < n_cols and display.columns[c] in {"Dataset", "Alias"}:
            cell.set_text_props(ha="left")
            cell.set_width(0.14 if display.columns[c] == "Dataset" else 0.08)
        else:
            cell.set_width(0.09)
    fig.text(0.01, 0.03, "Official downloaded SOKE paper VQ-VAE checkpoint. Lower is better.", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the official downloaded SOKE paper VQ-VAE on new-data validation clips.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR.parent / "outputs" / "preprocess_soke20" / "val_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR.parent / "outputs" / "runs" / "official_soke_paper_vqvae_val",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_DIR / "ref" / "SOKE" / "experiments" / "mgpt" / "vae" / "checkpoints" / "tokenizer.ckpt",
    )
    parser.add_argument("--body-model-root", type=Path, default=WORKSPACE_DIR / "body_models")
    parser.add_argument("--norm-source", choices=["official", "local"], default="official")
    parser.add_argument("--max-clips-per-dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-batch", type=int, default=4096)
    parser.add_argument("--pa-block-rows", type=int, default=32)
    parser.add_argument("--raw-block-rows", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing official SOKE checkpoint: {args.checkpoint}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

    df = pd.read_csv(args.manifest)
    if args.max_clips_per_dataset > 0:
        chunks = []
        for _, group in df.groupby("dataset", sort=True):
            n = min(int(args.max_clips_per_dataset), len(group))
            chunks.append(group.sample(n=n, random_state=args.seed))
        df = pd.concat(chunks, ignore_index=True).sort_values(["dataset", "clip_id"]).reset_index(drop=True)

    split_name = args.manifest.stem.replace("_manifest", "") or "split"
    suffix = f"official_soke_paper_vqvae_{split_name}"
    if args.norm_source == "local":
        suffix += "_local_norm"
    else:
        suffix += "_official_norm"
    if args.max_clips_per_dataset > 0:
        suffix += f"_limit{args.max_clips_per_dataset}"
    metrics_csv = args.output_root / f"{suffix}_metrics.csv"
    summary_csv = args.output_root / f"{suffix}_summary.csv"
    table_csv = args.output_root / f"{suffix}_summary_table.csv"
    summary_json = args.output_root / f"{suffix}_summary.json"
    table_png = args.output_root / "plots" / f"{suffix}_summary.png"

    existing: pd.DataFrame | None = None
    done: set[str] = set()
    if metrics_csv.exists() and not args.force:
        existing = pd.read_csv(metrics_csv)
        done = set(existing["clip_key"].astype(str)) if "clip_key" in existing.columns else set()
        if done:
            print(f"Resuming from {metrics_csv} with {len(done)} completed clips")

    model, model_meta = load_official_model(args.checkpoint, device)
    if args.norm_source == "official":
        mean, std, norm_source = official_mean_std_133()
    else:
        mean, std, norm_source = local_mean_std_133()
    decoder = SMPLXJ22Decoder(args.body_model_root, device)
    rows: list[dict[str, Any]] = []
    if existing is not None and not existing.empty:
        rows.extend(existing.to_dict("records"))

    started = time.time()
    for idx, row_tuple in enumerate(df.itertuples(index=False), start=1):
        row = pd.Series(row_tuple._asdict())
        dataset = str(row["dataset"])
        split = str(row["split"])
        clip_id = str(row["clip_id"])
        source_alias = str(row.get("source_alias", dataset))
        clip_key = f"{dataset}/{split}/{clip_id}"
        if clip_key in done:
            continue

        feat133, aux36, betas10, transl3 = load_clip_features(row)
        pred133 = reconstruct_clip(model, feat133, mean, std, device)
        gt169 = soke133_to_smplx169(feat133, aux36, betas10, transl3)
        pred169 = soke133_to_smplx169(pred133, aux36, betas10, transl3)
        gt_j = decoder(gt169, frame_batch=args.frame_batch)
        pred_j = decoder(pred169, frame_batch=args.frame_batch)

        raw_costs = mpjpe_cost_matrix_blocked_torch(pred_j, gt_j, device=device, block_rows=args.raw_block_rows)
        pa_costs = pa_cost_matrix_blocked_torch(pred_j, gt_j, device=device, block_rows=args.pa_block_rows)
        mpjpe_mm = float(np.mean(np.linalg.norm(pred_j - gt_j, axis=-1)) * 1000.0)
        pa_mpjpe_mm = float(np.diag(pa_costs).mean() * 1000.0)
        dtw_mpjpe_mm = float(dtw_last_cell(raw_costs.astype(np.float64)) * 1000.0)
        dtw_pa_mpjpe_mm = float(dtw_last_cell(pa_costs.astype(np.float64)) * 1000.0)
        rec = {
            "clip_key": clip_key,
            "dataset": dataset,
            "source_alias": source_alias,
            "split": split,
            "clip_id": clip_id,
            "pose_dir": str(row["pose_dir"]),
            "num_frames": int(feat133.shape[0]),
            "source_fps": float(row.get("source_fps", np.nan)),
            "target_fps": float(row.get("target_fps", np.nan)),
            "norm_source": args.norm_source,
            "model": "official_soke_paper_vqvae",
            "mpjpe_mm_lower_better": mpjpe_mm,
            "pa_mpjpe_mm_lower_better": pa_mpjpe_mm,
            "dtw_mpjpe_mm_lower_better": dtw_mpjpe_mm,
            "dtw_pa_mpjpe_mm_lower_better": dtw_pa_mpjpe_mm,
        }
        rows.append(rec)
        done.add(clip_key)

        if idx == 1 or idx % int(args.progress_every) == 0 or idx == len(df):
            elapsed = (time.time() - started) / 60.0
            print(
                f"official-soke {idx}/{len(df)} | {dataset} | frames={feat133.shape[0]} "
                f"mpjpe={mpjpe_mm:.3f} pa={pa_mpjpe_mm:.3f} dtw={dtw_mpjpe_mm:.3f} "
                f"dtw_pa={dtw_pa_mpjpe_mm:.3f} elapsed={elapsed:.1f}m",
                flush=True,
            )
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_csv, index=False)
    summary_df = summarize_by_dataset(metrics_df)
    summary_df.to_csv(summary_csv, index=False)
    table_df = summary_table(summary_df)
    table_df.to_csv(table_csv, index=False)
    write_table_png(table_df, table_png, "Official SOKE Paper VQ-VAE Validation Metrics (lower better)")

    payload = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "checkpoint": str(args.checkpoint),
        "norm_source": args.norm_source,
        "norm_source_detail": norm_source,
        "body_model_root": str(args.body_model_root),
        "n_clips": int(len(metrics_df)),
        "max_clips_per_dataset": int(args.max_clips_per_dataset),
        "model_meta": model_meta,
        "metrics_csv": str(metrics_csv),
        "summary_csv": str(summary_csv),
        "summary_table_csv": str(table_csv),
        "summary_table_png": str(table_png),
        "per_dataset": summary_df.to_dict("records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote:", metrics_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", table_csv)
    print("Wrote:", summary_json)
    print("Wrote:", table_png)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
