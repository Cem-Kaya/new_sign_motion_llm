from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import smplx  # noqa: E402
from eval_nb51_soke20_metrics import (  # noqa: E402
    NB51Runner,
    dtw_last_cell,
    load_npz_entry,
    maybe_normalize_body_model_root,
    pa_cost_matrix_blocked_torch,
    reconstruct_full169,
    set_seed,
)
from soke20_preprocess import (  # noqa: E402
    DEFAULT_SOURCE_FPS_BY_DATASET,
    process_row,
)


class SMPLXBodyHandDecoder:
    """Decode canonical 169D SMPL-X arrays into body and hand joint subsets."""

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
    def __call__(self, motion169: np.ndarray, frame_batch: int) -> dict[str, np.ndarray]:
        arr = np.asarray(motion169, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 169:
            raise ValueError(f"Expected [T,169] SMPL-X motion, got {arr.shape}")

        body_chunks: list[np.ndarray] = []
        hand_chunks: list[np.ndarray] = []
        lhand_chunks: list[np.ndarray] = []
        rhand_chunks: list[np.ndarray] = []
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
            joints = out.joints[:, :52, :].detach().cpu().numpy().astype(np.float32)
            joints = joints - joints[:, [0], :]
            body_chunks.append(joints[:, :22])
            lhand_chunks.append(joints[:, 22:37])
            rhand_chunks.append(joints[:, 37:52])
            hand_chunks.append(joints[:, 22:52])

        return {
            "body": np.concatenate(body_chunks, axis=0),
            "hand": np.concatenate(hand_chunks, axis=0),
            "lhand": np.concatenate(lhand_chunks, axis=0),
            "rhand": np.concatenate(rhand_chunks, axis=0),
        }


def build_source_fps_map(df: pd.DataFrame) -> dict[str, float]:
    out = dict(DEFAULT_SOURCE_FPS_BY_DATASET)
    if "source_fps" not in df.columns:
        return out
    for dataset, group in df.groupby("dataset"):
        values = pd.to_numeric(group["source_fps"], errors="coerce").dropna()
        if not values.empty:
            out[str(dataset)] = float(values.iloc[0])
    return out


def cache_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        out_root=args.preprocess_root,
        target_fps=args.target_fps,
        default_source_fps=args.default_source_fps,
        max_duration_sec=args.max_duration_sec,
        max_duration_all=args.max_duration_all,
        window_size=args.window_size,
        force=False,
    )


def ensure_cache(row: pd.Series, args: argparse.Namespace, source_fps_by_dataset: dict[str, float]) -> Path:
    cache_path = Path(str(row["cache_npz"]))
    if cache_path.exists():
        return cache_path
    rec = process_row(row, cache_args(args), source_fps_by_dataset)
    if rec.get("status") != "ok":
        raise RuntimeError(f"Could not rebuild cache for {row.get('clip_id')}: {rec.get('reason')}")
    rebuilt = Path(str(rec["cache_npz"]))
    if not rebuilt.exists():
        raise FileNotFoundError(f"Cache rebuild reported ok but file is missing: {rebuilt}")
    return rebuilt


def part_pa_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    device: torch.device,
    block_rows: int,
) -> tuple[float, float]:
    pa_costs = pa_cost_matrix_blocked_torch(pred, gt, device=device, block_rows=block_rows)
    pa_jpe_mm = float(np.diag(pa_costs).mean() * 1000.0)
    dtw_pa_mm = float(dtw_last_cell(pa_costs.astype(np.float64)) * 1000.0)
    return pa_jpe_mm, dtw_pa_mm


def summarize(rows_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "body_pa_jpe_mm_lower_better",
        "hand_lravg_pa_jpe_mm_lower_better",
        "hand_pa_jpe_mm_lower_better",
        "lhand_pa_jpe_mm_lower_better",
        "rhand_pa_jpe_mm_lower_better",
        "body_dtw_pa_mm_lower_better",
        "hand_lravg_dtw_pa_mm_lower_better",
        "hand_dtw_pa_mm_lower_better",
        "lhand_dtw_pa_mm_lower_better",
        "rhand_dtw_pa_mm_lower_better",
    ]
    rows: list[dict[str, Any]] = []
    for (dataset, source_alias), group in rows_df.groupby(["dataset", "source_alias"], dropna=False):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SOKE-table-style body/hand split metrics for the NB51 SOKE20 tokenizer."
    )
    parser.add_argument("--preprocess-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-clips-per-dataset", type=int, default=0)
    parser.add_argument("--stride-size", type=int, default=32)
    parser.add_argument("--frame-batch", type=int, default=4096)
    parser.add_argument("--pa-block-rows", type=int, default=24)
    parser.add_argument("--compute-combined-hand", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--default-source-fps", type=float, default=24.0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--max-duration-all", action="store_true")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=25)
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

    suffix = f"{args.split}_soke_table_body_hand"
    if args.max_clips_per_dataset > 0:
        suffix += f"_limit{args.max_clips_per_dataset}"
    out_csv = args.run_root / f"{suffix}_metrics.csv"
    summary_csv = args.run_root / f"{suffix}_summary.csv"
    summary_json = args.run_root / f"{suffix}_summary.json"

    existing: pd.DataFrame | None = None
    done: set[str] = set()
    if out_csv.exists() and not args.force:
        existing = pd.read_csv(out_csv)
        if not existing.empty:
            done = set(existing["clip_key"].astype(str))
            print(f"Resuming from {out_csv} with {len(done)} completed clips")

    source_fps_by_dataset = build_source_fps_map(df)
    runner = NB51Runner(args.run_root, args.checkpoint, device)
    decoder = SMPLXBodyHandDecoder(args.body_model_root, device)
    rows: list[dict[str, Any]] = []
    if existing is not None and not existing.empty:
        rows.extend(existing.to_dict("records"))

    started = time.time()
    rebuilt = 0
    for idx, row_tuple in enumerate(df.itertuples(index=False), start=1):
        row = pd.Series(row_tuple._asdict())
        dataset = str(row["dataset"])
        split = str(row["split"])
        clip_id = str(row["clip_id"])
        source_alias = str(row.get("source_alias", dataset))
        clip_key = f"{dataset}/{split}/{clip_id}"
        if clip_key in done:
            continue

        before_exists = Path(str(row["cache_npz"])).exists()
        cache_path = ensure_cache(row, args, source_fps_by_dataset)
        rebuilt += int(not before_exists)
        feat, copy_feat, meta_vec = load_npz_entry(cache_path)
        pred_feat = runner.predict(feat, stride_size=args.stride_size)
        gt169 = reconstruct_full169(feat, copy_feat, meta_vec)
        pr169 = reconstruct_full169(pred_feat, copy_feat, meta_vec)
        gt_parts = decoder(gt169, frame_batch=args.frame_batch)
        pr_parts = decoder(pr169, frame_batch=args.frame_batch)

        body_pa, body_dtw_pa = part_pa_metrics(
            pr_parts["body"],
            gt_parts["body"],
            device=device,
            block_rows=args.pa_block_rows,
        )
        lhand_pa, lhand_dtw_pa = part_pa_metrics(
            pr_parts["lhand"],
            gt_parts["lhand"],
            device=device,
            block_rows=args.pa_block_rows,
        )
        rhand_pa, rhand_dtw_pa = part_pa_metrics(
            pr_parts["rhand"],
            gt_parts["rhand"],
            device=device,
            block_rows=args.pa_block_rows,
        )
        if int(args.compute_combined_hand):
            hand_pa, hand_dtw_pa = part_pa_metrics(
                pr_parts["hand"],
                gt_parts["hand"],
                device=device,
                block_rows=args.pa_block_rows,
            )
        else:
            hand_pa = float("nan")
            hand_dtw_pa = float("nan")

        rows.append(
            {
                "clip_key": clip_key,
                "dataset": dataset,
                "source_alias": source_alias,
                "split": split,
                "clip_id": clip_id,
                "cache_npz": str(cache_path),
                "num_frames": int(feat.shape[0]),
                "body_pa_jpe_mm_lower_better": body_pa,
                "hand_lravg_pa_jpe_mm_lower_better": float((lhand_pa + rhand_pa) * 0.5),
                "hand_pa_jpe_mm_lower_better": hand_pa,
                "lhand_pa_jpe_mm_lower_better": lhand_pa,
                "rhand_pa_jpe_mm_lower_better": rhand_pa,
                "body_dtw_pa_mm_lower_better": body_dtw_pa,
                "hand_lravg_dtw_pa_mm_lower_better": float((lhand_dtw_pa + rhand_dtw_pa) * 0.5),
                "hand_dtw_pa_mm_lower_better": hand_dtw_pa,
                "lhand_dtw_pa_mm_lower_better": lhand_dtw_pa,
                "rhand_dtw_pa_mm_lower_better": rhand_dtw_pa,
            }
        )
        done.add(clip_key)

        if idx == 1 or idx % int(args.progress_every) == 0 or idx == len(df):
            elapsed = (time.time() - started) / 60.0
            print(
                f"split-metrics {idx}/{len(df)} | {dataset} | frames={feat.shape[0]} "
                f"body_pa={body_pa:.3f} hand_lr_pa={(lhand_pa + rhand_pa) * 0.5:.3f} "
                f"body_dtw={body_dtw_pa:.3f} hand_lr_dtw={(lhand_dtw_pa + rhand_dtw_pa) * 0.5:.3f} "
                f"rebuilt={rebuilt} elapsed={elapsed:.1f}m",
                flush=True,
            )
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_csv, index=False)
    summary_df = summarize(rows_df)
    summary_df.to_csv(summary_csv, index=False)
    payload = {
        "run_root": str(args.run_root),
        "preprocess_root": str(args.preprocess_root),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "n_clips": int(len(rows_df)),
        "rebuilt_cache_files": int(rebuilt),
            "joint_definition": "SMPL-X model joints: body=joints[0:22], hand=joints[22:52], lhand=joints[22:37], rhand=joints[37:52], root-relative before PA alignment. The SOKE-table Hand value should use hand_lravg_* columns, the mean of lhand/rhand, because SOKE's tokenizer has separate hand streams.",
        "metrics": {
            "reconstruction_columns": "body/hand PA-JPE mm lower better, matching SOKE table caption's PA-MPJPE reconstruction protocol as closely as this local SMPL-X joint set allows.",
            "generation_columns_proxy": "body/hand DTW-PA mm lower better over tokenizer reconstruction, not text-to-motion generation.",
        },
        "metrics_csv": str(out_csv),
        "summary_csv": str(summary_csv),
        "per_dataset": summary_df.to_dict("records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote:", out_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", summary_json)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
