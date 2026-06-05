from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from smplx_motiongpt_preprocess import SMPLXMotionGPTCanonicalizer


SOKE_BAD_HOW2SIGN_IDS = {
    "0DU7wWLK-QU_0-8-rgb_front",
    "0ICZi26jdaQ_28-5-rgb_front",
    "0vNfEYst_tQ_11-8-rgb_front",
    "13X0vEMNm7M_8-5-rgb_front",
    "14weIYQswlE_23-8-rgb_front",
    "1B56XMJ-j1Q_13-8-rgb_front",
    "1P0oKY4FNyI_0-8-rgb_front",
    "1dpRaxOTfZs_0-8-rgb_front",
    "1ei1kVTw23A_29-8-rgb_front",
    "1spCnuBmWYk_0-8-rgb_front",
    "2-vXO7MMLJc_0-5-rgb_front",
    "21PbS6wnHtY_0-5-rgb_front",
    "3tyfxL2wO-M_0-8-rgb_front",
    "BpYDl3AO4B8_0-1-rgb_front",
    "CH7AviIr0-0_14-8-rgb_front",
    "CJ8RyW9pzKU_6-8-rgb_front",
    "D0T7ho08Q3o_25-2-rgb_front",
    "Db5SUQvNsHc_18-1-rgb_front",
    "Eh697LCFjTw_0-3-rgb_front",
    "F-p1IdedNbg_23-8-rgb_front",
    "aUBQCNegrYc_13-1-rgb_front",
    "cvn7htBA8Xc_9-8-rgb_front",
    "czBrBQgZIuc_19-5-rgb_front",
    "dbSAB8F8GYc_11-9-rgb_front",
    "doMosV-zfCI_7-2-rgb_front",
    "dvBdWGLzayI_10-8-rgb_front",
    "eBrlZcccILg_26-3-rgb_front",
    "39FN42e41r0_17-1-rgb_front",
    "a4Nxq0QV_WA_9-3-rgb_front",
    "fzrJBu2qsM8_11-8-rgb_front",
    "g3Cc_1-V31U_12-3-rgb_front",
}

FRAME_KEYS_179 = [
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_shape",
    "smplx_expr",
]

DEFAULT_SOURCE_FPS_BY_DATASET = {
    "Neural-Sign-Actors": 24.0,  # This is the How2Sign source in this data bundle.
    "CSL-Daily-Fittings": 30.0,
    "PHOENIX": 25.0,
}

CANON = SMPLXMotionGPTCanonicalizer(include_betas=True)
LEG_JOINT_IDS = [1, 2, 4, 5, 7, 8, 10, 11]
LHAND_JOINT_IDS = list(range(22, 37))
RHAND_JOINT_IDS = list(range(37, 52))
BODY_KEEP_JOINT_IDS = [j for j in range(22) if j not in LEG_JOINT_IDS]
ROT6D_START = 5
CANON_BETAS = np.arange(317, 327, dtype=np.int64)


def joint_dims(joint_ids: list[int], per_joint: int) -> np.ndarray:
    out: list[int] = []
    for joint_id in joint_ids:
        start = joint_id * per_joint
        out.extend(range(start, start + per_joint))
    return np.asarray(out, dtype=np.int64)


ROT6D_BODY_KEEP = ROT6D_START + joint_dims(BODY_KEEP_JOINT_IDS, 6)
ROT6D_LEG = ROT6D_START + joint_dims(LEG_JOINT_IDS, 6)
ROT6D_LHAND = ROT6D_START + joint_dims(LHAND_JOINT_IDS, 6)
ROT6D_RHAND = ROT6D_START + joint_dims(RHAND_JOINT_IDS, 6)


def safe_name(value: Any, max_len: int = 180) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (out or "item")[:max_len]


def frame_sort_key(path: Path) -> tuple[Any, ...]:
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return tuple(int(x) for x in nums[-3:])
    return (path.name,)


def load_frame(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict frame PKL, got {type(obj)!r} at {path}")
    return obj


def frame_to_smplx182(frame: dict[str, Any]) -> np.ndarray:
    missing = [k for k in FRAME_KEYS_179 if k not in frame]
    if missing:
        raise KeyError(f"Missing SMPL-X keys: {missing}")
    pose179 = np.concatenate([np.asarray(frame[k], dtype=np.float32).reshape(-1) for k in FRAME_KEYS_179], axis=0)
    if pose179.shape[0] != 179:
        raise ValueError(f"Expected 179 pose dims from frame keys, got {pose179.shape}")
    cam = np.asarray(frame.get("cam_trans", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)[:3]
    out = np.zeros(182, dtype=np.float32)
    out[:179] = pose179
    out[179:182] = cam
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def sample_frame_paths(paths: list[Path], source_fps: float, target_fps: float) -> list[Path]:
    if len(paths) <= 1:
        return paths
    if source_fps <= 0 or target_fps <= 0 or abs(source_fps - target_fps) < 1e-8:
        return paths
    if source_fps <= target_fps:
        # SOKE-style preprocessing never upsamples missing frames.
        return paths
    count = int(target_fps * len(paths) / source_fps)
    count = max(1, min(len(paths), count))
    stride = float(len(paths)) / float(count)
    return [paths[int(math.floor(i * stride))] for i in range(count)]


def build_canonical_face_target(smplx182: np.ndarray) -> dict[str, np.ndarray]:
    smplx182 = np.asarray(smplx182, dtype=np.float32)
    pose156 = smplx182[:, :156]
    jaw3 = smplx182[:, 156:159]
    expr10 = smplx182[:, 169:179]
    betas10 = smplx182[:, 159:169]
    transl3 = smplx182[:, 179:182]

    full169 = np.concatenate([pose156, betas10, transl3], axis=-1).astype(np.float32)
    canon_feat, meta = CANON.encode(full169)
    meta_vec = np.asarray([meta.yaw0, meta.floor_y, meta.origin_x, meta.origin_z], dtype=np.float32)

    body84 = canon_feat[:, ROT6D_BODY_KEEP]
    lhand90 = canon_feat[:, ROT6D_LHAND]
    rhand90 = canon_feat[:, ROT6D_RHAND]
    body_face97 = np.concatenate([body84, jaw3, expr10], axis=-1).astype(np.float32)
    pred277 = np.concatenate([body_face97, lhand90, rhand90], axis=-1).astype(np.float32)

    copy63 = np.concatenate(
        [
            canon_feat[:, :5],
            canon_feat[:, ROT6D_LEG],
            canon_feat[:, CANON_BETAS],
        ],
        axis=-1,
    ).astype(np.float32)

    return {
        "pred": np.nan_to_num(pred277, nan=0.0, posinf=0.0, neginf=0.0),
        "copy": np.nan_to_num(copy63, nan=0.0, posinf=0.0, neginf=0.0),
        "meta": meta_vec,
    }


def scan_soke_data(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    csl_root = data_root / "CSL-Daily-Fittings"
    csl_pose_root = csl_root / "csl-daily_pose" / "csl-daily_pose"
    for split in ("train", "val", "test"):
        ann_path = csl_root / f"csl_clean.{split}"
        if ann_path.exists():
            with gzip.open(ann_path, "rb") as f:
                anns = pickle.load(f)
            for ann in anns:
                name = str(ann["name"])
                rows.append(
                    {
                        "dataset": "CSL-Daily-Fittings",
                        "source_alias": "csl",
                        "split": split,
                        "clip_id": name,
                        "pose_dir": str(csl_pose_root / name),
                        "frame_pkls": int(ann.get("num_frames", -1)),
                        "text": ann.get("text", ""),
                        "gloss": ann.get("gloss", ""),
                    }
                )
        elif csl_pose_root.exists():
            for pose_dir in sorted(csl_pose_root.iterdir()):
                if pose_dir.is_dir():
                    rows.append(
                        {
                            "dataset": "CSL-Daily-Fittings",
                            "source_alias": "csl",
                            "split": split,
                            "clip_id": pose_dir.name,
                            "pose_dir": str(pose_dir),
                            "frame_pkls": len(list(pose_dir.glob("*.pkl"))),
                        }
                    )

    h2s_root = data_root / "Neural-Sign-Actors"
    for split in ("train", "val", "test"):
        pose_root = h2s_root / f"{split}_poses" / "poses"
        if pose_root.exists():
            for pose_dir in sorted(pose_root.iterdir()):
                if pose_dir.is_dir():
                    rows.append(
                        {
                            "dataset": "Neural-Sign-Actors",
                            "source_alias": "how2sign",
                            "split": split,
                            "clip_id": pose_dir.name,
                            "pose_dir": str(pose_dir),
                            "frame_pkls": len(list(pose_dir.glob("*.pkl"))),
                        }
                    )

    phoenix_root = data_root / "phoenix_poses"
    for split in ("train", "dev", "test"):
        pose_root = phoenix_root / split
        if pose_root.exists():
            for pose_dir in sorted(pose_root.iterdir()):
                if pose_dir.is_dir():
                    rows.append(
                        {
                            "dataset": "PHOENIX",
                            "source_alias": "phoenix",
                            "split": split,
                            "clip_id": pose_dir.name,
                            "pose_dir": str(pose_dir),
                            "frame_pkls": len(list(pose_dir.glob("*.pkl"))),
                        }
                    )
    return pd.DataFrame(rows)


def remap_pose_dirs_to_data_root(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Make a prebuilt absolute-path index portable across local and Colab roots."""
    if "pose_dir" not in df.columns:
        return df

    data_root = Path(data_root)
    marker_names = ("CSL-Daily-Fittings", "Neural-Sign-Actors", "phoenix_poses")
    csl_pose_roots = [
        data_root / "CSL-Daily-Fittings" / "csl-daily_pose" / "csl-daily_pose",
        data_root / "CSL-Daily-Fittings" / "New folder" / "csl-daily_pose" / "csl-daily_pose",
        data_root / "CSL-Daily-Fittings" / "New folder" / "csl-daily_pose",
        data_root / "CSL-Daily-Fittings" / "csl-daily_pose",
    ]
    csl_rows = df[df.get("dataset", "") == "CSL-Daily-Fittings"] if "dataset" in df.columns else pd.DataFrame()
    if not csl_rows.empty and "clip_id" in csl_rows.columns:
        sample_clip = str(csl_rows.iloc[0]["clip_id"])
        for idx, root in enumerate(csl_pose_roots):
            if (root / sample_clip).exists():
                csl_pose_roots = [root] + csl_pose_roots[:idx] + csl_pose_roots[idx + 1 :]
                break

    def from_marker(path_text: str) -> Path | None:
        parts = Path(str(path_text)).parts
        for marker in marker_names:
            if marker in parts:
                idx = parts.index(marker)
                return data_root / Path(*parts[idx:])
        return None

    def fallback(row: pd.Series) -> Path:
        dataset = str(row.get("dataset", ""))
        split = str(row.get("split", ""))
        clip_id = str(row.get("clip_id", ""))
        if dataset == "CSL-Daily-Fittings":
            for root in csl_pose_roots:
                candidate = root / clip_id
                if candidate.exists():
                    return candidate
            return csl_pose_roots[0] / clip_id
        if dataset == "Neural-Sign-Actors":
            return data_root / "Neural-Sign-Actors" / f"{split}_poses" / "poses" / clip_id
        if dataset == "PHOENIX":
            return data_root / "phoenix_poses" / split / clip_id
        return Path(str(row.get("pose_dir", "")))

    out = df.copy()
    remapped: list[str] = []
    for _, row in out.iterrows():
        original = Path(str(row["pose_dir"]))
        if original.exists():
            remapped.append(str(original))
            continue
        fallback_candidate = fallback(row)
        if fallback_candidate.exists():
            remapped.append(str(fallback_candidate))
            continue
        marker_candidate = from_marker(str(original))
        remapped.append(str(marker_candidate if marker_candidate is not None else fallback_candidate))
    out["pose_dir"] = remapped
    return out


def load_index(data_root: Path, index_csv: Path | None) -> pd.DataFrame:
    if index_csv is not None and index_csv.exists():
        df = pd.read_csv(index_csv)
        if "source_alias" not in df.columns:
            df["source_alias"] = df["dataset"].map(
                {
                    "Neural-Sign-Actors": "how2sign",
                    "CSL-Daily-Fittings": "csl",
                    "PHOENIX": "phoenix",
                }
            ).fillna(df["dataset"])
        return remap_pose_dirs_to_data_root(df, data_root)
    return scan_soke_data(data_root)


def process_row(row: pd.Series, args: argparse.Namespace, source_fps_by_dataset: dict[str, float]) -> dict[str, Any]:
    dataset = str(row["dataset"])
    split = str(row["split"])
    clip_id = str(row["clip_id"])
    pose_dir = Path(row["pose_dir"])
    source_alias = str(row.get("source_alias", dataset))
    source_fps = float(source_fps_by_dataset.get(dataset, args.default_source_fps))

    rec: dict[str, Any] = {
        "dataset": dataset,
        "source_alias": source_alias,
        "split": split,
        "clip_id": clip_id,
        "pose_dir": str(pose_dir),
        "source_fps": source_fps,
        "target_fps": float(args.target_fps),
    }

    if not pose_dir.exists():
        rec.update(status="bad", reason="missing_pose_dir")
        return rec

    frame_files = sorted(pose_dir.glob("*.pkl"), key=frame_sort_key)
    rec["source_frames"] = int(len(frame_files))
    rec["source_duration_sec"] = float(len(frame_files) / max(source_fps, 1e-6))

    if dataset == "Neural-Sign-Actors" and clip_id in SOKE_BAD_HOW2SIGN_IDS:
        rec.update(status="bad", reason="soke_bad_how2sign_id")
        return rec
    if len(frame_files) < 4:
        rec.update(status="bad", reason="soke_too_short_lt4_source_frames")
        return rec
    if args.max_duration_sec > 0:
        apply_duration_drop = args.max_duration_all or dataset == "Neural-Sign-Actors"
        if apply_duration_drop and rec["source_duration_sec"] >= float(args.max_duration_sec):
            rec.update(status="bad", reason="soke_how2sign_duration_ge_max")
            return rec

    sampled_files = sample_frame_paths(frame_files, source_fps=source_fps, target_fps=float(args.target_fps))
    rec["target_frames"] = int(len(sampled_files))
    rec["target_duration_sec"] = float(len(sampled_files) / float(args.target_fps))
    if len(sampled_files) < int(args.window_size):
        rec.update(status="bad", reason="nb51_too_short_after_20fps")
        return rec

    cache_dir = (
        Path(args.out_root)
        / "cache_soke20"
        / safe_name(dataset)
        / safe_name(split)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / f"{safe_name(clip_id)}.npz"
    rec["cache_npz"] = str(cache_npz)

    if cache_npz.exists() and not args.force:
        rec.update(status="ok", reason="cached")
        return rec

    smplx182 = np.stack([frame_to_smplx182(load_frame(p)) for p in sampled_files], axis=0).astype(np.float32)
    can = build_canonical_face_target(smplx182)
    np.savez_compressed(cache_npz, pred=can["pred"], copy=can["copy"], meta=can["meta"])
    rec.update(status="ok", reason="built")
    return rec


def split_role(split: str) -> str:
    if split == "train":
        return "train"
    if split in {"val", "dev"}:
        return "val"
    if split == "test":
        return "test"
    return "other"


def parse_source_fps_json(value: str | None) -> dict[str, float]:
    out = deepcopy(DEFAULT_SOURCE_FPS_BY_DATASET)
    if value:
        extra = json.loads(value)
        for k, v in extra.items():
            out[str(k)] = float(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SOKE-style 20 fps caches for the NB51 tokenizer.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, default=None)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--default-source-fps", type=float, default=24.0)
    parser.add_argument("--source-fps-json", type=str, default=None)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--max-duration-all", action="store_true")
    parser.add_argument("--max-clips-per-split", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    source_fps_by_dataset = parse_source_fps_json(args.source_fps_json)
    df = load_index(args.data_root, args.index_csv)
    if df.empty:
        raise RuntimeError(f"No clips found under {args.data_root}")

    df = df[df.get("pose_exists", True).astype(bool)].copy() if "pose_exists" in df.columns else df.copy()
    df["role"] = df["split"].map(split_role)
    df = df[df["role"].isin(["train", "val", "test"])].reset_index(drop=True)

    if args.max_clips_per_split > 0:
        rng = np.random.default_rng(args.seed)
        chunks = []
        for _, group in df.groupby(["dataset", "split"], sort=True):
            n = min(int(args.max_clips_per_split), len(group))
            chunks.append(group.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
        df = pd.concat(chunks, ignore_index=True).sort_values(["dataset", "split", "clip_id"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    total = len(df)
    print(f"Processing {total} clips into {args.out_root}")
    print("source_fps_by_dataset:", source_fps_by_dataset)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        if idx == 1 or idx % int(args.progress_every) == 0 or idx == total:
            ok_so_far = sum(1 for r in rows if r.get("status") == "ok")
            print(f"  preprocess {idx}/{total} | ok={ok_so_far} | bad={len(rows)-ok_so_far}")
        try:
            rows.append(process_row(pd.Series(row._asdict()), args, source_fps_by_dataset))
        except Exception as exc:
            rows.append(
                {
                    "dataset": getattr(row, "dataset", ""),
                    "split": getattr(row, "split", ""),
                    "clip_id": getattr(row, "clip_id", ""),
                    "pose_dir": getattr(row, "pose_dir", ""),
                    "status": "bad",
                    "reason": f"exception:{type(exc).__name__}:{exc}",
                }
            )

    manifest = pd.DataFrame(rows)
    manifest["role"] = manifest["split"].map(split_role)
    all_csv = args.out_root / "manifest_all.csv"
    bad_csv = args.out_root / "bad_manifest.csv"
    manifest.to_csv(all_csv, index=False)
    manifest[manifest["status"] != "ok"].to_csv(bad_csv, index=False)

    ok = manifest[manifest["status"] == "ok"].copy()
    for role in ("train", "val", "test"):
        ok[ok["role"] == role].to_csv(args.out_root / f"{role}_manifest.csv", index=False)

    summary = (
        manifest.groupby(["dataset", "split", "status", "reason"], dropna=False)
        .size()
        .reset_index(name="clips")
        .sort_values(["dataset", "split", "status", "reason"])
    )
    summary_csv = args.out_root / "preprocess_summary.csv"
    summary.to_csv(summary_csv, index=False)

    config = {
        "target_fps": args.target_fps,
        "source_fps_by_dataset": source_fps_by_dataset,
        "window_size": args.window_size,
        "max_duration_sec": args.max_duration_sec,
        "max_duration_all": bool(args.max_duration_all),
        "soke_bad_how2sign_ids": sorted(SOKE_BAD_HOW2SIGN_IDS),
    }
    (args.out_root / "preprocess_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Wrote:", all_csv)
    print("Wrote:", bad_csv)
    print("Wrote:", summary_csv)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
