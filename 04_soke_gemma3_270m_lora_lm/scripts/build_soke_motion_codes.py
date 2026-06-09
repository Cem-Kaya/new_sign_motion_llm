from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
NEW_DATA_ROOT = BUNDLE_ROOT.parent
EXP_ROOT = NEW_DATA_ROOT.parent
SIBLING_02_SCRIPTS = NEW_DATA_ROOT / "02_soke20fps_VQVAE_tokenizer" / "scripts"
if str(SIBLING_02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SIBLING_02_SCRIPTS))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from eval_official_soke_vqvae_newdata import (  # noqa: E402
    load_clip_features,
    load_official_model,
    local_mean_std_133,
    official_mean_std_133,
)
from soke_gemma_data import (  # noqa: E402
    SokePartCodecs,
    flattened_target_text,
    soke_concat_tokens,
    write_codecs,
)


DEFAULT_MANIFEST_ROOT = NEW_DATA_ROOT / "02_soke20fps_VQVAE_tokenizer" / "outputs" / "preprocess_soke20"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "outputs" / "soke_motion_codes"
DEFAULT_CHECKPOINT = EXP_ROOT / "ref" / "SOKE" / "experiments" / "mgpt" / "vae" / "checkpoints" / "tokenizer.ckpt"


def load_pickle_maybe_gzip(path: Path) -> Any:
    if path.suffix in {".gz", ".gzip"}:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except OSError:
        with path.open("rb") as f:
            return pickle.load(f)


def key_variants(clip_id: str) -> set[str]:
    value = str(clip_id).strip()
    variants = {value}
    if value.endswith(".pkl") or value.endswith(".mp4"):
        variants.add(Path(value).stem)
    variants.add(Path(value).stem)
    if "/" in value:
        variants.add(Path(value).name)
        variants.add(Path(value).stem)
    return {v for v in variants if v}


def add_text(text_by_key: dict[tuple[str, str], str], source_alias: str, clip_id: str, text: Any) -> None:
    if text is None:
        return
    value = str(text).strip()
    if not value or value.lower() == "nan":
        return
    for key in key_variants(str(clip_id)):
        text_by_key[(source_alias, key)] = value


def load_text_csv(path: Path, text_by_key: dict[tuple[str, str], str]) -> int:
    df = pd.read_csv(path, sep=None, engine="python")
    id_cols = ["clip_id", "name", "sentence_name", "SENTENCE_NAME"]
    text_cols = ["text", "SENTENCE", "sentence", "translation", "gloss"]
    id_col = next((c for c in id_cols if c in df.columns), None)
    text_col = next((c for c in text_cols if c in df.columns), None)
    if id_col is None or text_col is None:
        return 0
    source_col = "source_alias" if "source_alias" in df.columns else "src" if "src" in df.columns else None
    n = 0
    for row in df.itertuples(index=False):
        clip_id = str(getattr(row, id_col))
        source_alias = str(getattr(row, source_col)) if source_col else ""
        text = getattr(row, text_col)
        if source_alias:
            add_text(text_by_key, source_alias, clip_id, text)
        else:
            for alias in ("how2sign", "csl", "phoenix"):
                add_text(text_by_key, alias, clip_id, text)
        n += 1
    return n


def load_annotation_pickle(path: Path, source_alias: str, text_by_key: dict[tuple[str, str], str]) -> int:
    data = load_pickle_maybe_gzip(path)
    if not isinstance(data, list):
        return 0
    n = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("clip_id") or item.get("SENTENCE_NAME")
        text = item.get("text") or item.get("translation") or item.get("gloss")
        if name is not None:
            add_text(text_by_key, source_alias, str(name), text)
            n += 1
    return n


def build_text_index(extra_text_indexes: list[Path]) -> tuple[dict[tuple[str, str], str], list[dict[str, Any]]]:
    text_by_key: dict[tuple[str, str], str] = {}
    sources: list[dict[str, Any]] = []

    for path in [
        EXP_ROOT / "artifacts" / "59_tokenizer_soke_nb51_nb57_metric_table" / "pkl_to_rgb_front_index.csv",
        EXP_ROOT / "artifacts" / "17_how2sign_rgb_smplx_triptych" / "pkl_to_rgb_front_index.csv",
    ]:
        if path.exists():
            before = len(text_by_key)
            n = load_text_csv(path, text_by_key)
            sources.append({"path": str(path), "kind": "how2sign_csv", "rows_seen": n, "new_keys": len(text_by_key) - before})

    how2sign_root = EXP_ROOT / "DATA" / "how2sign_original" / "How2Sign" / "sentence_level"
    for split in ("train", "val", "test"):
        for rel in [
            Path(split) / "text" / "en" / "raw_text" / f"how2sign_{split}.csv",
            Path(split) / "text" / "en" / "raw_text" / "re_aligned" / f"how2sign_realigned_{split}.csv",
        ]:
            path = how2sign_root / rel
            if path.exists():
                before = len(text_by_key)
                n = load_text_csv(path, text_by_key)
                sources.append({"path": str(path), "kind": "how2sign_raw_text_csv", "rows_seen": n, "new_keys": len(text_by_key) - before})

    csl_root = EXP_ROOT / "DATA" / "SOKE_DATA" / "CSL-Daily-Fittings"
    for split in ("train", "val", "test"):
        path = csl_root / f"csl_clean.{split}"
        if path.exists():
            before = len(text_by_key)
            n = load_annotation_pickle(path, "csl", text_by_key)
            sources.append({"path": str(path), "kind": "csl_pickle", "rows_seen": n, "new_keys": len(text_by_key) - before})

    for root in [
        EXP_ROOT / "DATA" / "SOKE_DATA" / "PHOENIX",
        EXP_ROOT / "DATA" / "SOKE_DATA" / "Phoenix_2014T",
        EXP_ROOT / "DATA" / "SOKE_DATA" / "phoenix_poses",
    ]:
        for name in ("phoenix14t.train", "phoenix14t.dev", "phoenix14t.val", "phoenix14t.test"):
            path = root / name
            if path.exists():
                before = len(text_by_key)
                n = load_annotation_pickle(path, "phoenix", text_by_key)
                sources.append({"path": str(path), "kind": "phoenix_pickle", "rows_seen": n, "new_keys": len(text_by_key) - before})

    legacy_code_root = EXP_ROOT / "artifacts" / "47_soke20_colab_results_analysis" / "colab_copy" / "motion_codes"
    for split in ("train", "val", "test"):
        path = legacy_code_root / f"{split}_soke_motion_codes.jsonl"
        if path.exists():
            before = len(text_by_key)
            n = load_text_json(path, text_by_key)
            sources.append({"path": str(path), "kind": "legacy_soke_code_jsonl_text", "rows_seen": n, "new_keys": len(text_by_key) - before})

    for path in extra_text_indexes:
        before = len(text_by_key)
        if path.suffix.lower() == ".csv":
            n = load_text_csv(path, text_by_key)
            kind = "extra_csv"
        elif path.suffix.lower() in {".jsonl", ".json"}:
            n = load_text_json(path, text_by_key)
            kind = "extra_json"
        else:
            n = load_annotation_pickle(path, "", text_by_key)
            kind = "extra_pickle"
        sources.append({"path": str(path), "kind": kind, "rows_seen": n, "new_keys": len(text_by_key) - before})

    return text_by_key, sources


def load_text_json(path: Path, text_by_key: dict[tuple[str, str], str]) -> int:
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else list(payload.values()) if isinstance(payload, dict) else []
    n = 0
    for item in rows:
        if not isinstance(item, dict):
            continue
        clip_id = item.get("clip_id") or item.get("name") or item.get("sentence_name") or item.get("SENTENCE_NAME")
        text = item.get("text") or item.get("SENTENCE") or item.get("sentence") or item.get("translation") or item.get("gloss")
        source_alias = item.get("source_alias") or item.get("src")
        if clip_id is None:
            continue
        if source_alias:
            add_text(text_by_key, str(source_alias), str(clip_id), text)
        else:
            for alias in ("how2sign", "csl", "phoenix"):
                add_text(text_by_key, alias, str(clip_id), text)
        n += 1
    return n


def resolve_text(text_by_key: dict[tuple[str, str], str], source_alias: str, clip_id: str) -> str | None:
    source_alias = str(source_alias)
    clip_id = str(clip_id)
    candidates = [
        (source_alias, clip_id),
        ("how2sign" if source_alias == "neural_sign_actors" else source_alias, clip_id),
        ("", clip_id),
    ]
    for key in candidates:
        value = text_by_key.get(key)
        if value:
            return value
    return None


def adjust_frames_like_soke(feat: np.ndarray, min_motion_len: int = 40, max_motion_len: int = 400, unit_len: int = 4) -> np.ndarray:
    x = np.asarray(feat, dtype=np.float32)
    n = int(x.shape[0])
    if n <= 0:
        return x
    if n < min_motion_len:
        idx = np.linspace(0, n - 1, num=min_motion_len, dtype=int)
        return x[idx].astype(np.float32)
    if n > max_motion_len:
        idx = np.linspace(0, n - 1, num=max_motion_len, dtype=int)
        return x[idx].astype(np.float32)
    keep = (n // unit_len) * unit_len
    keep = max(unit_len, keep)
    start = (n - keep) // 2
    return x[start : start + keep].astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SOKE-style three-stream motion-code JSONL for Gemma causal LM training.")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-source", choices=["official", "local"], default="official")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--text-index", type=Path, action="append", default=[])
    parser.add_argument("--max-clips-per-split", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"device": str(device), "checkpoint": str(args.checkpoint), "norm_source": args.norm_source})

    text_by_key, text_sources = build_text_index(args.text_index)
    (args.output_root / "text_sources.json").write_text(json.dumps(text_sources, indent=2, ensure_ascii=False), encoding="utf-8")
    print({"text_keys": len(text_by_key), "text_sources": text_sources})

    model, model_meta = load_official_model(args.checkpoint, device)
    if args.norm_source == "official":
        mean, std, norm_source = official_mean_std_133()
    else:
        mean, std, norm_source = local_mean_std_133()
    codecs = SokePartCodecs(
        body_size=int(model_meta["model_cfg"]["code_num_body"]),
        lhand_size=int(model_meta["model_cfg"]["code_num_lhand"]),
        rhand_size=int(model_meta["model_cfg"]["code_num_rhand"]),
    )
    write_codecs(args.output_root, codecs)

    summary_rows: list[dict[str, Any]] = []
    for split in args.splits:
        manifest = args.manifest_root / f"{split}_manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing split manifest: {manifest}")
        df = pd.read_csv(manifest)
        if "status" in df.columns:
            df = df[df["status"].astype(str) == "ok"].reset_index(drop=True)
        if args.max_clips_per_split > 0:
            df = df.sample(n=min(args.max_clips_per_split, len(df)), random_state=args.seed).sort_values(["dataset", "clip_id"]).reset_index(drop=True)

        out_path = args.output_root / f"{split}_soke_motion_codes.jsonl"
        bad_path = args.output_root / f"{split}_skipped_rows.jsonl"
        if args.force:
            out_path.unlink(missing_ok=True)
            bad_path.unlink(missing_ok=True)

        done: set[str] = set()
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done.add(str(json.loads(line).get("clip_key")))
            print(f"Resuming {split}: {len(done)} completed rows from {out_path}")

        n_seen = n_written = n_missing_text = n_failed = 0
        started = time.time()
        with out_path.open("a", encoding="utf-8") as out_f, bad_path.open("a", encoding="utf-8") as bad_f:
            for idx, row_tuple in enumerate(df.itertuples(index=False), start=1):
                row = pd.Series(row_tuple._asdict())
                dataset = str(row["dataset"])
                source_alias = str(row.get("source_alias", dataset))
                clip_id = str(row["clip_id"])
                clip_key = f"{dataset}/{split}/{clip_id}"
                if clip_key in done:
                    continue
                n_seen += 1
                text = resolve_text(text_by_key, source_alias, clip_id)
                if not text:
                    n_missing_text += 1
                    bad_f.write(json.dumps({"clip_key": clip_key, "reason": "missing_text", "dataset": dataset, "source_alias": source_alias, "clip_id": clip_id}, ensure_ascii=False) + "\n")
                    continue
                try:
                    feat133, _, _, _ = load_clip_features(row)
                    feat133_soke = adjust_frames_like_soke(feat133)
                    feat_n = ((feat133_soke - mean[None, :]) / std[None, :]).astype(np.float32)
                    xb = torch.from_numpy(feat_n[None]).to(device)
                    with torch.no_grad():
                        parts = model.encode_parts(xb)
                    body = parts["body"][0].detach().cpu().numpy().astype(int).tolist()
                    lhand = parts["lhand"][0].detach().cpu().numpy().astype(int).tolist()
                    rhand = parts["rhand"][0].detach().cpu().numpy().astype(int).tolist()
                    code_len = min(len(body), len(lhand), len(rhand))
                    rec = {
                        "clip_key": clip_key,
                        "dataset": dataset,
                        "source_alias": source_alias,
                        "split": split,
                        "clip_id": clip_id,
                        "text": text,
                        "pose_dir": str(row["pose_dir"]),
                        "source_fps": float(row.get("source_fps", np.nan)),
                        "target_fps": float(row.get("target_fps", 20.0)),
                        "num_frames_source": int(feat133.shape[0]),
                        "num_frames_soke": int(feat133_soke.shape[0]),
                        "code_len": int(code_len),
                        "body_ids": body[:code_len],
                        "lhand_ids": lhand[:code_len],
                        "rhand_ids": rhand[:code_len],
                        "body_text_soke_concat": soke_concat_tokens(codecs, "body", body[:code_len]),
                        "lhand_text_soke_concat": soke_concat_tokens(codecs, "lhand", lhand[:code_len]),
                        "rhand_text_soke_concat": soke_concat_tokens(codecs, "rhand", rhand[:code_len]),
                        "flat_triplet_text": flattened_target_text(codecs, body[:code_len], lhand[:code_len], rhand[:code_len]),
                        "tokenizer_checkpoint": str(args.checkpoint),
                        "norm_source": args.norm_source,
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1
                except Exception as exc:
                    n_failed += 1
                    bad_f.write(json.dumps({"clip_key": clip_key, "reason": "exception", "error": repr(exc), "dataset": dataset, "source_alias": source_alias, "clip_id": clip_id}, ensure_ascii=False) + "\n")
                if idx == 1 or idx % int(args.progress_every) == 0 or idx == len(df):
                    elapsed = (time.time() - started) / 60.0
                    print(
                        f"{split} {idx}/{len(df)} written={n_written} missing_text={n_missing_text} failed={n_failed} elapsed={elapsed:.1f}m",
                        flush=True,
                    )

        summary_rows.append(
            {
                "split": split,
                "manifest_rows": int(len(df)),
                "rows_seen_this_run": int(n_seen),
                "rows_written_this_run": int(n_written),
                "missing_text_this_run": int(n_missing_text),
                "failed_this_run": int(n_failed),
                "output_jsonl": str(out_path),
                "skipped_jsonl": str(bad_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_root / "build_summary.csv", index=False)
    meta = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_meta": model_meta,
        "norm_source": norm_source,
        "codecs": codecs.to_json_dict(),
        "text_sources": text_sources,
        "summary": summary_rows,
    }
    (args.output_root / "build_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
