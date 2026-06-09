from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd

from soke_gemma_data import SokeGemmaCausalDataset, load_codecs, load_instructions, read_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent


def motion_only_text(text: str) -> str:
    prefixes = ("<motion_id_", "<hand_id_", "<rhand_id_")
    return " ".join(tok for tok in str(text).split() if tok.startswith(prefixes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SOKE-Gemma JSONL rows and rendered instruction samples.")
    parser.add_argument("--code-root", type=Path, default=BUNDLE_ROOT / "outputs" / "soke_motion_codes")
    parser.add_argument("--instructions-root", type=Path, default=BUNDLE_ROOT / "instructions")
    parser.add_argument("--split", default="train")
    parser.add_argument("--stage", choices=["lm_pretrain", "lm_instruct"], default="lm_pretrain")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--tokenizer-error-policy", choices=["skip", "raise"], default="skip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_path = args.code_root / f"{args.split}_soke_motion_codes.jsonl"
    if not code_path.exists():
        raise FileNotFoundError(f"Missing code JSONL: {code_path}")
    rows = read_jsonl(code_path, limit=args.row_limit if args.row_limit > 0 else None)
    codecs = load_codecs(args.code_root)
    inst_path = args.instructions_root / ("template_pretrain.json" if args.stage == "lm_pretrain" else "template_instructions.json")
    tasks = load_instructions(inst_path)
    ds = SokeGemmaCausalDataset(rows, tasks, codecs, random_drop=False, seed=args.seed)

    print(json.dumps({
        "code_root": str(args.code_root),
        "split": args.split,
        "stage": args.stage,
        "physical_rows": len(rows),
        "row_limit": int(args.row_limit),
        "instruction_tasks": len(tasks),
        "logical_rows": len(ds),
        "codecs": codecs.to_json_dict(),
    }, indent=2, ensure_ascii=False))

    table = pd.DataFrame(rows)
    if not table.empty:
        summary = table.groupby(["dataset", "source_alias"]).agg(
            clips=("clip_id", "count"),
            code_len_mean=("code_len", "mean"),
            code_len_median=("code_len", "median"),
            frames_mean=("num_frames_soke", "mean"),
        ).reset_index()
        print("\nDataset summary:")
        print(summary.to_string(index=False))

    rng = random.Random(args.seed)
    picks = [rng.randrange(len(ds)) for _ in range(min(args.n, len(ds)))]
    samples = [ds[i] for i in picks]
    print("\nRandom rendered samples:")
    for i, sample in zip(picks, samples):
        print("=" * 100)
        print(json.dumps({
            "logical_idx": i,
            "clip_key": sample["clip_key"],
            "dataset": sample["dataset"],
            "source_alias": sample["source_alias"],
            "code_len": sample["code_len"],
            "target_token_count": sample["target_token_count"],
            "caption": sample["caption"],
            "prompt": sample["prompt"],
            "target_prefix": " ".join(sample["target"].split()[:24]),
        }, indent=2, ensure_ascii=False))

    if args.tokenizer:
        try:
            from transformers import AddedToken, AutoTokenizer

            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            kwargs = {"token": token} if token else {}
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **kwargs)
            tokenizer.add_tokens([
                AddedToken(tok, lstrip=True, rstrip=False, normalized=False, special=False)
                for tok in codecs.added_tokens()
            ])
            print("\nTokenizer audit:")
            for sample in samples[: min(5, len(samples))]:
                motion_text = motion_only_text(sample["target"])
                pieces = motion_text.split()
                ids = tokenizer(motion_text, add_special_tokens=False).input_ids
                print(json.dumps({
                    "clip_key": sample["clip_key"],
                    "whitespace_tokens": len(pieces),
                    "tokenizer_ids": len(ids),
                    "ok": len(pieces) == len(ids),
                }, ensure_ascii=False))
        except Exception as exc:
            if args.tokenizer_error_policy == "raise":
                raise
            print("\nTokenizer audit skipped:")
            print(f"  tokenizer={args.tokenizer}")
            print(f"  reason={type(exc).__name__}: {exc}")
            print("  Set HF_TOKEN/HUGGING_FACE_HUB_TOKEN after accepting the Gemma license to enable this audit.")


if __name__ == "__main__":
    main()
