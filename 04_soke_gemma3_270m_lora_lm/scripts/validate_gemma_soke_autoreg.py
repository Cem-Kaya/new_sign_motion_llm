from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
NEW_DATA_ROOT = BUNDLE_ROOT.parent
EXP_ROOT = NEW_DATA_ROOT.parent
SIBLING_02_SCRIPTS = NEW_DATA_ROOT / "02_soke20fps_VQVAE_tokenizer" / "scripts"

for path in [SCRIPT_DIR, SIBLING_02_SCRIPTS, EXP_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_nb51_soke20_metrics import (  # noqa: E402
    SMPLXJ22Decoder,
    dtw_last_cell,
    mpjpe_cost_matrix_blocked_torch,
    pa_cost_matrix_blocked_torch,
)
from eval_official_soke_vqvae_newdata import (  # noqa: E402
    load_clip_features,
    load_official_model,
    local_mean_std_133,
    official_mean_std_133,
    soke133_to_smplx169,
)
from soke_gemma_data import (  # noqa: E402
    SokePartCodecs,
    crop_code_ids_like_soke,
    flatten_triplet_tokens,
    load_codecs,
    read_jsonl,
    render_template,
    soke_motion_placeholder_variants,
)
from train_gemma_soke_lora import parse_generated_triplets  # noqa: E402


DEFAULT_DRIVE_ADAPTER = Path(
    "/mnt/y/Drive/My_Drive/folder/COLAB/Tokenizer/04_soke_gemma3_270m_lora_lm/"
    "outputs/runs/gemma3_270m_lora_soke_flat_lm/lm_instruct/last_adapter"
)
DEFAULT_OFFICIAL_TOKENIZER = EXP_ROOT / "ref" / "SOKE" / "experiments" / "mgpt" / "vae" / "checkpoints" / "tokenizer.ckpt"

CJK_RE = re.compile(r"[\u3400-\u9fff]")
WORD_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\sA-Za-z0-9_\u3400-\u9fff]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_named_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for group_name, group in payload.items():
        if not isinstance(group, dict):
            continue
        for task_name, task in group.items():
            if not isinstance(task, dict):
                continue
            inputs = [str(x) for x in task.get("input", [])]
            outputs = [str(x) for x in task.get("output", [])]
            if not inputs or not outputs:
                continue
            tasks.append(
                {
                    "group": str(group_name),
                    "name": str(task_name),
                    "input": inputs,
                    "output": outputs,
                    "input_template": inputs[0],
                    "output_template": outputs[0],
                }
            )
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def is_text_to_motion_task(task: dict[str, Any]) -> bool:
    outputs = task.get("output", [])
    return task.get("group") == "Text-to-Motion" and any("<Motion_Placeholder>" in str(x) for x in outputs)


def is_caption_task(task: dict[str, Any]) -> bool:
    outputs = task.get("output", [])
    return any("<Caption_Placeholder>" in str(x) for x in outputs)


def tokenize_text(text: str) -> list[str]:
    value = str(text).strip().lower()
    return [tok for tok in WORD_RE.findall(value) if tok.strip()]


def ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1))


def corpus_bleu(predictions: Sequence[str], references: Sequence[str], max_order: int) -> float:
    matches_by_order = [0] * int(max_order)
    possible_by_order = [0] * int(max_order)
    pred_len = 0
    ref_len = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenize_text(pred)
        ref_tokens = tokenize_text(ref)
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)
        for order in range(1, int(max_order) + 1):
            pred_ng = ngrams(pred_tokens, order)
            ref_ng = ngrams(ref_tokens, order)
            overlap = pred_ng & ref_ng
            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += max(len(pred_tokens) - order + 1, 0)
    if pred_len == 0:
        return 0.0
    precisions = []
    for i in range(int(max_order)):
        if possible_by_order[i] == 0:
            precisions.append(0.0)
        elif matches_by_order[i] == 0:
            precisions.append(1.0 / (possible_by_order[i] * 2.0))
        else:
            precisions.append(matches_by_order[i] / possible_by_order[i])
    if min(precisions) <= 0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / int(max_order))
    bp = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1))
    return float(100.0 * bp * geo_mean)


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for tok_a in a:
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (len(b) + 1)
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_tokens = tokenize_text(pred)
    ref_tokens = tokenize_text(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    precision = lcs / max(len(pred_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return float(100.0 * (2.0 * precision * recall) / (precision + recall))


def code_text(ids: Sequence[int]) -> str:
    return " ".join(str(int(x)) for x in ids)


def flat_triplet_code_text(body: Sequence[int], lhand: Sequence[int], rhand: Sequence[int]) -> str:
    n = min(len(body), len(lhand), len(rhand))
    tokens: list[str] = []
    for b, lh, rh in zip(body[:n], lhand[:n], rhand[:n]):
        tokens.extend([f"body_{int(b)}", f"lhand_{int(lh)}", f"rhand_{int(rh)}"])
    return " ".join(tokens)


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def motion_token_metric_record(
    target_body: Sequence[int],
    target_lhand: Sequence[int],
    target_rhand: Sequence[int],
    pred_body: Sequence[int],
    pred_lhand: Sequence[int],
    pred_rhand: Sequence[int],
) -> dict[str, float | str]:
    parts = {
        "body": (target_body, pred_body),
        "lhand": (target_lhand, pred_lhand),
        "rhand": (target_rhand, pred_rhand),
    }
    rec: dict[str, float | str] = {}
    for part, (ref_ids, pred_ids) in parts.items():
        ref_text = code_text(ref_ids)
        pred_text = code_text(pred_ids)
        rec[f"{part}_reference_code_text"] = ref_text
        rec[f"{part}_generated_code_text"] = pred_text
        for order in range(1, 5):
            rec[f"{part}_bleu_{order}_higher_better"] = corpus_bleu([pred_text], [ref_text], order)
        rec[f"{part}_rouge_l_higher_better"] = rouge_l_f1(pred_text, ref_text)

    ref_flat = flat_triplet_code_text(target_body, target_lhand, target_rhand)
    pred_flat = flat_triplet_code_text(pred_body, pred_lhand, pred_rhand)
    rec["combined_flat_reference_code_text"] = ref_flat
    rec["combined_flat_generated_code_text"] = pred_flat
    for order in range(1, 5):
        stream_vals = [float(rec[f"{part}_bleu_{order}_higher_better"]) for part in parts]
        rec[f"combined_stream_avg_bleu_{order}_higher_better"] = safe_mean(stream_vals)
        rec[f"combined_flat_bleu_{order}_higher_better"] = corpus_bleu([pred_flat], [ref_flat], order)
    rec["combined_stream_avg_rouge_l_higher_better"] = safe_mean(
        [float(rec[f"{part}_rouge_l_higher_better"]) for part in parts]
    )
    rec["combined_flat_rouge_l_higher_better"] = rouge_l_f1(pred_flat, ref_flat)
    return rec


def clean_generated_caption(text: str, eos_token: str | None = None) -> str:
    value = str(text)
    if eos_token:
        value = value.split(eos_token, 1)[0]
    for marker in ["<end_of_turn>", "</s>", "<eos>"]:
        value = value.replace(marker, "")
    value = re.sub(r"<(?:motion|hand|rhand)_id_\d+>", " ", value)
    lines = [line.strip() for line in value.strip().splitlines() if line.strip()]
    if not lines:
        return value.strip()
    return lines[0].strip()


def soke_adjust_indices(num_frames: int, min_motion_len: int = 40, max_motion_len: int = 400, unit_len: int = 4) -> np.ndarray:
    n = int(num_frames)
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    if n < min_motion_len:
        return np.linspace(0, n - 1, num=min_motion_len, dtype=np.int64)
    if n > max_motion_len:
        return np.linspace(0, n - 1, num=max_motion_len, dtype=np.int64)
    keep = max(int(unit_len), (n // int(unit_len)) * int(unit_len))
    start = (n - keep) // 2
    return np.arange(start, start + keep, dtype=np.int64)


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float32)
    target = int(target_len)
    if arr.shape[0] == target:
        return arr
    if arr.shape[0] <= 0:
        raise ValueError("Cannot resample an empty sequence")
    if target <= 1:
        return arr[:1].copy()
    src_x = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target, dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((target, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst_x, src_x, flat[:, col]).astype(np.float32)
    return out.reshape((target, *arr.shape[1:])).astype(np.float32)


@torch.no_grad()
def decode_ids_to_feat133(
    tokenizer_model: torch.nn.Module,
    body_ids: Sequence[int],
    lhand_ids: Sequence[int],
    rhand_ids: Sequence[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    n = min(len(body_ids), len(lhand_ids), len(rhand_ids))
    if n <= 0:
        body_ids = lhand_ids = rhand_ids = [0]
    else:
        body_ids = [int(x) for x in body_ids[:n]]
        lhand_ids = [int(x) for x in lhand_ids[:n]]
        rhand_ids = [int(x) for x in rhand_ids[:n]]
    body_t = torch.tensor([body_ids], dtype=torch.long, device=device)
    lhand_t = torch.tensor([lhand_ids], dtype=torch.long, device=device)
    rhand_t = torch.tensor([rhand_ids], dtype=torch.long, device=device)
    pred_norm = tokenizer_model.decode_parts(body_t, lhand_t, rhand_t)[0].detach().float().cpu().numpy()
    return (pred_norm * std[None, :] + mean[None, :]).astype(np.float32)


def load_adapter_model(args: argparse.Namespace, codecs: SokePartCodecs) -> tuple[Any, torch.nn.Module]:
    from peft import PeftModel
    from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    auth_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=bool(args.trust_remote_code), **auth_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(
        [
            AddedToken(tok, lstrip=True, rstrip=False, normalized=False, special=False)
            for tok in codecs.added_tokens()
        ]
    )
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        **auth_kwargs,
    }
    if args.attn_implementation and args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    base.resize_token_embeddings(len(tokenizer))
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    return tokenizer, model


def render_eval_prompt(row: dict[str, Any], task: dict[str, Any], codecs: SokePartCodecs) -> dict[str, Any]:
    body, lhand, rhand = crop_code_ids_like_soke(row["body_ids"], row["lhand_ids"], row["rhand_ids"], random_drop=False)
    flat_tokens = flatten_triplet_tokens(codecs, body, lhand, rhand)
    motion_text = " ".join(flat_tokens)
    variants = soke_motion_placeholder_variants(codecs, flat_tokens)
    fps = float(row.get("target_fps", 20.0) or 20.0)
    code_len = len(body)
    prompt = render_template(
        str(task["input_template"]),
        caption=str(row["text"]),
        motion_text=motion_text,
        num_frames=code_len,
        fps=fps,
        **variants,
    )
    target = render_template(
        str(task["output_template"]),
        caption=str(row["text"]),
        motion_text=motion_text,
        num_frames=code_len,
        fps=fps,
        **variants,
    )
    return {
        "prompt": prompt.rstrip() + "\n",
        "target": target.strip(),
        "target_body": body,
        "target_lhand": lhand,
        "target_rhand": rhand,
        "target_triplets": min(len(body), len(lhand), len(rhand)),
    }


@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0, input_ids.shape[1] :], skip_special_tokens=False)


def compute_motion_metrics(
    row: dict[str, Any],
    target_body: Sequence[int],
    target_lhand: Sequence[int],
    target_rhand: Sequence[int],
    pred_body: Sequence[int],
    pred_lhand: Sequence[int],
    pred_rhand: Sequence[int],
    tokenizer_model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    smplx_decoder: SMPLXJ22Decoder,
    device: torch.device,
    frame_batch: int,
    raw_block_rows: int,
    pa_block_rows: int,
) -> dict[str, float]:
    feat133, aux36, betas10, transl3 = load_clip_features(pd.Series(row))
    idx = soke_adjust_indices(feat133.shape[0])
    feat133 = feat133[idx]
    aux36 = aux36[idx]
    betas10 = betas10[idx]
    transl3 = transl3[idx]

    target_frames = max(1, int(min(len(target_body), len(target_lhand), len(target_rhand))) * 4)
    gt_feat133 = resample_sequence(feat133, target_frames)
    gt_aux36 = resample_sequence(aux36, target_frames)
    gt_betas10 = resample_sequence(betas10, target_frames)
    gt_transl3 = resample_sequence(transl3, target_frames)
    gt169 = soke133_to_smplx169(gt_feat133, gt_aux36, gt_betas10, gt_transl3)

    pred_feat133 = decode_ids_to_feat133(tokenizer_model, pred_body, pred_lhand, pred_rhand, mean, std, device)
    pred_frames = int(pred_feat133.shape[0])
    pred_aux36 = resample_sequence(gt_aux36, pred_frames)
    pred_betas10 = resample_sequence(gt_betas10, pred_frames)
    pred_transl3 = resample_sequence(gt_transl3, pred_frames)
    pred169 = soke133_to_smplx169(pred_feat133, pred_aux36, pred_betas10, pred_transl3)

    gt_j = smplx_decoder(gt169, frame_batch=frame_batch)
    pred_j = smplx_decoder(pred169, frame_batch=frame_batch)
    pred_j_same_len = resample_sequence(pred_j, gt_j.shape[0])
    raw_costs = mpjpe_cost_matrix_blocked_torch(pred_j, gt_j, device=device, block_rows=raw_block_rows)
    pa_same_len = pa_cost_matrix_blocked_torch(pred_j_same_len, gt_j, device=device, block_rows=pa_block_rows)

    return {
        "target_frames": float(gt_j.shape[0]),
        "generated_frames": float(pred_j.shape[0]),
        "length_ratio_closer_to_1": float(pred_j.shape[0] / max(gt_j.shape[0], 1)),
        "mpjpe_mm_lower_better": float(np.mean(np.linalg.norm(pred_j_same_len - gt_j, axis=-1)) * 1000.0),
        "pa_mpjpe_mm_lower_better": float(np.diag(pa_same_len).mean() * 1000.0),
        "dtw_mpjpe_mm_lower_better": float(dtw_last_cell(raw_costs.astype(np.float64)) * 1000.0),
    }


def corpus_rouge_l(preds: Sequence[str], refs: Sequence[str]) -> float:
    return safe_mean([rouge_l_f1(pred, ref) for pred, ref in zip(preds, refs)])


def code_stream_scores(group: pd.DataFrame, stream: str) -> dict[str, float]:
    if stream == "combined_stream_avg":
        part_scores = [code_stream_scores(group, part) for part in ["body", "lhand", "rhand"]]
        return {
            **{
                f"bleu_{order}_higher_better": safe_mean(
                    [scores[f"bleu_{order}_higher_better"] for scores in part_scores]
                )
                for order in range(1, 5)
            },
            "rouge_l_higher_better": safe_mean([scores["rouge_l_higher_better"] for scores in part_scores]),
        }
    if stream == "combined_flat":
        pred_col = "combined_flat_generated_code_text"
        ref_col = "combined_flat_reference_code_text"
    else:
        pred_col = f"{stream}_generated_code_text"
        ref_col = f"{stream}_reference_code_text"
    preds = group[pred_col].fillna("").astype(str).tolist() if pred_col in group else []
    refs = group[ref_col].fillna("").astype(str).tolist() if ref_col in group else []
    out = {f"bleu_{order}_higher_better": corpus_bleu(preds, refs, order) for order in range(1, 5)}
    out["rouge_l_higher_better"] = corpus_rouge_l(preds, refs)
    return out


def summarize(rows_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    group_cols = ["eval_track", "task_group", "task_name", "dataset"]
    metric_cols = [
        "bleu_1_higher_better",
        "bleu_2_higher_better",
        "bleu_3_higher_better",
        "bleu_4_higher_better",
        "rouge_l_higher_better",
        "mpjpe_mm_lower_better",
        "pa_mpjpe_mm_lower_better",
        "dtw_mpjpe_mm_lower_better",
        "length_ratio_closer_to_1",
    ]
    for keys, group in rows_df.groupby(group_cols, dropna=False):
        rec = {col: key for col, key in zip(group_cols, keys)}
        rec["examples"] = int(len(group))
        if str(rec["eval_track"]) == "Caption":
            preds = group["generated_caption"].fillna("").astype(str).tolist()
            refs = group["reference_caption"].fillna("").astype(str).tolist()
            for order in range(1, 5):
                rec[f"bleu_{order}_higher_better"] = corpus_bleu(preds, refs, order)
            rec["rouge_l_higher_better"] = corpus_rouge_l(preds, refs)
        elif str(rec["eval_track"]) == "Text-to-Motion":
            for stream in ["body", "lhand", "rhand", "combined_stream_avg", "combined_flat"]:
                scores = code_stream_scores(group, stream)
                for key, value in scores.items():
                    rec[f"{stream}_{key}"] = value
        for col in metric_cols:
            if col in rec:
                continue
            if col in group.columns and group[col].notna().any():
                rec[col] = float(group[col].mean())
            else:
                rec[col] = float("nan")
        summary_rows.append(rec)
    summary_df = pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)

    compact_rows: list[dict[str, Any]] = []
    for dataset, group in rows_df[rows_df["eval_track"] == "Caption"].groupby("dataset", dropna=False):
        preds = group["generated_caption"].fillna("").astype(str).tolist()
        refs = group["reference_caption"].fillna("").astype(str).tolist()
        rec = {
            "Track": "Caption",
            "Stream": "text",
            "Dataset": dataset,
            "Examples": int(len(group)),
            **{f"BLEU-{order} higher better": corpus_bleu(preds, refs, order) for order in range(1, 5)},
            "ROUGE-L higher better": corpus_rouge_l(preds, refs),
            "MPJPE mm lower better": float("nan"),
            "PA-MPJPE mm lower better": float("nan"),
            "DTW-MPJPE mm lower better": float("nan"),
            "Length ratio closer to 1": float("nan"),
        }
        compact_rows.append(rec)
    for dataset, group in rows_df[rows_df["eval_track"] == "Text-to-Motion"].groupby("dataset", dropna=False):
        for stream in ["body", "lhand", "rhand", "combined_stream_avg", "combined_flat"]:
            scores = code_stream_scores(group, stream)
            rec = {
                "Track": "Text-to-Motion",
                "Stream": stream,
                "Dataset": dataset,
                "Examples": int(len(group)),
                **{f"BLEU-{order} higher better": scores[f"bleu_{order}_higher_better"] for order in range(1, 5)},
                "ROUGE-L higher better": scores["rouge_l_higher_better"],
                "MPJPE mm lower better": float("nan"),
                "PA-MPJPE mm lower better": float("nan"),
                "DTW-MPJPE mm lower better": float("nan"),
                "Length ratio closer to 1": float("nan"),
            }
            if stream == "combined_stream_avg":
                for src, dst in [
                    ("mpjpe_mm_lower_better", "MPJPE mm lower better"),
                    ("pa_mpjpe_mm_lower_better", "PA-MPJPE mm lower better"),
                    ("dtw_mpjpe_mm_lower_better", "DTW-MPJPE mm lower better"),
                    ("length_ratio_closer_to_1", "Length ratio closer to 1"),
                ]:
                    vals = group[src].dropna().astype(float) if src in group else pd.Series(dtype=float)
                    rec[dst] = float(vals.mean()) if len(vals) else float("nan")
            compact_rows.append(rec)
    compact_df = pd.DataFrame(compact_rows).sort_values(["Track", "Dataset", "Stream"]).reset_index(drop=True)
    return summary_df, compact_df


def make_body_hand_stream_table(compact_df: pd.DataFrame) -> pd.DataFrame:
    """Readable long-form dataset table with one row per body/hand stream."""
    t2m = compact_df[compact_df["Track"] == "Text-to-Motion"].copy()
    if t2m.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for dataset, group in t2m.groupby("Dataset", dropna=False):
        combined = group[group["Stream"] == "combined_stream_avg"]
        if not combined.empty:
            crow = combined.iloc[0]
            motion_values = {
                "Full MPJPE mm lower better": crow.get("MPJPE mm lower better", float("nan")),
                "Full PA-MPJPE mm lower better": crow.get("PA-MPJPE mm lower better", float("nan")),
                "Full DTW-MPJPE mm lower better": crow.get("DTW-MPJPE mm lower better", float("nan")),
                "Full length ratio closer to 1": crow.get("Length ratio closer to 1", float("nan")),
            }
        else:
            motion_values = {
                "Full MPJPE mm lower better": float("nan"),
                "Full PA-MPJPE mm lower better": float("nan"),
                "Full DTW-MPJPE mm lower better": float("nan"),
                "Full length ratio closer to 1": float("nan"),
            }
        for stream, label in [
            ("body", "Body"),
            ("lhand", "LHand"),
            ("rhand", "RHand"),
            ("combined_stream_avg", "StreamAvg"),
        ]:
            match = group[group["Stream"] == stream]
            if match.empty:
                continue
            row = match.iloc[0]
            rec: dict[str, Any] = {
                "Dataset": dataset,
                "Stream": label,
                "Examples": int(row.get("Examples", 0)),
                **{f"BLEU-{order} higher better": row.get(f"BLEU-{order} higher better", float("nan")) for order in range(1, 5)},
                "ROUGE-L higher better": row.get("ROUGE-L higher better", float("nan")),
                **motion_values,
            }
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["Dataset", "Stream"]).reset_index(drop=True)


def make_body_hand_task_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Keep each T2M instruction task while separating body/hand stream scores."""
    t2m = summary_df[summary_df["eval_track"] == "Text-to-Motion"].copy()
    if t2m.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, src in t2m.sort_values(["dataset", "task_group", "task_name"]).iterrows():
        rec: dict[str, Any] = {
            "Task group": src.get("task_group", ""),
            "Task": src.get("task_name", ""),
            "Dataset": src.get("dataset", ""),
            "Examples": int(src.get("examples", 0)),
            "Full MPJPE mm lower better": src.get("mpjpe_mm_lower_better", float("nan")),
            "Full PA-MPJPE mm lower better": src.get("pa_mpjpe_mm_lower_better", float("nan")),
            "Full DTW-MPJPE mm lower better": src.get("dtw_mpjpe_mm_lower_better", float("nan")),
            "Full length ratio closer to 1": src.get("length_ratio_closer_to_1", float("nan")),
        }
        for stream, label in [
            ("body", "Body"),
            ("lhand", "LHand"),
            ("rhand", "RHand"),
            ("combined_stream_avg", "StreamAvg"),
        ]:
            for order in range(1, 5):
                rec[f"{label} BLEU-{order} higher better"] = src.get(
                    f"{stream}_bleu_{order}_higher_better", float("nan")
                )
            rec[f"{label} ROUGE-L higher better"] = src.get(f"{stream}_rouge_l_higher_better", float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows).reset_index(drop=True)


def write_table_png(table_df: pd.DataFrame, path: Path, title: str) -> None:
    if table_df.empty:
        return
    display = table_df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        elif pd.api.types.is_integer_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(int(x)))
        else:
            display[col] = display[col].astype(str)
    labels = [str(c).replace(" ", "\n") for c in display.columns]
    fig_width = max(14.5, 1.18 * len(display.columns))
    fig_height = max(3.2, 0.55 * len(display) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    table = ax.table(cellText=display.values, colLabels=labels, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.65)
    n_cols = display.shape[1]
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.8)
        if row_idx == 0:
            cell.set_facecolor("#e8edf3")
            cell.set_text_props(weight="bold", color="#111827")
            cell.set_height(cell.get_height() * 1.55)
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8fafc")
        else:
            cell.set_facecolor("#ffffff")
        if col_idx < n_cols and display.columns[col_idx] in {"Track", "Stream", "Dataset"}:
            cell.set_text_props(ha="left")
            cell.set_width(0.12 if display.columns[col_idx] != "Stream" else 0.14)
        else:
            cell.set_width(0.085)
    fig.text(
        0.01,
        0.025,
        "Validation uses autoregressive greedy generation. BLEU/ROUGE are higher better; MPJPE/PA/DTW are lower better.",
        fontsize=8.5,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_rows(rows: list[dict[str, Any]], max_rows: int, seed: int) -> list[dict[str, Any]]:
    if int(max_rows) <= 0 or int(max_rows) >= len(rows):
        return list(rows)
    rng = random.Random(int(seed))
    return rng.sample(list(rows), int(max_rows))


def limit_tasks(tasks: list[dict[str, Any]], max_tasks: int) -> list[dict[str, Any]]:
    if int(max_tasks) <= 0:
        return tasks
    return tasks[: int(max_tasks)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive validation for Gemma SOKE LoRA text/motion tasks.")
    parser.add_argument("--code-root", type=Path, default=BUNDLE_ROOT / "outputs" / "soke_motion_codes")
    parser.add_argument("--instructions-root", type=Path, default=BUNDLE_ROOT / "instructions")
    parser.add_argument("--output-root", type=Path, default=BUNDLE_ROOT / "05_autoreg_validation")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_DRIVE_ADAPTER)
    parser.add_argument("--base-model", default=os.environ.get("SOKE_GEMMA_BASE_MODEL", "google/gemma-3-270m"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-t2m-tasks", type=int, default=0)
    parser.add_argument("--max-caption-tasks", type=int, default=0)
    parser.add_argument("--include-text-to-motion", type=int, default=1)
    parser.add_argument("--include-caption", type=int, default=1)
    parser.add_argument("--max-new-motion-tokens", type=int, default=384)
    parser.add_argument("--max-new-text-tokens", type=int, default=96)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", type=int, default=0)
    parser.add_argument("--official-tokenizer-checkpoint", type=Path, default=DEFAULT_OFFICIAL_TOKENIZER)
    parser.add_argument("--norm-source", choices=["official", "local"], default="official")
    parser.add_argument("--body-model-root", type=Path, default=EXP_ROOT / "body_models")
    parser.add_argument("--frame-batch", type=int, default=2048)
    parser.add_argument("--raw-block-rows", type=int, default=64)
    parser.add_argument("--pa-block-rows", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.adapter.exists():
        raise FileNotFoundError(f"Missing LoRA adapter: {args.adapter}")
    if not args.official_tokenizer_checkpoint.exists():
        raise FileNotFoundError(f"Missing official SOKE tokenizer checkpoint: {args.official_tokenizer_checkpoint}")
    rows_path = args.code_root / f"{args.split}_soke_motion_codes.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing code rows: {rows_path}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.split}_autoreg"
    if args.max_rows > 0:
        run_name += f"_limit{args.max_rows}"
    out_dir = args.output_root / run_name
    if out_dir.exists() and args.force:
        for path in out_dir.glob("*"):
            if path.is_file():
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    details_csv = out_dir / "autoreg_validation_rows.csv"
    summary_csv = out_dir / "autoreg_validation_summary_by_task_dataset.csv"
    table_csv = out_dir / "autoreg_validation_compact_table.csv"
    table_png = out_dir / "autoreg_validation_compact_table.png"
    body_hand_stream_csv = out_dir / "autoreg_validation_body_hand_stream_metrics.csv"
    body_hand_stream_png = out_dir / "autoreg_validation_body_hand_stream_metrics.png"
    body_hand_task_csv = out_dir / "autoreg_validation_body_hand_metrics_by_task_dataset.csv"
    body_hand_task_png = out_dir / "autoreg_validation_body_hand_metrics_by_task_dataset.png"
    summary_json = out_dir / "autoreg_validation_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        json.dumps(
            {
                "kind": "preflight",
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "base_model": args.base_model,
                "adapter": str(args.adapter),
                "split": args.split,
                "max_rows": int(args.max_rows),
                "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    codecs = load_codecs(args.code_root)
    tasks_all = load_named_tasks(args.instructions_root / "template_instructions.json")
    t2m_tasks = limit_tasks([task for task in tasks_all if is_text_to_motion_task(task)], args.max_t2m_tasks)
    caption_tasks = limit_tasks([task for task in tasks_all if is_caption_task(task)], args.max_caption_tasks)
    data_rows = choose_rows(read_jsonl(rows_path), args.max_rows, args.seed)

    tokenizer, model = load_adapter_model(args, codecs)
    model.to(device)
    soke_tokenizer, soke_meta = load_official_model(args.official_tokenizer_checkpoint, device)
    if args.norm_source == "official":
        mean, std, norm_detail = official_mean_std_133()
    else:
        mean, std, norm_detail = local_mean_std_133()
    smplx_decoder = SMPLXJ22Decoder(args.body_model_root, device)

    rows_out: list[dict[str, Any]] = []
    total_examples = 0
    if args.include_text_to_motion:
        total_examples += len(data_rows) * len(t2m_tasks)
    if args.include_caption:
        total_examples += len(data_rows) * len(caption_tasks)
    started = time.time()
    example_idx = 0

    for row_idx, data_row in enumerate(data_rows):
        for eval_track, tasks in [
            ("Text-to-Motion", t2m_tasks if args.include_text_to_motion else []),
            ("Caption", caption_tasks if args.include_caption else []),
        ]:
            for task in tasks:
                example_idx += 1
                rendered = render_eval_prompt(data_row, task, codecs)
                max_new = (
                    min(int(args.max_new_motion_tokens), max(24, int(rendered["target_triplets"]) * 3 + 32))
                    if eval_track == "Text-to-Motion"
                    else int(args.max_new_text_tokens)
                )
                gen_text = generate_continuation(model, tokenizer, rendered["prompt"], device, max_new_tokens=max_new)
                rec: dict[str, Any] = {
                    "row_index": int(row_idx),
                    "example_index": int(example_idx),
                    "eval_track": eval_track,
                    "task_group": task["group"],
                    "task_name": task["name"],
                    "dataset": data_row.get("dataset", ""),
                    "source_alias": data_row.get("source_alias", ""),
                    "split": data_row.get("split", args.split),
                    "clip_id": data_row.get("clip_id", ""),
                    "clip_key": data_row.get("clip_key", ""),
                    "reference_caption": data_row.get("text", ""),
                    "prompt": rendered["prompt"],
                    "target": rendered["target"],
                    "generated_text": gen_text,
                    "max_new_tokens": int(max_new),
                }
                if eval_track == "Caption":
                    generated_caption = clean_generated_caption(gen_text, tokenizer.eos_token)
                    rec["generated_caption"] = generated_caption
                    rec["rouge_l_higher_better"] = rouge_l_f1(generated_caption, str(data_row.get("text", "")))
                else:
                    pred_body, pred_lhand, pred_rhand, recognized_tokens = parse_generated_triplets(gen_text, codecs)
                    pred_triplets = min(len(pred_body), len(pred_lhand), len(pred_rhand))
                    rec.update(
                        {
                            "recognized_motion_tokens": int(recognized_tokens),
                            "generated_triplets": int(pred_triplets),
                            "target_triplets": int(rendered["target_triplets"]),
                            "valid_triplet_token_ratio_higher_better": float((pred_triplets * 3) / max(int(recognized_tokens), 1)),
                        }
                    )
                    rec.update(
                        motion_token_metric_record(
                            rendered["target_body"],
                            rendered["target_lhand"],
                            rendered["target_rhand"],
                            pred_body,
                            pred_lhand,
                            pred_rhand,
                        )
                    )
                    try:
                        rec.update(
                            compute_motion_metrics(
                                data_row,
                                rendered["target_body"],
                                rendered["target_lhand"],
                                rendered["target_rhand"],
                                pred_body,
                                pred_lhand,
                                pred_rhand,
                                soke_tokenizer,
                                mean,
                                std,
                                smplx_decoder,
                                device,
                                frame_batch=args.frame_batch,
                                raw_block_rows=args.raw_block_rows,
                                pa_block_rows=args.pa_block_rows,
                            )
                        )
                    except Exception as exc:
                        rec["motion_metric_error"] = repr(exc)
                rows_out.append(rec)
                if example_idx == 1 or example_idx % int(args.progress_every) == 0 or example_idx == total_examples:
                    elapsed = (time.time() - started) / 60.0
                    print(
                        f"autoreg {example_idx}/{total_examples} | {eval_track} | {data_row.get('dataset')} | "
                        f"{task['name']} | elapsed={elapsed:.1f}m",
                        flush=True,
                    )
                if len(rows_out) % max(1, int(args.progress_every)) == 0:
                    pd.DataFrame(rows_out).to_csv(details_csv, index=False)

    rows_df = pd.DataFrame(rows_out)
    rows_df.to_csv(details_csv, index=False)
    summary_df, compact_df = summarize(rows_df)
    body_hand_stream_df = make_body_hand_stream_table(compact_df)
    body_hand_task_df = make_body_hand_task_table(summary_df)
    summary_df.to_csv(summary_csv, index=False)
    compact_df.to_csv(table_csv, index=False)
    body_hand_stream_df.to_csv(body_hand_stream_csv, index=False)
    body_hand_task_df.to_csv(body_hand_task_csv, index=False)
    write_table_png(compact_df, table_png, "Gemma SOKE Autoregressive Validation Table")
    write_table_png(
        body_hand_stream_df,
        body_hand_stream_png,
        "Gemma SOKE Body/Hand Stream Metrics And Full Motion Metrics",
    )
    write_table_png(
        body_hand_task_df,
        body_hand_task_png,
        "Gemma SOKE Body/Hand Code Metrics And Full Motion Metrics By Task",
    )
    payload = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_root": str(args.code_root),
        "instructions_root": str(args.instructions_root),
        "adapter": str(args.adapter),
        "base_model": args.base_model,
        "split": args.split,
        "max_rows": int(args.max_rows),
        "max_t2m_tasks": int(args.max_t2m_tasks),
        "max_caption_tasks": int(args.max_caption_tasks),
        "norm_source": args.norm_source,
        "norm_source_detail": norm_detail,
        "soke_tokenizer_meta": soke_meta,
        "details_csv": str(details_csv),
        "summary_csv": str(summary_csv),
        "compact_table_csv": str(table_csv),
        "compact_table_png": str(table_png),
        "body_hand_stream_csv": str(body_hand_stream_csv),
        "body_hand_stream_png": str(body_hand_stream_png),
        "body_hand_task_csv": str(body_hand_task_csv),
        "body_hand_task_png": str(body_hand_task_png),
        "examples": int(len(rows_df)),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote:", details_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", table_csv)
    print("Wrote:", table_png)
    print("Wrote:", body_hand_stream_csv)
    print("Wrote:", body_hand_stream_png)
    print("Wrote:", body_hand_task_csv)
    print("Wrote:", body_hand_task_png)
    print("Wrote:", summary_json)
    if not compact_df.empty:
        print(compact_df.to_string(index=False))


if __name__ == "__main__":
    main()
