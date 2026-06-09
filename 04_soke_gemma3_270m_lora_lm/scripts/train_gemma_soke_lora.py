from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from soke_gemma_data import SokeGemmaCausalDataset, crop_code_ids_like_soke, load_codecs, load_instructions, read_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rsync_tree(src: Path, dst: Path, retries: int = 3, sleep_sec: float = 10.0, delete: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, int(retries) + 1):
        try:
            if shutil.which("rsync"):
                cmd = ["rsync", "-a", "--partial", "--delay-updates", "--info=progress2"]
                if delete:
                    cmd.append("--delete")
                cmd.extend([f"{src}/", f"{dst}/"])
                subprocess.run(cmd, check=True)
            else:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            return
        except Exception:
            if attempt >= int(retries):
                raise
            time.sleep(float(sleep_sec))


def adapter_weight_exists(path: Path) -> bool:
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def adapter_ready(path: Path) -> bool:
    path = Path(path)
    return (path / "adapter_config.json").exists() and adapter_weight_exists(path)


def resolve_resume_info(args: argparse.Namespace) -> dict[str, Any] | None:
    if not bool(args.resume_training):
        return None
    adapter = args.run_root / str(args.resume_adapter_name)
    state_path = args.run_root / "last_train_state.pt"
    if not adapter_ready(adapter) or not state_path.exists():
        return None
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    epoch = int(state.get("epoch", 0))
    if epoch <= 0:
        raise RuntimeError(f"Invalid resume state epoch in {state_path}: {epoch}")
    return {
        "adapter": adapter,
        "state_path": state_path,
        "state": state,
        "epoch": epoch,
        "global_step": int(state.get("global_step", 0)),
    }


def copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prune_epoch_saves(root: Path, keep: int) -> list[str]:
    if int(keep) <= 0 or not root.exists():
        return []
    saves = sorted([p for p in root.glob("epoch_*") if p.is_dir()])
    to_delete = saves[: max(0, len(saves) - int(keep))]
    removed: list[str] = []
    for path in to_delete:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    return removed


def create_epoch_snapshot(
    run_root: Path,
    *,
    epoch: int,
    global_step: int,
    best_epoch: int | None,
    best_loss: float,
) -> Path:
    snapshot = run_root / "epoch_saves" / f"epoch_{int(epoch):04d}"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True, exist_ok=True)

    for name in ["last_adapter", "best_adapter"]:
        copy_path(run_root / name, snapshot / name)
    for name in [
        "last_train_state.pt",
        "best_train_state.pt",
        "history.csv",
        "training_events.jsonl",
        "latest_status.json",
        "train_args.json",
        "model_trainable_config.json",
        "optimizer_param_groups.json",
        "tokenization_audit.csv",
    ]:
        copy_path(run_root / name, snapshot / name)
    copy_path(run_root / "motion_val", snapshot / "motion_val")

    manifest = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_loss": float(best_loss),
        "last_adapter_has_weights": adapter_weight_exists(snapshot / "last_adapter"),
        "best_adapter_has_weights": adapter_weight_exists(snapshot / "best_adapter"),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["last_adapter_has_weights"] or not manifest["best_adapter_has_weights"]:
        raise RuntimeError(f"Incomplete epoch snapshot: {snapshot}")
    return snapshot


def verify_synced_run(dest: Path, latest_snapshot_name: str | None = None) -> None:
    required_dirs = [dest / "last_adapter", dest / "best_adapter"]
    required_files = [dest / "history.csv", dest / "latest_status.json", dest / "train_args.json"]
    if latest_snapshot_name:
        snap = dest / "epoch_saves" / latest_snapshot_name
        required_dirs.extend([snap / "last_adapter", snap / "best_adapter"])
        required_files.append(snap / "snapshot_manifest.json")

    missing = [str(path) for path in required_files if not path.exists()]
    missing.extend(str(path) for path in required_dirs if not adapter_weight_exists(path))
    if missing:
        raise RuntimeError(f"Drive sync verification failed. Missing/incomplete: {missing}")


def collate_examples(examples: list[dict[str, Any]], tokenizer: Any, max_seq_len: int) -> dict[str, torch.Tensor | list[Any]]:
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    meta: dict[str, list[Any]] = {k: [] for k in ["clip_key", "dataset", "source_alias", "code_len", "target_token_count"]}
    eos = tokenizer.eos_token or ""
    for ex in examples:
        prompt = ex["prompt"].rstrip() + "\n"
        target = ex["target"].strip()
        if eos:
            full = prompt + target + eos
        else:
            full = prompt + target
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = tokenizer(full, add_special_tokens=False).input_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(full_ids) > int(max_seq_len):
            full_ids = full_ids[: int(max_seq_len)]
            labels = labels[: int(max_seq_len)]
        input_rows.append(full_ids)
        label_rows.append(labels)
        for key in meta:
            meta[key].append(ex.get(key))
    max_len = max(len(x) for x in input_rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(input_rows), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(input_rows), max_len), dtype=torch.long)
    labels = torch.full((len(input_rows), max_len), -100, dtype=torch.long)
    for i, (ids, labs) in enumerate(zip(input_rows, label_rows)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
        labels[i, : len(labs)] = torch.tensor(labs, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, **meta}


def resolve_lora_targets(model: torch.nn.Module, requested: str) -> list[str]:
    if requested and requested != "auto":
        return [x.strip() for x in requested.split(",") if x.strip()]
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in suffixes:
                found.add(leaf)
    if found:
        return sorted(found)
    fallback = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] not in {"lm_head", "embed_tokens", "wte"}
        }
    )
    if not fallback:
        raise RuntimeError(
            "Could not find safe Linear modules for LoRA targets. "
            "Set --lora-target-modules explicitly for this model family."
        )
    return fallback


def check_torchao_compatibility() -> dict[str, Any]:
    try:
        from packaging.version import Version
    except Exception:
        return {"torchao_version": "", "torchao_status": "packaging_unavailable"}
    try:
        torchao_version = importlib_metadata.version("torchao")
    except importlib_metadata.PackageNotFoundError:
        return {"torchao_version": "", "torchao_status": "not_installed"}
    minimum = Version("0.16.0")
    if Version(torchao_version) < minimum:
        raise RuntimeError(
            f"Incompatible torchao=={torchao_version}. PEFT >=0.19 requires torchao >= {minimum} "
            "when torchao is installed. Run: pip install -U 'torchao>=0.16.0'"
        )
    return {"torchao_version": torchao_version, "torchao_status": "ok"}


def flash_attention_2_probe(device: torch.device) -> dict[str, Any]:
    status: dict[str, Any] = {
        "flash_attn_2_available": False,
        "flash_attn_version": "",
        "flash_attn_2_reason": "",
    }
    if device.type != "cuda":
        status["flash_attn_2_reason"] = "cuda_unavailable"
        return status
    try:
        status["flash_attn_version"] = importlib_metadata.version("flash_attn")
    except importlib_metadata.PackageNotFoundError:
        status["flash_attn_2_reason"] = "flash_attn_not_installed"
        return status
    try:
        from transformers.utils import is_flash_attn_2_available

        status["flash_attn_2_available"] = bool(is_flash_attn_2_available())
        if not status["flash_attn_2_available"]:
            status["flash_attn_2_reason"] = "transformers_reports_unavailable"
    except Exception as exc:
        status["flash_attn_2_available"] = True
        status["flash_attn_2_reason"] = f"transformers_probe_failed_assuming_available:{type(exc).__name__}"
    return status


def configure_attention_backend(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    requested_attn_impl = str(getattr(args, "attn_implementation", ""))
    probe = flash_attention_2_probe(device)
    if requested_attn_impl == "auto":
        args.attn_implementation = "flash_attention_2" if probe["flash_attn_2_available"] else "sdpa"

    attn_impl = str(getattr(args, "attn_implementation", ""))
    requested = str(getattr(args, "sdpa_kernel", "auto"))
    status: dict[str, Any] = {
        "attn_implementation_requested": requested_attn_impl,
        "attn_implementation": attn_impl,
        "sdpa_kernel": requested,
        "flash_sdp_enabled": None,
        "mem_efficient_sdp_enabled": None,
        "math_sdp_enabled": None,
        **probe,
    }
    if attn_impl == "flash_attention_2":
        if device.type != "cuda":
            raise RuntimeError('attn_implementation="flash_attention_2" requires CUDA')
        if not status["flash_attn_version"]:
            raise RuntimeError(
                'Missing flash-attn for attn_implementation="flash_attention_2". '
                "Install it with: pip install -U flash-attn --no-build-isolation"
            )
        if not status["flash_attn_2_available"]:
            raise RuntimeError(
                'Transformers reports FlashAttention-2 is unavailable. '
                "Check CUDA, torch, and flash-attn installation."
            )
        return status

    if attn_impl != "sdpa":
        return status

    if requested == "auto":
        pass
    elif device.type != "cuda":
        raise RuntimeError(f"--sdpa-kernel {requested!r} requires CUDA")
    elif not hasattr(torch.backends, "cuda") or not hasattr(torch.backends.cuda, "enable_flash_sdp"):
        raise RuntimeError("This PyTorch build does not expose CUDA SDPA backend controls")
    elif requested == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    elif requested == "mem_efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    elif requested == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        raise ValueError(f"Unknown --sdpa-kernel: {requested}")

    if hasattr(torch.backends, "cuda"):
        for key, fn_name in [
            ("flash_sdp_enabled", "flash_sdp_enabled"),
            ("mem_efficient_sdp_enabled", "mem_efficient_sdp_enabled"),
            ("math_sdp_enabled", "math_sdp_enabled"),
        ]:
            fn = getattr(torch.backends.cuda, fn_name, None)
            if callable(fn):
                status[key] = bool(fn())
    return status


def configure_torch_runtime(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    use_tf32 = bool(args.tf32) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        if use_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    return {
        "tf32": bool(use_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else False,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if device.type == "cuda" else False,
    }


def data_loader_kwargs(args: argparse.Namespace, *, shuffle: bool, collate: Any) -> dict[str, Any]:
    num_workers = int(args.num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": bool(args.pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    return kwargs


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def motion_only_text(text: str) -> str:
    prefixes = ("<motion_id_", "<hand_id_", "<rhand_id_")
    return " ".join(tok for tok in str(text).split() if tok.startswith(prefixes))


MOTION_TOKEN_RE = re.compile(r"<(motion|hand|rhand)_id_(\d+)>")


def select_text_to_motion_task(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    for task in tasks:
        inputs = [str(x) for x in task.get("input", [])]
        outputs = [str(x) for x in task.get("output", [])]
        has_caption_input = any("<Caption_Placeholder>" in x for x in inputs)
        has_motion_output = any("<Motion_Placeholder>" in x for x in outputs)
        has_motion_input = any("<Motion_Placeholder>" in x for x in inputs)
        if has_caption_input and has_motion_output and not has_motion_input:
            return task
    return tasks[0]


def parse_generated_triplets(text: str, codecs: Any) -> tuple[list[int], list[int], list[int], int]:
    raw = [(part, int(idx)) for part, idx in MOTION_TOKEN_RE.findall(str(text))]
    body: list[int] = []
    lhand: list[int] = []
    rhand: list[int] = []
    state = "motion"
    cur_body = cur_lhand = None
    for part, idx in raw:
        if state == "motion":
            if part != "motion" or idx >= codecs.size("body"):
                continue
            cur_body = idx
            state = "hand"
        elif state == "hand":
            if part == "motion" and idx < codecs.size("body"):
                cur_body = idx
                state = "hand"
                continue
            if part != "hand" or idx >= codecs.size("lhand"):
                state = "motion"
                cur_body = None
                cur_lhand = None
                continue
            cur_lhand = idx
            state = "rhand"
        else:
            if part == "motion" and idx < codecs.size("body"):
                cur_body = idx
                cur_lhand = None
                state = "hand"
                continue
            if part == "rhand" and idx < codecs.size("rhand") and cur_body is not None and cur_lhand is not None:
                body.append(int(cur_body))
                lhand.append(int(cur_lhand))
                rhand.append(int(idx))
            state = "motion"
            cur_body = None
            cur_lhand = None
    return body, lhand, rhand, len(raw)


@torch.no_grad()
def evaluate_motion_code_generation(
    model: torch.nn.Module,
    tokenizer: Any,
    val_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    codecs: Any,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    if not val_rows:
        return {}
    n_sample = int(math.ceil(len(val_rows) * max(0.0, float(args.motion_val_fraction))))
    if int(args.motion_val_max_rows) > 0:
        n_sample = min(n_sample, int(args.motion_val_max_rows))
    n_sample = max(1, min(n_sample, len(val_rows)))
    rng = random.Random(int(args.seed) + int(epoch) * 1009)
    selected = rng.sample(val_rows, n_sample)
    task = select_text_to_motion_task(tasks)
    ds = SokeGemmaCausalDataset(selected, [task], codecs, random_drop=False, seed=int(args.seed) + int(epoch) * 7919)

    rows: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        sample = ds[idx]
        row = selected[int(sample["data_idx"])]
        target_body, target_lhand, target_rhand = crop_code_ids_like_soke(
            row["body_ids"],
            row["lhand_ids"],
            row["rhand_ids"],
            random_drop=False,
        )
        target_len = max(1, min(len(target_body), len(target_lhand), len(target_rhand)))
        prompt = sample["prompt"].rstrip() + "\n"
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        max_new = min(int(args.motion_val_max_new_tokens), max(12, target_len * 3 + 8))
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(out_ids[0, input_ids.shape[1] :], skip_special_tokens=False)
        pred_body, pred_lhand, pred_rhand, recognized_tokens = parse_generated_triplets(gen_text, codecs)
        pred_len = min(len(pred_body), len(pred_lhand), len(pred_rhand))
        compare_len = min(pred_len, target_len)
        body_correct = sum(int(pred_body[i] == int(target_body[i])) for i in range(compare_len))
        lhand_correct = sum(int(pred_lhand[i] == int(target_lhand[i])) for i in range(compare_len))
        rhand_correct = sum(int(pred_rhand[i] == int(target_rhand[i])) for i in range(compare_len))
        total_target_tokens = target_len * 3
        total_correct = body_correct + lhand_correct + rhand_correct
        rec = {
            "epoch": int(epoch),
            "row_index": int(idx),
            "clip_key": row.get("clip_key", ""),
            "dataset": row.get("dataset", ""),
            "source_alias": row.get("source_alias", ""),
            "target_triplets": int(target_len),
            "generated_triplets": int(pred_len),
            "recognized_motion_tokens": int(recognized_tokens),
            "valid_triplet_token_ratio_higher_better": float((pred_len * 3) / max(recognized_tokens, 1)),
            "length_ratio_closer_to_1": float(pred_len / max(target_len, 1)),
            "body_code_accuracy_higher_better": float(body_correct / max(target_len, 1)),
            "lhand_code_accuracy_higher_better": float(lhand_correct / max(target_len, 1)),
            "rhand_code_accuracy_higher_better": float(rhand_correct / max(target_len, 1)),
            "motion_code_accuracy_higher_better": float(total_correct / max(total_target_tokens, 1)),
            "has_motion_triplets": float(pred_len > 0),
        }
        rows.append(rec)

    out_dir = args.run_root / "motion_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame(rows)
    rows_path = out_dir / f"motion_code_epoch_{int(epoch):04d}_rows.csv"
    rows_df.to_csv(rows_path, index=False)
    metric_cols = [
        "valid_triplet_token_ratio_higher_better",
        "length_ratio_closer_to_1",
        "body_code_accuracy_higher_better",
        "lhand_code_accuracy_higher_better",
        "rhand_code_accuracy_higher_better",
        "motion_code_accuracy_higher_better",
        "has_motion_triplets",
    ]
    summary = {
        "epoch": int(epoch),
        "sample_rows": int(len(rows_df)),
        "rows_csv": str(rows_path),
    }
    for col in metric_cols:
        summary[f"{col}_mean"] = float(rows_df[col].mean()) if col in rows_df else float("nan")
        summary[f"{col}_median"] = float(rows_df[col].median()) if col in rows_df else float("nan")
    summary_path = out_dir / "motion_code_summary.csv"
    summary_df = pd.DataFrame([summary])
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        prev = prev[prev["epoch"].astype(int) != int(epoch)] if "epoch" in prev.columns else prev
        summary_df = pd.concat([prev, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    (out_dir / f"motion_code_epoch_{int(epoch):04d}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {f"motion_val_{k}": v for k, v in summary.items() if isinstance(v, (int, float))}


def build_tokenizer_and_model(args: argparse.Namespace, codecs: Any) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AddedToken, AutoModelForCausalLM, AutoTokenizer

    dep_meta = check_torchao_compatibility()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    auth_kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=bool(args.trust_remote_code),
        **auth_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    motion_tokens = codecs.added_tokens()
    base_vocab_size = len(tokenizer)
    n_added = tokenizer.add_tokens([
        AddedToken(tok, lstrip=True, rstrip=False, normalized=False, special=False)
        for tok in motion_tokens
    ])
    motion_token_ids = [int(tokenizer.convert_tokens_to_ids(tok)) for tok in motion_tokens]
    trainable_motion_token_ids = sorted({tid for tid in motion_token_ids if tid >= base_vocab_size})
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        **auth_kwargs,
    }
    if args.attn_implementation and args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if bool(args.gradient_checkpointing):
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    targets = resolve_lora_targets(model, args.lora_target_modules)
    train_motion_tokens = bool(args.train_motion_token_embeddings) and bool(trainable_motion_token_ids)
    init_adapter = Path(args.init_adapter) if args.init_adapter else None
    if init_adapter is not None:
        if not init_adapter.exists():
            raise FileNotFoundError(f"Missing init adapter: {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            trainable_token_indices=trainable_motion_token_ids if train_motion_tokens else None,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    meta = {
        "base_vocab_size": int(base_vocab_size),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "motion_tokens_requested": int(len(motion_tokens)),
        "motion_tokens_added": int(n_added),
        "train_motion_token_embeddings": bool(train_motion_tokens),
        "trainable_motion_token_count": int(len(trainable_motion_token_ids)),
        "trainable_motion_token_id_min": int(min(trainable_motion_token_ids)) if trainable_motion_token_ids else None,
        "trainable_motion_token_id_max": int(max(trainable_motion_token_ids)) if trainable_motion_token_ids else None,
        "lora_target_modules": targets,
        "init_adapter": str(init_adapter) if init_adapter is not None else "",
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "attn_implementation": str(args.attn_implementation),
        "sdpa_kernel": str(args.sdpa_kernel),
        **dep_meta,
    }
    return tokenizer, model, meta


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    lora_params: list[torch.nn.Parameter] = []
    motion_token_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    group_counts = {"lora": 0, "motion_token_embeddings": 0, "other_trainable": 0}
    group_names: dict[str, list[str]] = {key: [] for key in group_counts}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "trainable_tokens_delta" in name:
            motion_token_params.append(param)
            group_counts["motion_token_embeddings"] += int(param.numel())
            group_names["motion_token_embeddings"].append(name)
        elif "lora_" in name:
            lora_params.append(param)
            group_counts["lora"] += int(param.numel())
            group_names["lora"].append(name)
        else:
            other_params.append(param)
            group_counts["other_trainable"] += int(param.numel())
            group_names["other_trainable"].append(name)

    if not any((lora_params, motion_token_params, other_params)):
        raise RuntimeError("No trainable parameters found after PEFT wrapping")

    param_groups: list[dict[str, Any]] = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(args.lr), "name": "lora"})
    if motion_token_params:
        param_groups.append({"params": motion_token_params, "lr": float(args.motion_token_lr), "name": "motion_token_embeddings"})
    if other_params:
        param_groups.append({"params": other_params, "lr": float(args.lr), "name": "other_trainable"})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.99), weight_decay=0.0)
    summary = {
        "param_groups": [
            {
                "name": group.get("name", f"group_{idx}"),
                "lr": float(group["lr"]),
                "trainable_params": int(sum(param.numel() for param in group["params"])),
            }
            for idx, group in enumerate(param_groups)
        ],
        "trainable_param_counts": group_counts,
        "trainable_param_names": group_names,
    }
    return optimizer, summary


def optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{idx}")): float(group["lr"])
        for idx, group in enumerate(optimizer.param_groups)
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict[str, float]:
    model.eval()
    non_blocking = bool(getattr(loader, "pin_memory", False))
    loss_sum = 0.0
    n_batches = 0
    correct = 0
    total = 0
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=non_blocking)
        out = model(
            input_ids=batch["input_ids"].to(device, non_blocking=non_blocking),
            attention_mask=batch["attention_mask"].to(device, non_blocking=non_blocking),
            labels=labels,
        )
        loss_sum += float(out.loss.detach().cpu())
        n_batches += 1
        logits = out.logits[:, :-1, :]
        shifted = labels[:, 1:]
        mask = shifted != -100
        if mask.any():
            pred = logits.argmax(dim=-1)
            correct += int((pred[mask] == shifted[mask]).sum().detach().cpu())
            total += int(mask.sum().detach().cpu())
        if max_batches and n_batches >= int(max_batches):
            break
    return {
        "loss": loss_sum / max(n_batches, 1),
        "token_accuracy": correct / max(total, 1),
        "tokens": float(total),
        "batches": float(n_batches),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Gemma 3 270M LoRA on SOKE flattened motion-token rows.")
    parser.add_argument("--code-root", type=Path, default=BUNDLE_ROOT / "outputs" / "soke_motion_codes")
    parser.add_argument("--instructions-root", type=Path, default=BUNDLE_ROOT / "instructions")
    parser.add_argument("--run-root", type=Path, default=BUNDLE_ROOT / "outputs" / "runs" / "gemma3_270m_lora_soke")
    parser.add_argument("--base-model", default=os.environ.get("SOKE_GEMMA_BASE_MODEL", "google/gemma-3-270m"))
    parser.add_argument("--init-adapter", type=Path, default=None)
    parser.add_argument("--stage", choices=["lm_pretrain", "lm_instruct"], default="lm_pretrain")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--resume-training", type=int, default=1)
    parser.add_argument("--resume-adapter-name", default="last_adapter")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--motion-token-lr", type=float, default=2e-5)
    parser.add_argument("--eta-min", type=float, default=1e-6)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--gradient-checkpointing", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--sdpa-kernel", choices=["auto", "flash", "mem_efficient", "math"], default="auto")
    parser.add_argument("--max-train-logical-rows", type=int, default=0)
    parser.add_argument("--max-val-logical-rows", type=int, default=2048)
    parser.add_argument("--eval-max-batches", type=int, default=0)
    parser.add_argument("--validation-every-epochs", type=int, default=5)
    parser.add_argument("--motion-val-enabled", type=int, default=1)
    parser.add_argument("--motion-val-every-epochs", type=int, default=5)
    parser.add_argument("--motion-val-fraction", type=float, default=0.10)
    parser.add_argument("--motion-val-max-rows", type=int, default=256)
    parser.add_argument("--motion-val-max-new-tokens", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--persistent-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--tf32", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=0)
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", type=int, default=0)
    parser.add_argument("--torch-compile-dynamic", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="auto")
    parser.add_argument("--train-motion-token-embeddings", type=int, default=1)
    parser.add_argument("--random-drop", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--keep-local-epoch-saves", type=int, default=5)
    parser.add_argument("--sync-to-drive", type=int, default=0)
    parser.add_argument("--drive-sync-dest", type=Path, default=Path("/content/drive/MyDrive/folder/COLAB/Tokenizer/04_soke_gemma3_270m_lora_lm/runs/gemma3_270m_lora_soke"))
    parser.add_argument("--drive-sync-retries", type=int, default=5)
    parser.add_argument("--drive-sync-sleep-sec", type=float, default=20.0)
    parser.add_argument("--drive-sync-delete", type=int, default=1)
    parser.add_argument("--keep-drive-epoch-saves", type=int, default=5)
    parser.add_argument("--trust-remote-code", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.run_root.mkdir(parents=True, exist_ok=True)
    resume_info = resolve_resume_info(args)
    if resume_info is not None:
        args.init_adapter = Path(resume_info["adapter"])
    (args.run_root / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_status = configure_torch_runtime(args, device)
    attention_status = configure_attention_backend(args, device)
    (args.run_root / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    preflight = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "base_model": args.base_model,
        "run_root": str(args.run_root),
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "virtual_batch_size": int(args.batch_size) * int(args.grad_accum),
        "max_seq_len": int(args.max_seq_len),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "persistent_workers": bool(args.persistent_workers) if int(args.num_workers) > 0 else False,
        "prefetch_factor": int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        "torch_compile": bool(args.torch_compile),
        "torch_compile_mode": str(args.torch_compile_mode),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "attn_implementation": str(args.attn_implementation),
        **runtime_status,
        **attention_status,
        "resume_training": bool(args.resume_training),
        "resume_found": bool(resume_info is not None),
        "resume_epoch": int(resume_info["epoch"]) if resume_info is not None else 0,
        "resume_global_step": int(resume_info["global_step"]) if resume_info is not None else 0,
    }
    print(json.dumps({"kind": "preflight", **preflight}, ensure_ascii=False), flush=True)

    codecs = load_codecs(args.code_root)
    inst_path = args.instructions_root / ("template_pretrain.json" if args.stage == "lm_pretrain" else "template_instructions.json")
    tasks = load_instructions(inst_path)
    train_rows = read_jsonl(args.code_root / "train_soke_motion_codes.jsonl")
    val_rows = read_jsonl(args.code_root / "val_soke_motion_codes.jsonl")
    tokenizer, model, model_meta = build_tokenizer_and_model(args, codecs)
    (args.run_root / "model_trainable_config.json").write_text(json.dumps(model_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Audit atomic SOKE motion-token handling before training.
    audit_ds = SokeGemmaCausalDataset(train_rows[: min(16, len(train_rows))], tasks, codecs, random_drop=False, seed=args.seed)
    audit_rows = []
    for i in range(len(audit_ds)):
        sample = audit_ds[i]
        motion_text = motion_only_text(sample["target"])
        expected = len(motion_text.split())
        got = len(tokenizer(motion_text, add_special_tokens=False).input_ids)
        audit_rows.append({"idx": i, "expected_motion_tokens": expected, "tokenized_ids": got, "ok": expected == got})
    pd.DataFrame(audit_rows).to_csv(args.run_root / "tokenization_audit.csv", index=False)
    if not all(r["ok"] for r in audit_rows):
        raise RuntimeError(f"Motion-token atomic audit failed. See {args.run_root / 'tokenization_audit.csv'}")

    train_ds = SokeGemmaCausalDataset(
        train_rows,
        tasks,
        codecs,
        random_drop=bool(args.random_drop),
        max_logical_rows=args.max_train_logical_rows,
        seed=args.seed,
    )
    val_ds = SokeGemmaCausalDataset(
        val_rows,
        tasks,
        codecs,
        random_drop=False,
        max_logical_rows=args.max_val_logical_rows,
        seed=args.seed + 100000,
    )
    collate = lambda batch: collate_examples(batch, tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_ds, **data_loader_kwargs(args, shuffle=True, collate=collate))
    val_loader = DataLoader(val_ds, **data_loader_kwargs(args, shuffle=False, collate=collate))

    model.to(device)
    optimizer, optimizer_meta = build_optimizer(args, model)
    if bool(args.torch_compile):
        model = torch.compile(
            model,
            mode=str(args.torch_compile_mode),
            fullgraph=bool(args.torch_compile_fullgraph),
            dynamic=bool(args.torch_compile_dynamic),
        )
    (args.run_root / "optimizer_param_groups.json").write_text(json.dumps(optimizer_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.eta_min)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.dtype == "fp16"))
    use_amp = device.type == "cuda" and args.dtype in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    history_path = args.run_root / "history.csv"
    history: list[dict[str, Any]] = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    best_loss = math.inf
    best_epoch: int | None = None
    best_state_path = args.run_root / "best_train_state.pt"
    if best_state_path.exists():
        best_state = torch.load(best_state_path, map_location="cpu", weights_only=False)
        best_loss = float(best_state.get("best_loss", math.inf))
        best_epoch = int(best_state.get("best_epoch")) if best_state.get("best_epoch") is not None else None
    elif history:
        for old_row in history:
            try:
                val_loss = float(old_row.get("val_loss", math.nan))
            except Exception:
                val_loss = math.nan
            if math.isfinite(val_loss) and val_loss < best_loss:
                best_loss = val_loss
                best_epoch = int(old_row.get("epoch", 0))
    global_step = 0
    start_epoch = 1
    if resume_info is not None:
        state = resume_info["state"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state.get("global_step", resume_info["global_step"]))
        start_epoch = int(state.get("epoch", resume_info["epoch"])) + 1
    event_path = args.run_root / "training_events.jsonl"

    def write_event(kind: str, payload: dict[str, Any]) -> None:
        event = {"kind": kind, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        (args.run_root / "latest_status.json").write_text(json.dumps(event, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(event, ensure_ascii=False), flush=True)

    if resume_info is not None:
        write_event(
            "resume",
            {
                "resume_adapter": str(resume_info["adapter"]),
                "resume_state_path": str(resume_info["state_path"]),
                "resume_epoch": int(resume_info["epoch"]),
                "start_epoch": int(start_epoch),
                "target_epochs": int(args.epochs),
                "global_step": int(global_step),
                "history_rows_loaded": int(len(history)),
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "best_loss": float(best_loss),
                "train_logical_rows": len(train_ds),
                "val_logical_rows": len(val_ds),
                "tasks": len(tasks),
                "model_trainable_config": model_meta,
                "optimizer_param_groups": optimizer_meta["param_groups"],
            },
        )
    else:
        write_event(
            "start",
            {
                "train_logical_rows": len(train_ds),
                "val_logical_rows": len(val_ds),
                "tasks": len(tasks),
                "model_trainable_config": model_meta,
                "optimizer_param_groups": optimizer_meta["param_groups"],
            },
        )
    if start_epoch > int(args.epochs):
        payload = {
            "resume_epoch": int(start_epoch - 1),
            "target_epochs": int(args.epochs),
            "global_step": int(global_step),
            "run_root": str(args.run_root),
        }
        write_event("already_complete", payload)
        return
    for epoch in range(int(start_epoch), int(args.epochs) + 1):
        model.train()
        started = time.time()
        loss_sum = 0.0
        n_examples = 0
        optimizer.zero_grad(set_to_none=True)
        non_blocking = bool(args.pin_memory)
        for micro_idx, batch in enumerate(train_loader, start=1):
            labels = batch["labels"].to(device, non_blocking=non_blocking)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"].to(device, non_blocking=non_blocking),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=non_blocking),
                    labels=labels,
                )
                loss = out.loss / max(1, int(args.grad_accum))
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            raw_loss = float(out.loss.detach().cpu())
            bsz = int(batch["input_ids"].shape[0])
            loss_sum += raw_loss * bsz
            n_examples += bsz
            if micro_idx % int(args.grad_accum) == 0 or micro_idx == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or (int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0):
                    write_event(
                        "optimizer_step",
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "micro_idx": micro_idx,
                            "loss": raw_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "learning_rates": optimizer_lrs(optimizer),
                        },
                    )
        train_loss = loss_sum / max(n_examples, 1)
        should_validate = (
            epoch == 1
            or epoch == int(args.epochs)
            or (int(args.validation_every_epochs) > 0 and epoch % int(args.validation_every_epochs) == 0)
        )
        if should_validate:
            val = evaluate(model, val_loader, device, max_batches=args.eval_max_batches)
        else:
            val = {"loss": float("nan"), "token_accuracy": float("nan"), "tokens": 0.0, "batches": 0.0}
        motion_val: dict[str, float] = {}
        should_motion_validate = (
            bool(args.motion_val_enabled)
            and should_validate
            and (
                epoch == 1
                or epoch == int(args.epochs)
                or (int(args.motion_val_every_epochs) > 0 and epoch % int(args.motion_val_every_epochs) == 0)
            )
        )
        if should_motion_validate:
            motion_val = evaluate_motion_code_generation(
                model,
                tokenizer,
                val_rows,
                tasks,
                codecs,
                args,
                device,
                epoch,
            )
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_token_accuracy_higher_better": val["token_accuracy"],
            "validated_this_epoch": bool(should_validate),
            "lr": optimizer.param_groups[0]["lr"],
            "learning_rates": optimizer_lrs(optimizer),
            "epoch_elapsed_sec": time.time() - started,
        }
        row.update(motion_val)
        history.append(row)
        pd.DataFrame(history).to_csv(args.run_root / "history.csv", index=False)
        scheduler.step()

        last_dir = args.run_root / "last_adapter"
        model_to_save = unwrap_compiled_model(model)
        model_to_save.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        torch.save({"epoch": epoch, "global_step": global_step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, args.run_root / "last_train_state.pt")
        if math.isfinite(float(val["loss"])) and val["loss"] < best_loss:
            best_loss = val["loss"]
            best_epoch = epoch
            best_dir = args.run_root / "best_adapter"
            model_to_save.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            torch.save({"epoch": epoch, "global_step": global_step, "best_loss": best_loss, "best_epoch": best_epoch}, args.run_root / "best_train_state.pt")
        write_event("epoch_end", row)
        if args.sync_to_drive and (epoch % max(1, int(args.save_every_epochs)) == 0 or epoch == int(args.epochs)):
            snapshot = create_epoch_snapshot(
                args.run_root,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_loss=best_loss,
            )
            removed_local = prune_epoch_saves(args.run_root / "epoch_saves", args.keep_local_epoch_saves)
            rsync_tree(
                args.run_root,
                args.drive_sync_dest,
                retries=args.drive_sync_retries,
                sleep_sec=args.drive_sync_sleep_sec,
                delete=bool(args.drive_sync_delete),
            )
            removed_drive = prune_epoch_saves(args.drive_sync_dest / "epoch_saves", args.keep_drive_epoch_saves)
            verify_synced_run(args.drive_sync_dest, latest_snapshot_name=snapshot.name)
            write_event(
                "drive_sync",
                {
                    "epoch": epoch,
                    "dest": str(args.drive_sync_dest),
                    "snapshot": snapshot.name,
                    "removed_local_epoch_saves": removed_local,
                    "removed_drive_epoch_saves": removed_drive,
                },
            )
        elif epoch % max(1, int(args.save_every_epochs)) == 0 or epoch == int(args.epochs):
            snapshot = create_epoch_snapshot(
                args.run_root,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_loss=best_loss,
            )
            removed_local = prune_epoch_saves(args.run_root / "epoch_saves", args.keep_local_epoch_saves)
            write_event("local_snapshot", {"epoch": epoch, "snapshot": snapshot.name, "removed_local_epoch_saves": removed_local})

    payload = {"best_val_loss": best_loss, "epochs": args.epochs, "run_root": str(args.run_root)}
    (args.run_root / "final_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_event("complete", payload)
    if args.sync_to_drive:
        rsync_tree(
            args.run_root,
            args.drive_sync_dest,
            retries=args.drive_sync_retries,
            sleep_sec=args.drive_sync_sleep_sec,
            delete=bool(args.drive_sync_delete),
        )
        prune_epoch_saves(args.drive_sync_dest / "epoch_saves", args.keep_drive_epoch_saves)
        verify_synced_run(args.drive_sync_dest)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        print(
            json.dumps(
                {
                    "kind": "cuda_oom",
                    "message": str(exc),
                    "suggestion": "Reduce HARDWARE_BATCH_SIZE and increase GRAD_ACCUM if you want the same virtual batch.",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
