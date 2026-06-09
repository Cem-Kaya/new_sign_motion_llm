from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class SokePartCodecs:
    body_size: int = 96
    lhand_size: int = 192
    rhand_size: int = 192

    def size(self, part: str) -> int:
        if part == "body":
            return int(self.body_size)
        if part == "lhand":
            return int(self.lhand_size)
        if part == "rhand":
            return int(self.rhand_size)
        raise KeyError(part)

    def start_id(self, part: str) -> int:
        return self.size(part)

    def end_id(self, part: str) -> int:
        return self.size(part) + 1

    def mask_id(self, part: str) -> int:
        return self.size(part) + 2

    def token(self, part: str, idx: int) -> str:
        prefix = {"body": "motion", "lhand": "hand", "rhand": "rhand"}[part]
        return f"<{prefix}_id_{int(idx)}>"

    def added_tokens(self) -> list[str]:
        tokens: list[str] = []
        for part in ("body", "lhand", "rhand"):
            tokens.extend(self.token(part, i) for i in range(self.size(part) + 3))
        return tokens

    def to_json_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "SokePartCodecs":
        return cls(
            body_size=int(data.get("body_size", 96)),
            lhand_size=int(data.get("lhand_size", 192)),
            rhand_size=int(data.get("rhand_size", 192)),
        )


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= int(limit):
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_codecs(code_root: str | Path) -> SokePartCodecs:
    path = Path(code_root) / "part_codecs.json"
    if path.exists():
        return SokePartCodecs.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
    return SokePartCodecs()


def write_codecs(code_root: str | Path, codecs: SokePartCodecs) -> None:
    path = Path(code_root) / "part_codecs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(codecs.to_json_dict(), indent=2), encoding="utf-8")


def load_instructions(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for task_group in payload.values():
        for task in task_group.values():
            if isinstance(task, dict) and task.get("input") and task.get("output"):
                tasks.append(task)
    if not tasks:
        raise ValueError(f"No SOKE-style instruction tasks found in {path}")
    return tasks


def render_template(
    template: str,
    *,
    caption: str,
    motion_text: str,
    num_frames: int,
    fps: float = 20.0,
    motion_predict_head: str = "",
    motion_predict_tail: str = "",
    motion_masked: str = "",
) -> str:
    seconds = math.floor(float(num_frames) / float(fps)) if fps else 0.0
    return (
        template.replace("<Caption_Placeholder>", str(caption))
        .replace("<Motion_Placeholder>", str(motion_text))
        .replace("<Frame_Placeholder>", str(int(num_frames)))
        .replace("<Framelen_Placeholder>", str(int(num_frames)))
        .replace("<Second_Placeholder>", f"{seconds:.1f}")
        .replace("<Motion_Placeholder_s1>", str(motion_predict_head))
        .replace("<Motion_Placeholder_s2>", str(motion_predict_tail))
        .replace("<Motion_Placeholder_Masked>", str(motion_masked))
    )


def soke_concat_tokens(codecs: SokePartCodecs, part: str, ids: Sequence[int]) -> str:
    return "".join(codecs.token(part, int(i)) for i in ids)


def soke_space_tokens(codecs: SokePartCodecs, part: str, ids: Sequence[int]) -> str:
    return " ".join(codecs.token(part, int(i)) for i in ids)


def flatten_triplet_tokens(
    codecs: SokePartCodecs,
    body_ids: Sequence[int],
    lhand_ids: Sequence[int],
    rhand_ids: Sequence[int],
    *,
    include_part_boundaries: bool = False,
) -> list[str]:
    n = min(len(body_ids), len(lhand_ids), len(rhand_ids))
    tokens: list[str] = []
    if include_part_boundaries:
        tokens.extend(
            [
                codecs.token("body", codecs.start_id("body")),
                codecs.token("lhand", codecs.start_id("lhand")),
                codecs.token("rhand", codecs.start_id("rhand")),
            ]
        )
    for b, lh, rh in zip(body_ids[:n], lhand_ids[:n], rhand_ids[:n]):
        tokens.extend(
            [
                codecs.token("body", int(b)),
                codecs.token("lhand", int(lh)),
                codecs.token("rhand", int(rh)),
            ]
        )
    if include_part_boundaries:
        tokens.extend(
            [
                codecs.token("body", codecs.end_id("body")),
                codecs.token("lhand", codecs.end_id("lhand")),
                codecs.token("rhand", codecs.end_id("rhand")),
            ]
        )
    return tokens


def flattened_target_text(
    codecs: SokePartCodecs,
    body_ids: Sequence[int],
    lhand_ids: Sequence[int],
    rhand_ids: Sequence[int],
    *,
    include_part_boundaries: bool = False,
) -> str:
    return " ".join(
        flatten_triplet_tokens(
            codecs,
            body_ids,
            lhand_ids,
            rhand_ids,
            include_part_boundaries=include_part_boundaries,
        )
    )


def soke_motion_placeholder_variants(
    codecs: SokePartCodecs,
    flat_triplet_tokens: Sequence[str],
    *,
    predict_ratio: float = 0.2,
    inbetween_ratio: float = 0.25,
) -> dict[str, str]:
    tokens = [str(tok) for tok in flat_triplet_tokens]
    triplet_len = max(1, len(tokens) // 3)
    predict_head = int(triplet_len * float(predict_ratio) + 1)
    masked_head = int(triplet_len * float(inbetween_ratio) + 1)
    masked_tail = int(triplet_len * (1.0 - float(inbetween_ratio)) + 1)
    predict_head = max(1, min(predict_head, triplet_len))
    masked_head = max(0, min(masked_head, triplet_len))
    masked_tail = max(masked_head, min(masked_tail, triplet_len))

    start_triplet = [
        codecs.token("body", codecs.start_id("body")),
        codecs.token("lhand", codecs.start_id("lhand")),
        codecs.token("rhand", codecs.start_id("rhand")),
    ]
    end_triplet = [
        codecs.token("body", codecs.end_id("body")),
        codecs.token("lhand", codecs.end_id("lhand")),
        codecs.token("rhand", codecs.end_id("rhand")),
    ]
    mask_triplet = [
        codecs.token("body", codecs.mask_id("body")),
        codecs.token("lhand", codecs.mask_id("lhand")),
        codecs.token("rhand", codecs.mask_id("rhand")),
    ]
    head_idx = predict_head * 3
    masked_head_idx = masked_head * 3
    masked_tail_idx = masked_tail * 3
    masked_repeats = max(1, masked_tail - masked_head)
    return {
        "motion_predict_head": " ".join(tokens[:head_idx] + end_triplet),
        "motion_predict_tail": " ".join(start_triplet + tokens[head_idx:]),
        "motion_masked": " ".join(tokens[:masked_head_idx] + mask_triplet * masked_repeats + tokens[masked_tail_idx:]),
    }


def crop_code_ids_like_soke(
    body_ids: Sequence[int],
    lhand_ids: Sequence[int],
    rhand_ids: Sequence[int],
    *,
    min_code_len: int = 10,
    max_code_len: int = 100,
    unit_length: int = 4,
    random_drop: bool = False,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int], list[int]]:
    n = min(len(body_ids), len(lhand_ids), len(rhand_ids))
    body = [int(x) for x in body_ids[:n]]
    lhand = [int(x) for x in lhand_ids[:n]]
    rhand = [int(x) for x in rhand_ids[:n]]
    if n == 0:
        return body, lhand, rhand
    if n < min_code_len:
        idx = [round(i * (n - 1) / max(min_code_len - 1, 1)) for i in range(min_code_len)]
        body = [body[i] for i in idx]
        lhand = [lhand[i] for i in idx]
        rhand = [rhand[i] for i in idx]
    elif n > max_code_len:
        idx = [round(i * (n - 1) / max(max_code_len - 1, 1)) for i in range(max_code_len)]
        body = [body[i] for i in idx]
        lhand = [lhand[i] for i in idx]
        rhand = [rhand[i] for i in idx]
    else:
        keep = (n // max(1, unit_length)) * max(1, unit_length)
        keep = max(1, keep)
        start = (n - keep) // 2
        body = body[start : start + keep]
        lhand = lhand[start : start + keep]
        rhand = rhand[start : start + keep]
    if random_drop and len(body) > 2:
        rng = rng or random
        if rng.choice([False, False, True]):
            if rng.choice([True, False]):
                body, lhand, rhand = body[:-1], lhand[:-1], rhand[:-1]
            else:
                body, lhand, rhand = body[1:], lhand[1:], rhand[1:]
    return body, lhand, rhand


class SokeGemmaCausalDataset:
    """SOKE-style dynamic row expansion for a single causal LM target stream."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tasks: Sequence[dict[str, Any]],
        codecs: SokePartCodecs,
        *,
        fps: float = 20.0,
        random_drop: bool = False,
        max_logical_rows: int = 0,
        seed: int = 42,
    ):
        self.rows = list(rows)
        self.tasks = list(tasks)
        self.codecs = codecs
        self.fps = float(fps)
        self.random_drop = bool(random_drop)
        self.max_logical_rows = int(max_logical_rows)
        self.seed = int(seed)
        if not self.rows:
            raise ValueError("SokeGemmaCausalDataset received no rows")
        if not self.tasks:
            raise ValueError("SokeGemmaCausalDataset received no instruction tasks")

    def __len__(self) -> int:
        n = len(self.rows) * len(self.tasks)
        if self.max_logical_rows > 0:
            return min(n, self.max_logical_rows)
        return n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data_idx = int(idx) % len(self.rows)
        task_idx = (int(idx) // len(self.rows)) % len(self.tasks)
        row = self.rows[data_idx]
        task = self.tasks[task_idx]
        rng = random.Random(self.seed + int(idx))
        body, lhand, rhand = crop_code_ids_like_soke(
            row["body_ids"],
            row["lhand_ids"],
            row["rhand_ids"],
            random_drop=self.random_drop,
            rng=rng,
        )
        flat_tokens = flatten_triplet_tokens(self.codecs, body, lhand, rhand)
        target = " ".join(flat_tokens)
        motion_variants = soke_motion_placeholder_variants(self.codecs, flat_tokens)
        input_template = rng.choice(list(task["input"]))
        output_template = rng.choice(list(task["output"]))
        caption = str(row["text"])
        soke_length = len(body)
        prompt = render_template(
            input_template,
            caption=caption,
            motion_text=target,
            num_frames=soke_length,
            fps=self.fps,
            **motion_variants,
        )
        rendered_target = render_template(
            output_template,
            caption=caption,
            motion_text=target,
            num_frames=soke_length,
            fps=self.fps,
            **motion_variants,
        )
        return {
            "prompt": prompt,
            "target": rendered_target,
            "caption": caption,
            "split": row.get("split", ""),
            "dataset": row.get("dataset", ""),
            "source_alias": row.get("source_alias", ""),
            "clip_id": row.get("clip_id", ""),
            "clip_key": row.get("clip_key", ""),
            "code_len": len(body),
            "target_token_count": len(rendered_target.split()),
            "task_idx": task_idx,
            "data_idx": data_idx,
        }


def estimate_logical_rows(code_root: str | Path, stage: str, instructions_root: str | Path) -> dict[str, int]:
    instructions = Path(instructions_root)
    if stage == "lm_pretrain":
        inst_path = instructions / "template_pretrain.json"
    else:
        inst_path = instructions / "template_instructions.json"
    tasks = load_instructions(inst_path)
    out: dict[str, int] = {}
    for split in ("train", "val", "test"):
        path = Path(code_root) / f"{split}_soke_motion_codes.jsonl"
        if path.exists():
            n = sum(1 for _ in path.open("r", encoding="utf-8"))
            out[split] = n * len(tasks)
    return out
