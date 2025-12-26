from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset

from finetune.utils import read_json


REQUIRED_KEYS = ("source", "target", "task")


@dataclass(frozen=True)
class LoadedData:
    raw: List[Dict[str, Any]]
    task_counts: Dict[str, int]


def _coerce_to_list(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    raise ValueError(
        "Training data JSON must be a list of dicts. "
        f"Got top-level type: {type(obj).__name__}"
    )


def load_training_json(path: str) -> LoadedData:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Expected your generator to have created it. "
            "If you used a custom output path, pass --data_path."
        )

    obj = read_json(p)
    data = _coerce_to_list(obj)

    if len(data) == 0:
        raise ValueError(f"Dataset is empty: {path}")

    bad = 0
    task_counter: Counter[str] = Counter()

    for i, ex in enumerate(data):
        if not isinstance(ex, dict):
            bad += 1
            continue
        missing = [k for k in REQUIRED_KEYS if k not in ex]
        if missing:
            bad += 1
            continue

        src = ex.get("source")
        tgt = ex.get("target")
        task = ex.get("task")

        if not isinstance(src, str) or not isinstance(tgt, str) or not isinstance(task, str):
            bad += 1
            continue

        if src.strip() == "" or tgt.strip() == "":
            bad += 1
            continue

        task_counter[task] += 1

    if bad > 0:
        raise ValueError(
            f"Dataset validation failed: {bad} invalid examples.\n"
            f"Each example must be a dict with string keys {REQUIRED_KEYS}, "
            "and non-empty 'source'/'target'."
        )

    return LoadedData(raw=data, task_counts=dict(task_counter))


def train_val_split(
    data: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1). Got {val_ratio}")

    n = len(data)
    n_val = max(1, int(n * val_ratio))
    if n_val >= n:
        raise ValueError(
            f"val_ratio too large: would allocate {n_val} validation samples out of {n} total."
        )

    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    val_idx = set(idx[:n_val])
    train = [data[i] for i in idx if i not in val_idx]
    val = [data[i] for i in idx if i in val_idx]
    return train, val


def to_trl_prompt_completion_dataset(data: List[Dict[str, Any]]) -> Dataset:
    """
    Convert your {source, target, task} into TRL prompt-completion format:

      {
        "prompt": [{"role":"user","content": source}],
        "completion": [{"role":"assistant","content": target}],
        "task": task
      }

    TRL SFTTrainer will apply the model's chat template automatically for conversational data.
    Loss is computed on completion tokens by default for prompt-completion datasets.
    """
    rows: List[Dict[str, Any]] = []
    for ex in data:
        rows.append(
            {
                "prompt": [{"role": "user", "content": ex["source"]}],
                "completion": [{"role": "assistant", "content": ex["target"]}],
                "task": ex["task"],
            }
        )
    return Dataset.from_list(rows)


def task_counts(data: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for ex in data:
        c[str(ex.get("task", "unknown"))] += 1
    return dict(c)
