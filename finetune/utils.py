from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class RunInfo:
    started_at_unix: int
    started_at_iso: str
    seed: int
    model_id: str
    data_path: str
    output_dir: str


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    # Ampere (8.0) and newer generally support bf16
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def pick_compute_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and cuda_bf16_supported():
        return torch.bfloat16
    return torch.float16


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def brief_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
    return info
