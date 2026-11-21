"""Seeding and environment capture (GPU, driver, CUDA, library versions, git commit)."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import os
import random
import subprocess
from typing import Any, Dict, Optional

LIBS = ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate", "huggingface_hub")


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy (if present), torch and CUDA. Call BEFORE the model is built."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _run(cmd: list) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def lib_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in LIBS:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def git_commit() -> str:
    """HEAD hash (+'-dirty' if uncommitted changes). GIT_COMMIT env overrides; 'unknown' without git."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    here = os.path.dirname(os.path.abspath(__file__))
    sha = _run(["git", "-C", here, "rev-parse", "HEAD"])
    if not sha:
        return "unknown"
    dirty = _run(["git", "-C", here, "status", "--porcelain"])
    return sha + ("-dirty" if dirty else "")


def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"gpu_name": None, "driver_version": None, "cuda_runtime": None}
    try:
        import torch
        info["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if drv:
        info["driver_version"] = drv.splitlines()[0]
    return info


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
