"""Run configuration and CLI parsing. Pure Python: no torch or transformers imports."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

TECHNIQUES = ("lora", "qlora", "dora", "rslora")
TASK = "banking77"

# Thesis settings (docs/PLAN.md Step 3).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


@dataclass(frozen=True)
class TrainConfig:
    model: str  # registry key or Hugging Face id
    technique: str = "lora"
    n: int = 1000
    seed: int = 42
    epochs: int = 3
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    effective_batch: int = 16
    micro_batch: int = 16  # per-forward batch; effective_batch % micro_batch must be 0
    max_length: int = 512  # ceiling; batches are padded dynamically, so this costs nothing
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    max_steps: Optional[int] = None  # smoke mode: stop after this many optimiser steps
    eval_limit: Optional[int] = None  # smoke mode: evaluate only the first k validation items
    eval_batch: int = 64
    max_new_tokens: int = 32
    data_seed: int = 42
    hub_repo: Optional[str] = None
    results: str = "runs/pilot.jsonl"
    predictions_dir: str = "runs/predictions"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        if self.technique not in TECHNIQUES:
            raise ValueError(f"technique must be one of {TECHNIQUES}, got {self.technique!r}")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.micro_batch < 1 or self.effective_batch % self.micro_batch != 0:
            raise ValueError(
                f"effective_batch ({self.effective_batch}) must be a multiple of "
                f"micro_batch ({self.micro_batch})"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    @property
    def grad_accum(self) -> int:
        return self.effective_batch // self.micro_batch

    def total_steps(self) -> int:
        """Optimiser steps for the whole run (last partial batch of an epoch counts as a step)."""
        per_epoch = -(-self.n // self.effective_batch)
        total = per_epoch * self.epochs
        return min(total, self.max_steps) if self.max_steps else total

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_id(model_id: str, technique: str, n: int, seed: int) -> str:
    """Unique, filesystem-safe id of one run."""
    return f"{model_id.split('/')[-1]}__{technique}__n{n}__seed{seed}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and evaluate one PEFT run on BANKING77.")
    p.add_argument("--model", required=True, help="registry key (e.g. qwen3-4b) or Hugging Face id")
    p.add_argument("--technique", choices=TECHNIQUES, default="lora")
    p.add_argument("--n", type=int, default=1000, help="labelled-example budget N")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=None, help="smoke mode: stop after k steps")
    p.add_argument("--eval-limit", type=int, default=None, help="smoke mode: first k val items")
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--hub-repo", default=None, help="private Hub repo 'user/name' for adapters")
    p.add_argument("--results", default="runs/pilot.jsonl")
    p.add_argument("--predictions-dir", default="runs/predictions")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    return p


def config_from_args(argv: Optional[List[str]] = None) -> TrainConfig:
    a = build_parser().parse_args(argv)
    return TrainConfig(
        model=a.model, technique=a.technique, n=a.n, seed=a.seed, epochs=a.epochs, lr=a.lr,
        effective_batch=a.effective_batch, micro_batch=a.micro_batch, max_length=a.max_length,
        max_steps=a.max_steps, eval_limit=a.eval_limit, eval_batch=a.eval_batch,
        hub_repo=a.hub_repo, results=a.results, predictions_dir=a.predictions_dir,
        checkpoint_dir=a.checkpoint_dir,
    )
