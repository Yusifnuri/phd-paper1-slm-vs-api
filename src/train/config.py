"""Run configuration and CLI parsing. Pure Python: no torch or transformers imports."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from .task_spec import MAX_NEW_TOKENS, TASK  # noqa: F401  (TASK re-exported)

TECHNIQUES = ("lora", "qlora", "dora", "rslora")
SAMPLING = ("random", "stratified")

# Thesis settings (docs/PLAN.md Step 3).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training protocol (docs/research/verification/05-design-redteam.md, "Training protocol").
S_MIN = 200                      # minimum optimiser steps per run
MIN_EPOCHS_AT_SMALL_N = 10       # batch is reduced at small N so S_MIN covers >= this many epochs
# Extended on the high side after the stage-2 pilot: the chosen lr was the old top value (4e-4) at
# N=200 and N=1,000 (gate G5; results/pilot-stage2-2026-11-22), per the protocol's edge rule.
BASE_LR_GRID = (5e-5, 1e-4, 2e-4, 4e-4, 8e-4)


def lr_grid(technique: str, r: int = LORA_R, alpha: int = LORA_ALPHA) -> Tuple[float, ...]:
    """Learning-rate grid per technique.

    rsLoRA scales the adapter by alpha/sqrt(r) instead of alpha/r, i.e. by sqrt(r) = 4 times more
    at r=16. Giving it the same lr would confound technique with effective step size, so its grid
    is the base grid divided by sqrt(r).
    """
    if technique == "rslora":
        return tuple(lr / (r ** 0.5) for lr in BASE_LR_GRID)
    return BASE_LR_GRID


def default_batch(n: int, max_batch: int = 16) -> int:
    """Effective batch: 16, reduced at small N so that S_MIN steps cover >= 10 epochs.

    epochs = S_MIN / ceil(n / B) >= 10  <=>  ceil(n / B) <= 20  <=>  B >= n / 20.
    N=50 -> 3, N=200 -> 10, N>=320 -> 16.
    """
    return max(1, min(max_batch, -(-n // (S_MIN // MIN_EPOCHS_AT_SMALL_N))))


@dataclass(frozen=True)
class TrainConfig:
    model: str  # registry key or Hugging Face id
    technique: str = "lora"
    n: int = 1000
    seed: int = 42        # training seed: adapter init, dropout, batch order
    pool_seed: int = 42   # which nested training pool (a replicate = pool_seed x seed)
    sampling: str = "random"  # "random" or "stratified" pools
    epochs: int = 3
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    effective_batch: Optional[int] = None  # None -> default_batch(n)
    micro_batch: Optional[int] = None      # None -> effective batch (one forward per step)
    s_min: int = S_MIN
    tune: bool = False    # replicate 0: one in-run early-stopped trial per lr, freeze the best (tuning.py)
    replicate: Optional[int] = None  # sets pool_seed = r, seed = 100 + r (protocol section 3)
    schedule_total: Optional[int] = None  # cosine schedule length; None -> total_steps()
    stop_step: Optional[int] = None       # stop training here (frozen arm); None -> end of schedule
    frozen_arms: Optional[str] = None     # tune: write here; final run: read lr/schedule/stop from here
    arm_sampling: Optional[str] = None    # use the arm frozen for this sampling (e.g. random-pool N=200 final)
    max_length: int = 1024  # ceiling: the shared prompt (77 labels) alone is ~500 tokens
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    max_steps: Optional[int] = None  # smoke mode: stop after this many optimiser steps
    eval_limit: Optional[int] = None  # smoke mode: evaluate only the first k validation items
    eval_batch: int = 64
    max_new_tokens: int = MAX_NEW_TOKENS
    hub_repo: Optional[str] = None
    results: str = "runs/pilot.jsonl"
    predictions_dir: str = "runs/predictions"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        if self.replicate is not None:
            from .tuning import replicate_seeds
            pool_seed, seed = replicate_seeds(self.replicate)
            object.__setattr__(self, "pool_seed", pool_seed)
            object.__setattr__(self, "seed", seed)
        if self.arm_sampling is not None and self.arm_sampling not in SAMPLING:
            raise ValueError(f"arm_sampling must be one of {SAMPLING}")
        if self.tune and self.replicate != 0:
            raise ValueError("--tune runs on replicate 0 only (pass --replicate 0)")
        if self.stop_step is not None and self.stop_step > self.schedule_length():
            raise ValueError("stop_step must not exceed the schedule length")
        if self.technique not in TECHNIQUES:
            raise ValueError(f"technique must be one of {TECHNIQUES}, got {self.technique!r}")
        if self.sampling not in SAMPLING:
            raise ValueError(f"sampling must be one of {SAMPLING}, got {self.sampling!r}")
        if self.n < 1 or self.epochs < 1:
            raise ValueError("n and epochs must be >= 1")
        if self.batch % self.micro != 0:
            raise ValueError(f"effective batch ({self.batch}) must be a multiple of micro_batch ({self.micro})")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    @property
    def batch(self) -> int:
        return self.effective_batch or default_batch(self.n)

    @property
    def micro(self) -> int:
        return self.micro_batch or self.batch

    @property
    def grad_accum(self) -> int:
        return self.batch // self.micro

    def steps_per_epoch(self) -> int:
        return -(-self.n // self.batch)

    def total_steps(self) -> int:
        """max(epochs * ceil(N/B), S_min); max_steps (smoke mode) overrides."""
        if self.max_steps:
            return self.max_steps
        return max(self.epochs * self.steps_per_epoch(), self.s_min)

    def schedule_length(self) -> int:
        """Steps of the cosine schedule (warm-up and decay are computed over this length)."""
        return self.schedule_total or self.total_steps()

    def steps_to_train(self) -> int:
        """Optimiser steps actually taken: stop_step if frozen, else the whole schedule."""
        if self.max_steps:
            return self.max_steps
        return self.stop_step or self.schedule_length()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_id(model_id: str, technique: str, n: int, seed: int, pool_seed: int = 42,
           sampling: str = "random") -> str:
    """Unique, filesystem-safe id of one run (a replicate is pool_seed x seed)."""
    tag = "" if sampling == "random" else "__strat"
    return f"{model_id.split('/')[-1]}__{technique}__n{n}__seed{seed}__pool{pool_seed}{tag}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and evaluate one PEFT run on BANKING77.")
    p.add_argument("--model", required=True, help="registry key (e.g. qwen3-4b) or Hugging Face id")
    p.add_argument("--technique", choices=TECHNIQUES, default="lora")
    p.add_argument("--n", type=int, default=1000, help="labelled-example budget N")
    p.add_argument("--seed", type=int, default=42, help="training seed")
    p.add_argument("--pool-seed", type=int, default=42, help="seed of the nested training pools")
    p.add_argument("--sampling", choices=SAMPLING, default="random")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--s-min", type=int, default=S_MIN, help="minimum optimiser steps")
    p.add_argument("--tune", action="store_true",
                   help="replicate 0 only: one early-stopped trial per lr, freeze the best arm")
    p.add_argument("--replicate", type=int, default=None,
                   help="replicate r: pool_seed=r, seed=100+r (overrides --seed/--pool-seed)")
    p.add_argument("--frozen-arms", default=None,
                   help="frozen_arms.json: written by --tune, read by final runs")
    p.add_argument("--arm-sampling", choices=SAMPLING, default=None,
                   help="final run: use the arm frozen for this sampling instead of --sampling")
    p.add_argument("--effective-batch", type=int, default=None, help="default: min(16, ceil(N/20))")
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--max-length", type=int, default=1024)
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
        model=a.model, technique=a.technique, n=a.n, seed=a.seed, pool_seed=a.pool_seed,
        sampling=a.sampling, epochs=a.epochs, lr=a.lr, s_min=a.s_min, tune=a.tune,
        replicate=a.replicate, frozen_arms=a.frozen_arms, arm_sampling=a.arm_sampling,
        effective_batch=a.effective_batch, micro_batch=a.micro_batch, max_length=a.max_length,
        max_steps=a.max_steps, eval_limit=a.eval_limit, eval_batch=a.eval_batch,
        hub_repo=a.hub_repo, results=a.results, predictions_dir=a.predictions_dir,
        checkpoint_dir=a.checkpoint_dir,
    )
