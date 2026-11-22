"""Tuned training procedure: pure selection logic and the frozen-arm file. No torch imports.

Protocol (docs/foundation/EXPERIMENT_PROTOCOL.md section 4, L-05, L-17, L-18):
  * Tuning runs on replicate 0 only (pool_seed 0, train seed 100), which is never analysed.
  * One trial per learning rate in the technique's grid (5 trials, L-25). Each trial is ONE cosine
    schedule to E_max epochs (N < 3,000: 10; N >= 3,000: 5), floored at S_min optimiser steps.
  * Validation on val(N) = the first min(1,000, N) rows of the fixed validation carve-out, at
    five evenly spaced checkpoints; early stopping with patience 2 inside the run (this replaces
    the old retrain-per-(lr, epochs) search, A5-12).
  * Score: validation accuracy, validation loss as the tie-break (classification); validation
    loss only for summarisation.
  * The chosen (lr, schedule_steps, stop_step) is frozen in frozen_arms.json. A final run on
    replicate r >= 1 trains the same cosine schedule and stops at stop_step, i.e. it reproduces
    exactly the procedure of the selected checkpoint, on a fresh pool.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VAL_CAP = 1000
N_CHECKPOINTS = 5
PATIENCE = 2
TUNING_REPLICATE = 0
METRICS = ("accuracy", "loss")  # accuracy: classification; loss: summarisation


class FrozenArmConflict(RuntimeError):
    """An arm is already frozen with a different configuration."""


def replicate_seeds(r: int) -> Tuple[int, int]:
    """Replicate r -> (pool_seed, train_seed) = (r, 100 + r)."""
    if r < 0:
        raise ValueError("replicate must be >= 0")
    return r, 100 + r


def max_epochs(n: int) -> int:
    """E_max of the tuning schedule: 10 below N=3,000, 5 from N=3,000."""
    return 10 if n < 3000 else 5


def val_size(n: int, cap: int = VAL_CAP) -> int:
    """val(N) = min(cap, N): the validation budget grows with the labelled budget."""
    return min(cap, n)


def schedule_steps(n: int, batch: int, s_min: int, e_max: Optional[int] = None) -> int:
    """Length of the cosine schedule in optimiser steps: max(E_max * ceil(N/B), S_min)."""
    e = e_max if e_max is not None else max_epochs(n)
    return max(e * -(-n // batch), s_min)


def checkpoint_steps(total: int, k: int = N_CHECKPOINTS) -> List[int]:
    """k evenly spaced checkpoints, the last one at the end of the schedule."""
    if total < 1 or k < 1:
        raise ValueError("total and k must be >= 1")
    return sorted({max(1, round(total * (i + 1) / k)) for i in range(k)})


def score(ck: Dict[str, Any], metric: str) -> Tuple[float, ...]:
    """Sort key, larger is better. Non-finite values rank last."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    loss = ck.get("val_loss")
    neg_loss = -loss if loss is not None and math.isfinite(loss) else -math.inf
    if metric == "loss":
        return (neg_loss,)
    acc = ck.get("accuracy")
    acc = acc if acc is not None and math.isfinite(acc) else -math.inf
    return (acc, neg_loss)


class EarlyStopper:
    """Feed checkpoint results in order; .update() returns True when training should stop."""

    def __init__(self, metric: str, patience: int = PATIENCE):
        self.metric, self.patience = metric, patience
        self.best: Optional[Tuple[float, ...]] = None
        self.bad = 0

    def update(self, ck: Dict[str, Any]) -> bool:
        s = score(ck, self.metric)
        if self.best is None or s > self.best:
            self.best, self.bad = s, 0
        else:
            self.bad += 1
        return self.bad >= self.patience


def select(trials: Sequence[Dict[str, Any]], grid: Sequence[float], metric: str,
           steps_per_epoch: int) -> Dict[str, Any]:
    """Best checkpoint over all trials.

    trials: [{"lr", "schedule_steps", "status", "checkpoints": [{"step", "accuracy", "val_loss"}]}].
    A trial with status other than "ok" (e.g. non-finite loss) never wins. Ties keep the earlier
    trial in grid order and the earlier checkpoint (fewer steps).
    Returns the frozen configuration plus a grid-edge flag (pilot gate G5).
    """
    best, best_s = None, None
    for t in trials:
        if t.get("status", "ok") != "ok":
            continue
        for ck in t["checkpoints"]:
            s = score(ck, metric)
            if s[0] == -math.inf:
                continue
            if best_s is None or s > best_s:
                best, best_s = (t, ck), s
    if best is None:
        raise RuntimeError("no trial produced a finite validation score")
    t, ck = best
    lr = t["lr"]
    edge = None
    if len(grid) > 1 and lr == min(grid):
        edge = "low"
    elif len(grid) > 1 and lr == max(grid):
        edge = "high"
    return {
        "lr": lr,
        "schedule_steps": t["schedule_steps"],
        "stop_step": ck["step"],
        "epochs_star": ck["step"] / steps_per_epoch,
        "metric": metric,
        "val_accuracy": ck.get("accuracy"),
        "val_loss": ck.get("val_loss"),
        "at_grid_edge": edge,
        "n_trials": len(trials),
    }


# ---------------------------------------------------------------- frozen_arms.json

def arm_key(task: str, model: str, technique: str, n: int, sampling: str) -> str:
    return f"{task}|{model}|{technique}|n{n}|{sampling}"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)


def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arms(path: str) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def freeze_arm(path: str, key: str, entry: Dict[str, Any]) -> str:
    """Add one frozen arm. Re-freezing the same key with a different entry raises.

    Writes atomically (temp file + rename) and returns the sha256 of the new file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive lock around read-modify-write so concurrent tuning jobs cannot lose each other's arms.
    with open(str(p) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        arms = load_arms(path)
        if key in arms:
            if _canonical(arms[key]) != _canonical(entry):
                raise FrozenArmConflict(
                    f"{key} is already frozen in {path} with a different configuration; "
                    "frozen arms are never re-selected (protocol section 4)")
            return file_hash(path)
        arms[key] = entry
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".frozen_arms.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_canonical(arms) + "\n")
        os.replace(tmp, p)
        return file_hash(path)


def load_arm(path: str, key: str) -> Dict[str, Any]:
    arms = load_arms(path)
    if key not in arms:
        raise KeyError(f"{key} is not frozen in {path}; run --tune on replicate 0 first")
    return arms[key]
