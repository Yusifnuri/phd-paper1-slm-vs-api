"""Blinded variance-component re-estimation (decision D-09; plan section 4 and 9).

Reads a paired difference matrix d[item, replicate] (d = FT - ICL) and reports ONLY the variance
components, the minimal detectable gap implied by them, the power for the target gap at the configured
number of replicates, and whether the D-10 replicate-extension trigger fires. The mean difference and its
sign are never computed in reported form, never returned and never printed.

How blinding is enforced (structurally, not by convention):
  * the grand mean is subtracted at the very first step (`_centre`); every later computation sees only
    the centred matrix, so no code path in this module holds the effect;
  * `BlindedReport` is a frozen dataclass whose field list is fixed and contains no location statistic
    (the test suite inspects the field names, the function signatures and the printed output);
  * numeric outputs are rounded to 9 significant digits, so two matrices that differ only in their
    mean (for example +0.05 and -0.05) give identical output even after floating-point noise;
  * the loader reads only the array `d` from a file and ignores every other key (labels, metadata);
  * the module does not call inference.crossed_contrast / inference.variance_components (they return
    the mean); it uses only inference.power_at and inference.minimal_detectable_gap, which take
    variance components and return power or a gap.

D-10 trigger (BANKING77 only): extend R from the configured 8 to at most 10 iff the power to detect a
3-point gap, at the configured R and the first-step Holm level, is below 80% at N=1,000 or N=3,000.
N=200 is excluded (underpowered a priori, reported as a bracket); BillSum is capped at R=5 and never
triggers. The extension itself is a decision for the supervisor, this script only reports the rule.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference as _inf  # noqa: E402  (power_at, minimal_detectable_gap only)

TARGET_GAP = 0.03          # 3 points of accuracy, in the units of d (fractions)
POWER_THRESHOLD = 0.80
ALPHA = 0.05 / 6           # first-step Holm level used in the plan's power tables (~0.0083)
R_CONFIGURED = {"banking77": 8, "billsum": 5}
R_MAX = {"banking77": 10, "billsum": 5}
TRIGGER_CELLS = (1000, 3000)   # N=200 excluded by D-10
_SIG = 9


def _round(x: float) -> float:
    return float(f"{x:.{_SIG}g}") if math.isfinite(x) else x


def _centre(d: np.ndarray) -> np.ndarray:
    """Remove the grand mean immediately; nothing downstream can recover it."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 2:
        raise ValueError("d must be a 2-D array [items, replicates]")
    if d.shape[0] < 2 or d.shape[1] < 2:
        raise ValueError("need at least 2 items and 2 replicates")
    if not np.all(np.isfinite(d)):
        raise ValueError("d contains non-finite values")
    return d - d.mean()


def _components(c: np.ndarray) -> Dict[str, float]:
    """ANOVA components from a centred matrix (same estimator as inference.variance_components)."""
    n, R = c.shape
    ci, cr = c.mean(1), c.mean(0)
    ms_i = R * (ci ** 2).sum() / (n - 1)
    ms_r = n * (cr ** 2).sum() / (R - 1)
    res = c - ci[:, None] - cr[None, :]
    ms_e = (res ** 2).sum() / ((n - 1) * (R - 1))
    return dict(ms_item=ms_i, ms_rep=ms_r, ms_res=ms_e, s2_item=max((ms_i - ms_e) / R, 0.0),
                s2_rep=max((ms_r - ms_e) / n, 0.0), s2_res=ms_e)


@dataclass(frozen=True)
class BlindedReport:
    """Everything the blinded analyst may see. No location statistic is, or may become, a field."""
    task: Optional[str]
    n_budget: Optional[int]
    n_items: int
    R_observed: int
    s2_item: float
    s2_rep: float
    s2_res: float
    R_configured: int
    alpha: float
    target_gap: float
    mdg_at_R_configured: float
    power_target_at_R_observed: float
    power_target_at_R_configured: float
    R_max: int
    power_target_at_R_max: float
    trigger_applicable: bool
    trigger_fires: Optional[bool]
    note: str

    def as_dict(self) -> Dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def blinded_report(d: np.ndarray, task: Optional[str] = None, n_budget: Optional[int] = None,
                   r_configured: Optional[int] = None, r_max: Optional[int] = None,
                   target_gap: float = TARGET_GAP, alpha: float = ALPHA,
                   power_threshold: float = POWER_THRESHOLD) -> BlindedReport:
    """Variance components, MDG, power for `target_gap` and the D-10 trigger, from d alone."""
    c = _centre(d)
    n, R_obs = c.shape
    comp = _components(c)
    s_i, s_r, s_e = comp["s2_item"], comp["s2_rep"], comp["s2_res"]
    R_cfg = r_configured if r_configured is not None else R_CONFIGURED.get(task or "", R_obs)
    R_hi = r_max if r_max is not None else R_MAX.get(task or "", R_cfg)

    def pw(R: int) -> float:
        return _inf.power_at(target_gap, s_i, s_r, s_e, n, R, alpha)

    mdg = _inf.minimal_detectable_gap(s_i, s_r, s_e, n, R_cfg, alpha=alpha, power=power_threshold)
    p_cfg = pw(R_cfg)
    applicable = task == "banking77" and n_budget in TRIGGER_CELLS and R_hi > R_cfg
    if task == "banking77" and n_budget == 200:
        note = "N=200 excluded from the trigger (D-10): underpowered a priori, reported as a bracket"
    elif task == "billsum":
        note = "BillSum is capped at R=5; the trigger does not apply"
    elif not applicable:
        note = "trigger not evaluated: needs task=banking77, n_budget in (1000, 3000) and r_max > r_configured"
    else:
        note = "D-10: fires iff power for the target gap at the configured R is below the threshold"
    fires = bool(p_cfg < power_threshold) if applicable else None
    return BlindedReport(task=task, n_budget=n_budget, n_items=n, R_observed=R_obs,
                         s2_item=_round(s_i), s2_rep=_round(s_r), s2_res=_round(s_e),
                         R_configured=R_cfg, alpha=alpha, target_gap=target_gap,
                         mdg_at_R_configured=_round(mdg), power_target_at_R_observed=_round(pw(R_obs)),
                         power_target_at_R_configured=_round(p_cfg), R_max=R_hi,
                         power_target_at_R_max=_round(pw(R_hi)), trigger_applicable=applicable,
                         trigger_fires=fires, note=note)


def overall_trigger(reports: Sequence[BlindedReport]) -> Dict[str, object]:
    """BANKING77 R-extension: fires iff it fires in any applicable cell (N=1,000 or 3,000)."""
    app = [r for r in reports if r.trigger_applicable]
    return {"cells_evaluated": [r.n_budget for r in app],
            "cells_firing": [r.n_budget for r in app if r.trigger_fires],
            "extend_R_to_max": bool(any(r.trigger_fires for r in app)) if app else None}


def load_matrix(path: str) -> np.ndarray:
    """Read d[item, replicate] from .npy, .npz (key 'd' only; every other key is ignored) or .csv."""
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p, allow_pickle=False)
    if p.endswith(".npz"):
        with np.load(p, allow_pickle=False) as z:
            if "d" not in z.files:
                raise ValueError(f"{p}: no array 'd'")
            return np.array(z["d"])
    if p.endswith(".csv"):
        return np.loadtxt(p, delimiter=",", ndmin=2)
    raise ValueError(f"unsupported matrix file {p!r} (use .npy, .npz or .csv)")


def format_report(r: BlindedReport) -> str:
    lines = [f"cell: task={r.task} n_budget={r.n_budget}  items={r.n_items}  R_observed={r.R_observed}",
             f"variance components: item={r.s2_item:.6g}  replicate={r.s2_rep:.6g}  residual={r.s2_res:.6g}",
             f"alpha={r.alpha:.4g}; MDG at R={r.R_configured} (80% power): {r.mdg_at_R_configured:.6g}",
             f"power for gap {r.target_gap:g}: R_observed={r.power_target_at_R_observed:.4f}  "
             f"R_configured={r.power_target_at_R_configured:.4f}  R_max({r.R_max})={r.power_target_at_R_max:.4f}",
             f"D-10 trigger: {'n/a' if r.trigger_fires is None else ('FIRES' if r.trigger_fires else 'does not fire')}"
             f"  ({r.note})"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", nargs="+", help="d[item, replicate] files (.npy/.npz/.csv), one per cell")
    ap.add_argument("--task", choices=sorted(R_CONFIGURED), required=True)
    ap.add_argument("--n-budget", type=int, nargs="+", required=True, help="N of each matrix file, same order")
    ap.add_argument("--r-configured", type=int, default=None)
    ap.add_argument("--r-max", type=int, default=None)
    ap.add_argument("--target-gap", type=float, default=TARGET_GAP)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if len(a.matrix) != len(a.n_budget):
        ap.error("--n-budget needs one value per matrix file")
    reps = [blinded_report(load_matrix(m), a.task, n, a.r_configured, a.r_max, a.target_gap)
            for m, n in zip(a.matrix, a.n_budget)]
    ov = overall_trigger(reps)
    if a.json:
        print(json.dumps({"cells": [r.as_dict() for r in reps], "r_extension": ov}, indent=1))
    else:
        print("\n\n".join(format_report(r) for r in reps))
        print(f"\nR-extension (D-10, {a.task}): {ov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
