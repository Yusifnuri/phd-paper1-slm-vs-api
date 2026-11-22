"""Build the paired difference matrices d[item, replicate] and the six-contrast Holm table.

Input: per-item prediction JSONL files in the shared schema (src/eval/schema.py), one set for the
fine-tuned arm and one for an in-context arm (SLM-ICL or API-ICL). Output: ContrastMatrix objects
(d = FT - ICL, positive = FT better) and, via `analyze`, the confirmatory table of the plan (section 4):
crossed-model estimate and CIs from analysis/inference.py, Holm over six contrasts, decision labels
('FT better' / 'ICL better' / 'inconclusive'; never 'equivalent').

Record conventions (documented here; the workers who write the records must follow them):
  * replicate of a record = extra["replicate"] if present, else extra["pool_seed"] (replicate r has
    pool_seed r; src/train/tuning.py::replicate_seeds). Missing both -> error.
  * extra["pool_hash"] (sha256 of the pool indices, src/train/data.py::indices_hash) is REQUIRED and must
    be equal for the two arms of one replicate (same pool: plan section 1). The ICL CLI already writes it;
    fine-tuned records currently carry only extra["split"], so the fine-tuning evaluator must add
    "pool_seed" (or "replicate") and "pool_hash" before this code can run on real predictions.
  * metric "accuracy": score = 1.0 if parse_ok and parsed == gold else 0.0 (unparseable = wrong).
  * metric "rouge_l": score = extra["rouge_l"], ROUGE-L F1 as a float in [0, 1] (the summary-evaluation
    worker stores it there); missing, non-numeric or outside [0, 1] -> error.
  * a stochastic arm (D-17: two full runs per replicate) is declared with runs_per_replicate=2; the
    item score is the mean of its runs (0, 0.5, 1 for accuracy). Runs of one replicate are told apart by run_id.
Scores stay in fractions; report_tables multiplies by 100 for display (points).

Unblinded: this module computes the effect. The blinded variance step lives in blinded_variance.py;
the `build` command here prints no effect size.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import inference as inf  # noqa: E402
from src.eval.schema import PredictionRecord, check_run_consistency, load_predictions  # noqa: E402

METRICS = ("accuracy", "rouge_l")
NS = (200, 1000, 3000)
TASKS = ("banking77", "billsum")
MARGIN = {"banking77": 0.03}     # BillSum has no equivalence outcome (plan section 4)
NO_EQUALITY = {"billsum"}


class ContrastError(ValueError):
    """The two arms cannot be paired into a valid difference matrix."""


def cell_name(task: str, n: int) -> str:
    return f"{task}_N{n}"


@dataclass(frozen=True)
class ContrastMatrix:
    task: str
    n_budget: int
    metric: str
    item_ids: Tuple[str, ...]
    replicates: Tuple[int, ...]
    d: np.ndarray                      # [item, replicate], FT - ICL
    pool_hashes: Tuple[str, ...]       # per replicate, identical for both arms

    def save(self, path: str) -> None:
        np.savez(path, d=self.d, item_ids=np.array(self.item_ids), replicates=np.array(self.replicates),
                 pool_hashes=np.array(self.pool_hashes), task=self.task, n_budget=self.n_budget,
                 metric=self.metric)

    @classmethod
    def load(cls, path: str) -> "ContrastMatrix":
        with np.load(path, allow_pickle=False) as z:
            return cls(str(z["task"]), int(z["n_budget"]), str(z["metric"]),
                       tuple(str(x) for x in z["item_ids"]), tuple(int(x) for x in z["replicates"]),
                       np.array(z["d"]), tuple(str(x) for x in z["pool_hashes"]))


def replicate_of(r: PredictionRecord) -> int:
    v = r.extra.get("replicate", r.extra.get("pool_seed"))
    if v is None or isinstance(v, bool) or not isinstance(v, int):
        raise ContrastError(f"run {r.run_id!r} item {r.item_id!r}: extra needs an integer 'replicate' or "
                            f"'pool_seed', got {v!r}")
    return v


def score_of(r: PredictionRecord, metric: str) -> float:
    if metric == "accuracy":
        return 1.0 if (r.parse_ok and r.parsed == r.gold) else 0.0
    if metric == "rouge_l":
        v = r.extra.get("rouge_l")
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ContrastError(f"run {r.run_id!r} item {r.item_id!r}: extra['rouge_l'] missing or not a "
                                f"number ({v!r})")
        if not 0.0 <= v <= 1.0:
            raise ContrastError(f"run {r.run_id!r} item {r.item_id!r}: extra['rouge_l']={v} is outside [0, 1] "
                                f"(store ROUGE-L F1 as a fraction)")
        return float(v)
    raise ContrastError(f"unknown metric {metric!r}; use one of {METRICS}")


@dataclass
class _ArmRep:
    scores: Dict[str, float]
    pool_hash: str
    golds: Dict[str, str]


def _arm_by_replicate(records: Iterable[PredictionRecord], label: str, task: str, n_budget: int,
                      metric: str, runs_per_replicate: int, system: Optional[str],
                      technique: Optional[str]) -> Dict[int, _ArmRep]:
    recs = [r for r in records if r.task == task and r.n_budget == n_budget
            and (system is None or r.system == system) and (technique is None or r.technique == technique)]
    if not recs:
        raise ContrastError(f"{label}: no records for task={task!r} n_budget={n_budget} "
                            f"system={system!r} technique={technique!r}")
    try:
        check_run_consistency(recs)
    except ValueError as e:
        raise ContrastError(f"{label}: {e}") from e
    arms = {(r.arm, r.system, r.technique) for r in recs}
    if len(arms) != 1:
        raise ContrastError(f"{label}: records mix several arms/systems/techniques {sorted(arms)}; "
                            f"filter with system=/technique=")
    by_rep: Dict[int, Dict[str, List[PredictionRecord]]] = {}
    for r in recs:
        by_rep.setdefault(replicate_of(r), {}).setdefault(r.run_id, []).append(r)
    out: Dict[int, _ArmRep] = {}
    for rep, runs in sorted(by_rep.items()):
        if len(runs) != runs_per_replicate:
            raise ContrastError(f"{label}: replicate {rep} has {len(runs)} run(s) {sorted(runs)}, "
                                f"expected {runs_per_replicate}")
        hashes, item_sets, per_run, golds = set(), [], [], {}
        for rid, rr in runs.items():
            for r in rr:
                h = r.extra.get("pool_hash")
                if not isinstance(h, str) or not h:
                    raise ContrastError(f"{label}: run {rid!r} item {r.item_id!r} has no extra['pool_hash']")
                hashes.add(h)
                g = json.dumps(r.gold, sort_keys=True)
                if golds.setdefault(r.item_id, g) != g:
                    raise ContrastError(f"{label}: replicate {rep} item {r.item_id!r}: gold differs between runs")
            item_sets.append({r.item_id for r in rr})
            per_run.append({r.item_id: score_of(r, metric) for r in rr})
        if len(hashes) != 1:
            raise ContrastError(f"{label}: replicate {rep} mixes pool hashes {sorted(hashes)}")
        if any(s != item_sets[0] for s in item_sets):
            raise ContrastError(f"{label}: replicate {rep}: runs cover different items")
        out[rep] = _ArmRep({i: float(np.mean([pr[i] for pr in per_run])) for i in item_sets[0]},
                           hashes.pop(), golds)
    return out


def build_matrix(ft: Iterable[PredictionRecord], icl: Iterable[PredictionRecord], task: str, n_budget: int,
                 metric: str = "accuracy", expected_replicates: Optional[Sequence[int]] = None,
                 ft_runs: int = 1, icl_runs: int = 1, ft_system: Optional[str] = None,
                 ft_technique: Optional[str] = None, icl_system: Optional[str] = None,
                 icl_technique: Optional[str] = None) -> ContrastMatrix:
    """d[i, r] = score_FT - score_ICL, aligned on item_id and replicate; raises ContrastError on any
    mismatch (items, replicates, pool hashes, golds, duplicates, missing fields)."""
    if metric not in METRICS:
        raise ContrastError(f"unknown metric {metric!r}")
    a = _arm_by_replicate(ft, "fine-tuned arm", task, n_budget, metric, ft_runs, ft_system, ft_technique)
    b = _arm_by_replicate(icl, "ICL arm", task, n_budget, metric, icl_runs, icl_system, icl_technique)
    if set(a) != set(b):
        raise ContrastError(f"replicates differ: fine-tuned {sorted(a)} vs ICL {sorted(b)}")
    reps = sorted(a)
    if expected_replicates is not None and reps != sorted(expected_replicates):
        raise ContrastError(f"missing or extra replicates: found {reps}, expected {sorted(expected_replicates)}")
    if len(reps) < 2:
        raise ContrastError(f"need at least 2 replicates, found {reps}")
    items = sorted(a[reps[0]].scores)
    for arm_name, arm in (("fine-tuned", a), ("ICL", b)):
        for r in reps:
            if sorted(arm[r].scores) != items:
                miss = sorted(set(items) - set(arm[r].scores))[:5]
                extra = sorted(set(arm[r].scores) - set(items))[:5]
                raise ContrastError(f"{arm_name} replicate {r}: item set differs from the reference "
                                    f"(missing e.g. {miss}, extra e.g. {extra})")
        hs = [arm[r].pool_hash for r in reps]
        if len(set(hs)) != len(hs):
            raise ContrastError(f"{arm_name}: two replicates share a pool hash (pool not redrawn?)")
    for r in reps:
        if a[r].pool_hash != b[r].pool_hash:
            raise ContrastError(f"replicate {r}: pool hashes differ between arms "
                                f"({a[r].pool_hash[:12]} vs {b[r].pool_hash[:12]})")
        if metric == "accuracy":
            bad = [i for i in items if a[r].golds[i] != b[r].golds[i]]
            if bad:
                raise ContrastError(f"replicate {r}: gold differs between arms for items {bad[:5]}")
    d = np.array([[a[r].scores[i] - b[r].scores[i] for r in reps] for i in items], dtype=float)
    return ContrastMatrix(task, n_budget, metric, tuple(items), tuple(reps), d,
                          tuple(a[r].pool_hash for r in reps))


# ---------------------------------------------------------------------------------------
def analyze(matrices: Dict[Tuple[str, int], ContrastMatrix], mdgs: Optional[Dict[str, float]] = None,
            sensitivity_M: Optional[Dict[str, int]] = None, exclude_n200: bool = False,
            alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Six-contrast Holm table (four without N=200: the pre-registered sensitivity family, L-15).
    `mdgs`: optional {cell_name: MDG} frozen from the validation-split runs (D-12).
    `sensitivity_M`: optional {task: M} (training-set size) adding the finite-population-corrected
    unadjusted p and 95% CI beside the primary one (plan section 2)."""
    ns = tuple(n for n in NS if not (exclude_n200 and n == 200))
    want = {(t, n) for t in TASKS for n in ns}
    if set(matrices) != want:
        raise ContrastError(f"need exactly the contrasts {sorted(want)}; got {sorted(matrices)}")
    results, margins = {}, {}
    for (t, n), m in matrices.items():
        if (t, n) != (m.task, m.n_budget):
            raise ContrastError(f"matrix for {(t, n)} is labelled {(m.task, m.n_budget)}")
        want_metric = "accuracy" if t == "banking77" else "rouge_l"
        if m.metric != want_metric:
            raise ContrastError(f"{cell_name(t, n)}: metric {m.metric!r}, expected {want_metric!r}")
        results[cell_name(t, n)] = inf.crossed_contrast(m.d)
        margins[cell_name(t, n)] = MARGIN.get(t, math.nan)
    dec = inf.decide_family(results, margins, mdgs, alpha,
                            no_equality=[cell_name(t, n) for (t, n) in matrices if t in NO_EQUALITY],
                            expected_size=len(want))
    rows = []
    for (t, n) in sorted(matrices, key=lambda k: (TASKS.index(k[0]), k[1])):
        nm = cell_name(t, n)
        r, d = results[nm], dec[nm]
        row = dict(cell=nm, task=t, n_budget=n, metric=matrices[(t, n)].metric, n_items=r.n, R=r.R,
                   delta=r.mean, se=r.se, df=r.df, p=d["p"], p_holm=d["p_holm"],
                   ci95_low=r.ci_low, ci95_high=r.ci_high, ci99_2_low=d["ci99_2"][0], ci99_2_high=d["ci99_2"][1],
                   label=d["label"], practically_equal_descriptive=d["practically_equal_descriptive"],
                   underpowered_a_priori=d["underpowered_a_priori"], ci_halfwidth=d["ci_halfwidth"],
                   margin=MARGIN.get(t), s2_item=r.s2_item, s2_rep=r.s2_rep, s2_res=r.s2_res,
                   note=("stratified-pool cell; bracket by rule until the D-12 MDG exists"
                         if n == 200 and (mdgs is None or nm not in mdgs) else ""))
        if sensitivity_M and t in sensitivity_M:
            rc = inf.crossed_contrast(matrices[(t, n)].d, f=n / sensitivity_M[t])
            row.update(p_corrected_unadjusted=rc.p, ci95_corrected_low=rc.ci_low, ci95_corrected_high=rc.ci_high)
        rows.append(row)
    assert all(r["label"] in (inf.FT_BETTER, inf.ICL_BETTER, inf.INCONCLUSIVE) for r in rows)
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build one d-matrix (.npz) for a (task, N) cell; prints no effect size")
    b.add_argument("--ft", nargs="+", required=True)
    b.add_argument("--icl", nargs="+", required=True)
    b.add_argument("--task", choices=TASKS, required=True)
    b.add_argument("--n-budget", type=int, required=True)
    b.add_argument("--metric", choices=METRICS, default=None)
    b.add_argument("--replicates", type=int, nargs="+", default=None)
    b.add_argument("--ft-runs", type=int, default=1)
    b.add_argument("--icl-runs", type=int, default=1, help="2 for a stochastic arm (D-17)")
    b.add_argument("--out", required=True)
    t = sub.add_parser("table", help="six matrices -> Holm table (JSON); this step is unblinded")
    t.add_argument("matrices", nargs="+")
    t.add_argument("--exclude-n200", action="store_true")
    t.add_argument("--mdgs", default=None, help="JSON {cell: MDG} from the validation-split runs")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        metric = a.metric or ("accuracy" if a.task == "banking77" else "rouge_l")
        m = build_matrix(load_predictions(a.ft), load_predictions(a.icl), a.task, a.n_budget, metric,
                         a.replicates, a.ft_runs, a.icl_runs)
        m.save(a.out)
        print(f"wrote {a.out}: {a.task} N={a.n_budget} {metric}, {len(m.item_ids)} items x {len(m.replicates)} replicates")
        return 0
    ms = [ContrastMatrix.load(p) for p in a.matrices]
    mdgs = json.load(open(a.mdgs)) if a.mdgs else None
    rows = analyze({(m.task, m.n_budget): m for m in ms}, mdgs, {"banking77": 9003}, a.exclude_n200)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {a.out}: {len(rows)} contrasts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
