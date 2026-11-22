"""Gated pilot, stage 2 (docs/foundation/EXPERIMENT_PROTOCOL.md section 8). Validation only.

    PYTHONUTF8=1 python3 scripts/run_stage2.py --hub-repo yusifnuri/paper1-pilot-adapters
    python3 scripts/run_stage2.py --dry-run          # print the plan
    python3 scripts/run_stage2.py --report-only      # gate report from existing results

Qwen3-4B, BANKING77, LoRA:
  1. tune on replicate 0: N=200 (stratified pool) and N=1,000 (random pool), 4 lr trials each;
  2. finals: replicates 1-2 at both N, plus one random-pool N=200 final (replicate 1) that uses
     the stratified N=200 arm;
  3. gate report G1, G4-G7. G2/G3 (vs kNN) need the kNN baseline, which is not implemented yet,
     so they are reported as NOT CHECKED, and the pilot cannot pass until they are.

Resumable: an arm already in frozen_arms.json is not re-tuned; a final already recorded "ok"
is skipped. Runs are sequential (never two at once: timing would be corrupted).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import task_spec, tuning  # noqa: E402
from src.train.config import run_id  # noqa: E402
from src.train.registry import resolve  # noqa: E402

# (n, sampling, replicate, arm_sampling) ; arm_sampling None = same as sampling
TUNES: List[Tuple[int, str]] = [(200, "stratified"), (1000, "random")]
FINALS: List[Tuple[int, str, int, Optional[str]]] = [
    (200, "stratified", 1, None), (200, "stratified", 2, None),
    (1000, "random", 1, None), (1000, "random", 2, None),
    (200, "random", 1, "stratified"),
]

# Thresholds (protocol section 8, table "Gates").
G1_MEAN, G1_MAX = 0.05, 0.08
G4_SLOPE = 0.08
G6_SPREAD = {200: 0.05, 1000: 0.03}
G7_RATIO = 1 / 3


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def base_cmd(model: str, n: int, sampling: str, replicate: int, a: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, "-m", "src.train.train", "--model", model, "--technique", "lora",
           "--n", str(n), "--sampling", sampling, "--replicate", str(replicate),
           "--frozen-arms", str(a.frozen_arms), "--results", str(a.results),
           "--predictions-dir", str(a.predictions_dir)]
    if a.hub_repo:
        cmd += ["--hub-repo", a.hub_repo]
    return cmd


def plan(a: argparse.Namespace) -> List[Tuple[str, List[str]]]:
    hf = resolve(a.model).hf_id
    arms = tuning.load_arms(str(a.frozen_arms))
    done = {r["run_id"] for r in read_rows(a.results) if r.get("status") == "ok"}
    steps: List[Tuple[str, List[str]]] = []
    for n, sampling in TUNES:
        key = tuning.arm_key(task_spec.TASK, hf, "lora", n, sampling)
        if key not in arms:
            steps.append((f"tune {key}", base_cmd(a.model, n, sampling, 0, a) + ["--tune"]))
    for n, sampling, r, arm_s in FINALS:
        pool_seed, seed = tuning.replicate_seeds(r)
        if run_id(hf, "lora", n, seed, pool_seed, sampling) in done:
            continue
        cmd = base_cmd(a.model, n, sampling, r, a)
        if arm_s:
            cmd += ["--arm-sampling", arm_s]
        steps.append((f"final n={n} {sampling} r={r}" + (f" (arm {arm_s})" if arm_s else ""), cmd))
    return steps


def gate_report(rows: List[Dict[str, Any]], arms: Dict[str, Any], hf: str) -> Dict[str, Any]:
    """Pure: gates from result rows and frozen arms. Values None = not enough data yet."""
    finals = [r for r in rows if r.get("status") == "ok" and r.get("model") == hf
              and r.get("frozen_arm") and r.get("technique") == "lora"]
    main = {200: [r for r in finals if r["n"] == 200 and r.get("sampling") == "stratified"],
            1000: [r for r in finals if r["n"] == 1000 and r.get("sampling") == "random"]}
    out: Dict[str, Any] = {}

    def acc(rs):
        return [r["accuracy"] for r in rs]

    inval = {n: [r["n_unparseable"] / r["n_eval"] for r in rs] for n, rs in main.items()}
    if all(inval.values()):
        means = {n: statistics.mean(v) for n, v in inval.items()}
        worst = max(x for v in inval.values() for x in v)
        out["G1"] = {"pass": all(m < G1_MEAN for m in means.values()) and worst <= G1_MAX,
                     "mean_invalid": means, "max_invalid": worst}
    else:
        out["G1"] = {"pass": None}
    out["G2"] = out["G3"] = {"pass": None, "note": "NOT CHECKED: kNN baseline not implemented"}
    if main[200] and main[1000]:
        slope = statistics.mean(acc(main[1000])) - statistics.mean(acc(main[200]))
        out["G4"] = {"pass": slope >= G4_SLOPE, "acc_1000_minus_200": slope}
    else:
        out["G4"] = {"pass": None}
    edges = {}
    for n, sampling in TUNES:
        key = tuning.arm_key(task_spec.TASK, hf, "lora", n, sampling)
        if key in arms:
            edges[n] = arms[key].get("at_grid_edge")
    out["G5"] = {"pass": (all(e is None for e in edges.values()) if len(edges) == len(TUNES) else None),
                 "edges": edges}
    spreads = {n: (max(acc(rs)) - min(acc(rs))) for n, rs in main.items() if len(rs) >= 2}
    out["G6"] = {"pass": (all(spreads[n] <= G6_SPREAD[n] for n in G6_SPREAD) if len(spreads) == 2 else None),
                 "spread": spreads}
    health = []
    for r in finals:
        first, last = r.get("first_train_loss"), r.get("final_train_loss")
        health.append(first is not None and last is not None and last <= first * G7_RATIO)
    out["G7"] = {"pass": (all(health) if health else None), "runs": len(health),
                 "note": "time-band part not checked automatically"}
    known = [g["pass"] for g in out.values()]
    out["overall"] = ("FAIL" if any(p is False for p in known)
                      else "INCOMPLETE" if any(p is None for p in known) else "PASS")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--hub-repo", default=None)
    p.add_argument("--frozen-arms", type=Path, default=ROOT / "runs/stage2/frozen_arms.json")
    p.add_argument("--results", type=Path, default=ROOT / "runs/stage2/results.jsonl")
    p.add_argument("--predictions-dir", type=Path, default=ROOT / "runs/stage2/predictions")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args(argv)

    if not a.report_only:
        steps = plan(a)
        print(f"{len(steps)} steps to run")
        for i, (label, cmd) in enumerate(steps, 1):
            print(f"\n=== [{i}/{len(steps)}] {label} ===\n{' '.join(cmd)}", flush=True)
            if a.dry_run:
                continue
            if subprocess.call(cmd, cwd=ROOT) != 0:
                print(f"!!! {label} failed; stopping (later steps depend on it)", flush=True)
                return 1
            if label.startswith("tune"):  # finals need the arm: re-plan after each tune
                return main(argv)
    if a.dry_run:
        return 0
    rep = gate_report(read_rows(a.results), tuning.load_arms(str(a.frozen_arms)), resolve(a.model).hf_id)
    print("\n" + json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
