"""API request counts per provider role, task and cell: one committed formula (issue I-24, I-29).

    python analysis/count_requests.py                 # protocol default
    python analysis/count_requests.py --billsum-val 300 --no-random-k-other
    python analysis/count_requests.py --with-n77 --json

Design (docs/foundation/EXPERIMENT_PROTOCOL.md sections 3 and 5, decisions L-02..L-06):
  * test items: BANKING77 3,080, BillSum 1,000 (same items for every arm);
  * N in {200, 1000, 3000}; validation items per cell = min(val cap, N) (L-05) for each of the 4 k;
  * the confirmatory ("selected") provider runs R_sel replicates per N with retrieved examples
    (BANKING77 8, BillSum 5), the two other providers R_other = 3, no N=3000 shortcut;
  * BANKING77 only: 3 random-pool N=200 replicates for the random-vs-stratified table (L-04);
  * random-k control: N=1,000, R=3, on the selected provider (default) and, optionally, the others;
  * one zero-shot pass per provider;
  * flip-rate repeat: 10% of test items, 2 reruns, per N (A3 section 2).
Nothing here prices a request; tokens and dollars come from the API harness cost gate.
Request counts are the same under sync and batch modes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

NS = (200, 1000, 3000)
KS_TUNED = 4  # values of k tried on validation (L-05)


@dataclass(frozen=True)
class Task:
    name: str
    n_test: int
    r_selected: int
    r_other: int = 3
    random_pool_n200_reps: int = 0   # BANKING77 only (L-04)
    val_cap: int = 1000


B77 = Task("banking77", 3080, 8, random_pool_n200_reps=3)
BILLSUM = Task("billsum", 1000, 5)


def count(task: Task, selected: bool, *, random_k_selected: bool = True, random_k_other: bool = False,
          with_n77: bool = False, val_cap: int = None, repeat_frac: float = 0.10, repeat_runs: int = 2
          ) -> Dict[str, int]:
    """Requests by component for one provider role on one task."""
    cap = task.val_cap if val_cap is None else val_cap
    r = task.r_selected if selected else task.r_other
    retrieved = task.n_test * r * len(NS)
    random_pool = task.n_test * task.random_pool_n200_reps
    n77 = task.n_test * 3 if (with_n77 and task.name == "banking77") else 0
    rk = random_k_selected if selected else random_k_other
    random_k = task.n_test * 3 if rk else 0  # N=1,000, R=3
    zero = task.n_test
    validation = sum(min(cap, n) for n in NS) * KS_TUNED
    repeats = round(repeat_frac * repeat_runs * task.n_test * len(NS))
    parts = {"retrieved": retrieved, "random_pool_n200": random_pool, "n77_exploratory": n77,
             "random_k": random_k, "zero_shot": zero, "validation": validation, "repeats": repeats}
    parts["total"] = sum(parts.values())
    return parts


def table(**kw) -> List[Dict]:
    bill_kw = dict(kw)
    if "billsum_val" in bill_kw:
        bill_kw.pop("billsum_val")
    rows = []
    for task in (B77, BILLSUM):
        k = dict(kw)
        k["val_cap"] = kw.get("billsum_val") if task is BILLSUM and kw.get("billsum_val") else None
        k.pop("billsum_val", None)
        for role, selected in (("selected", True), ("other", False)):
            rows.append({"task": task.name, "role": role, **count(task, selected, **k)})
    return rows


def totals(rows: List[Dict]) -> Dict[str, int]:
    out = {}
    for task in ("banking77", "billsum"):
        sel = next(r["total"] for r in rows if r["task"] == task and r["role"] == "selected")
        oth = next(r["total"] for r in rows if r["task"] == task and r["role"] == "other")
        out[task] = sel + 2 * oth  # one selected provider, two others
    out["all"] = out["banking77"] + out["billsum"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--billsum-val", type=int, default=None,
                   help="BillSum validation cap (default min(1000, N); A3 used 300)")
    p.add_argument("--no-random-k-selected", action="store_true")
    p.add_argument("--random-k-other", action="store_true", help="random-k also on the two other providers")
    p.add_argument("--with-n77", action="store_true", help="BANKING77 N=77 R=3 exploratory API cells (A5)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    rows = table(billsum_val=a.billsum_val, random_k_selected=not a.no_random_k_selected,
                 random_k_other=a.random_k_other, with_n77=a.with_n77)
    tot = totals(rows)
    if a.json:
        print(json.dumps({"rows": rows, "totals_one_selected_two_other": tot}, indent=1))
        return 0
    cols = ["retrieved", "random_pool_n200", "n77_exploratory", "random_k", "zero_shot", "validation", "repeats", "total"]
    print("task       role      " + " ".join(f"{c[:11]:>11}" for c in cols))
    for r in rows:
        print(f"{r['task']:10} {r['role']:9} " + " ".join(f"{r[c]:>11,}" for c in cols))
    print(f"\nTotals (1 selected + 2 other providers): banking77 {tot['banking77']:,}  "
          f"billsum {tot['billsum']:,}  all {tot['all']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
