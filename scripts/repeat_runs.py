#!/usr/bin/env python3
"""Run the identical final training run several times and report the run-to-run spread (review 06d, F5).

    python scripts/repeat_runs.py --times 3 --out-dir runs/repeat/b77_n200 -- \
        --model qwen3-4b --n 200 --sampling stratified --replicate 1 --frozen-arms runs/frozen_arms.json
    python scripts/repeat_runs.py --summarise-only --out-dir runs/repeat/b77_n200

Everything after `--` goes to `python -m src.train.train`; each repeat gets its own results,
predictions and checkpoint directory (the run id is the same by construction). Validation split only.
BillSum needs --gen-eval so that ROUGE-L exists.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent


def summarise(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Spread of repeated runs. All records must be the same configuration at the same commit."""
    if len(records) < 2:
        raise ValueError("need at least 2 repeats")
    for key in ("run_id", "git_commit", "lr", "stop_step", "pool_hash"):
        values = {json.dumps(r.get(key)) for r in records}
        if len(values) != 1:
            raise ValueError(f"repeats differ in {key}: {sorted(values)}")
    out: Dict[str, Any] = {"run_id": records[0]["run_id"], "git_commit": records[0]["git_commit"],
                           "repeats": len(records)}
    scores = {"accuracy": [r.get("accuracy") for r in records],
              "rouge_l": [(r.get("summary_eval") or {}).get("rouge_l") for r in records],
              "val_loss": [r.get("val_loss") for r in records]}
    for name, xs in scores.items():
        if all(x is not None for x in xs):
            out[name] = {"values": xs, "mean": statistics.fmean(xs), "sd": statistics.stdev(xs),
                         "range": max(xs) - min(xs)}
    return out


def read_repeats(out_dir: Path) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
            for p in sorted(out_dir.glob("repeat*/results.jsonl"))]


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.summarise_only:
        if not passthrough:
            ap.error("give the training arguments after --")
        for i in range(1, a.times + 1):
            d = a.out_dir / f"repeat{i}"
            if (d / "results.jsonl").exists():
                raise SystemExit(f"{d} already holds a result; use a new --out-dir")
            subprocess.run([sys.executable, "-m", "src.train.train", *passthrough,
                            "--results", str(d.resolve() / "results.jsonl"),
                            "--predictions-dir", str(d.resolve() / "predictions"),
                            "--checkpoint-dir", str(d.resolve() / "checkpoints")], cwd=ROOT, check=True)
    result = summarise(read_repeats(a.out_dir))
    (a.out_dir / "summary.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
