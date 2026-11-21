#!/usr/bin/env python3
"""Re-score saved prediction files with the shared parser; print old vs new accuracy.

    python scripts/rescore_predictions.py results/pilot-2026-11-21/predictions

The old pilot parser lower-cased the candidate but tested it against a case-sensitive label
set, so the mixed-case label Refund_not_showing_up could never be scored correct. The new
parser (src/train/task_spec.py, same as the API arm) fixes that. Nothing is rerun and no
file is modified. Old accuracy = the files' own parsed == gold; new = reparse of `prediction`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train.task_spec import parse_label  # noqa: E402


def rescore_file(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    old = sum(r["parsed"] == r["gold"] for r in rows)
    new = sum(parse_label(r["prediction"]) == r["gold"] for r in rows)
    unp_old = sum(not r["parse_ok"] for r in rows)
    unp_new = sum(parse_label(r["prediction"]) is None for r in rows)
    return len(rows), old / len(rows), new / len(rows), unp_old, unp_new


def main(argv):
    d = Path(argv[1] if len(argv) > 1 else "results/pilot-2026-11-21/predictions")
    files = sorted(d.glob("*.jsonl"))
    if not files:
        print(f"no .jsonl files in {d}")
        return 1
    deltas = []
    print(f"{'run':58s} {'n':>5s} {'old':>7s} {'new':>7s} {'delta':>7s} {'unparse old->new':>17s}")
    for f in files:
        n, old, new, uo, un = rescore_file(f)
        deltas.append(new - old)
        print(f"{f.stem:58s} {n:5d} {old:7.4f} {new:7.4f} {new - old:+7.4f} {uo:8d} ->{un:5d}")
    print(f"\n{len(files)} files; mean delta {sum(deltas) / len(deltas):+.4f}, max {max(deltas):+.4f}, "
          f"min {min(deltas):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
