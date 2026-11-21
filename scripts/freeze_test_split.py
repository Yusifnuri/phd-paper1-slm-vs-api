#!/usr/bin/env python3
"""Freeze and hash the BANKING77 test split. The ONLY code in this repo that reads it.

Run once, by the coordinator, when the test set is to be frozen (not during the pilot):

    python scripts/freeze_test_split.py            # writes results/test_split_fingerprint.json
    python scripts/freeze_test_split.py --check    # later: verify nothing changed

Writes only the size and a sha256 over (text, label) rows, never the rows themselves.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import data  # noqa: E402

OUT = Path("results/test_split_fingerprint.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset(data.DATASET_ID, split="test")
    names = ds.features["label"].names
    fp = {"dataset": data.DATASET_ID, "split": "test", "n": len(ds),
          "sha256": data.rows_fingerprint(list(ds["text"]), [names[i] for i in ds["label"]])}
    if len(ds) != data.N_TEST:
        raise SystemExit(f"expected {data.N_TEST} test rows, got {len(ds)}")
    if a.check:
        want = json.loads(OUT.read_text())
        if want != fp:
            raise SystemExit(f"TEST SPLIT CHANGED: {want} != {fp}")
        print("test split matches the frozen fingerprint")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(fp, indent=2) + "\n")
        print("wrote", OUT, fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
