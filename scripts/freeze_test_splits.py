#!/usr/bin/env python3
"""Freeze and hash the test splits (D-03): BANKING77 test, the seeded BillSum subset, LEDGAR placeholder.

    python scripts/freeze_test_splits.py            # writes results/test_split_fingerprint.json
    python scripts/freeze_test_splits.py --check    # later: verify nothing changed

Writes only sizes, seeds, dataset revisions and sha256 fingerprints, never rows. It prints no test
text. Together with scripts/freeze_test_split.py (the older BANKING77-only script) and
src/train/billsum.py (guarded reader) it is one of the few places that touch a test split
(scripts/verify_test_isolation.py lists them).
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.train import billsum, data  # noqa: E402

OUT = ROOT / "results" / "test_split_fingerprint.json"
LEDGAR_SEED, LEDGAR_SIZE = 42, 3000  # D-03


def _revision(dataset_id: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset_id).sha


def banking77_entry() -> dict:
    from datasets import load_dataset

    rev = _revision(data.DATASET_ID)
    ds = load_dataset(data.DATASET_ID, split="test", revision=rev)
    if len(ds) != data.N_TEST:
        raise SystemExit(f"expected {data.N_TEST} BANKING77 test rows, got {len(ds)}")
    names = ds.features["label"].names
    if list(names) != data.BANKING77_LABELS:
        raise SystemExit("BANKING77 label names differ from task_spec.BANKING77_LABELS")
    return {"dataset": data.DATASET_ID, "revision": rev, "split": "test", "n": len(ds),
            "sha256": data.rows_fingerprint(list(ds["text"]), [names[i] for i in ds["label"]]),
            "hash_definition": "src.train.data.rows_fingerprint over (text, label name) rows in original order"}


def billsum_entry() -> dict:
    rev_before = _revision(billsum.DATASET_ID)
    sub = billsum.load_us_test_subset_for_final_eval()  # the guarded reader, as final_eval will call it
    rev_after = _revision(billsum.DATASET_ID)
    if rev_before != rev_after:
        raise SystemExit("BillSum dataset revision changed during the download; rerun")
    if len(sub["indices"]) != billsum.TEST_SUBSET_SIZE:
        raise SystemExit("unexpected BillSum subset size")
    return {"dataset": billsum.DATASET_ID, "revision": rev_after, "split": "test",
            "n_full_test": sub["n_full_test"], "subset_seed": billsum.TEST_SUBSET_SEED,
            "n": len(sub["indices"]), "sha256": sub["hash"],
            "indices_sha256": hashlib.sha256(",".join(map(str, sub["indices"])).encode()).hexdigest(),
            "hash_definition": "src.train.billsum.rows_hash over (text, summary) rows of the seeded subset, original order",
            "note": "final_eval's guarded reader loads the default revision; this revision is the one hashed here"}


def ledgar_entry() -> dict:
    return {"status": "PLACEHOLDER_NOT_FROZEN", "n": LEDGAR_SIZE, "subset_seed": LEDGAR_SEED,
            "sha256": None, "role": "exploratory (D-03)",
            "note": "No LEDGAR loader exists in this repository yet; nothing was downloaded or hashed."}


def build() -> dict:
    return {"schema": 1, "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "banking77": banking77_entry(), "billsum": billsum_entry(), "ledgar": ledgar_entry()}


def _stable(fp: dict) -> dict:
    return {k: v for k, v in fp.items() if k != "created_utc"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fp = build()
    if a.check:
        want = json.loads(OUT.read_text())
        if _stable(want) != _stable(fp):
            raise SystemExit("TEST SPLIT FINGERPRINT CHANGED")
        print("test splits match the frozen fingerprints")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(fp, indent=2) + "\n")
        print("wrote", OUT)
        for k in ("banking77", "billsum"):
            print(k, fp[k]["n"], fp[k]["sha256"], fp[k]["revision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
