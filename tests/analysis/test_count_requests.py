"""The request counter reproduces the figures of the agent files under their stated readings."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("count_requests", ROOT / "analysis/count_requests.py")
cr = importlib.util.module_from_spec(spec)
sys.modules["count_requests"] = cr  # dataclasses need the module registered
spec.loader.exec_module(cr)


def row(rows, task, role):
    return next(r for r in rows if r["task"] == task and r["role"] == role)


def test_banking77_matches_a3_and_protocol():
    rows = cr.table()
    assert row(rows, "banking77", "selected")["total"] == 106_128   # A3, protocol section 3
    assert row(rows, "banking77", "other")["total"] == 50_688       # A3 and A5
    assert cr.totals(rows)["banking77"] == 207_504                  # protocol section 3


def test_billsum_default_is_a5_reading_with_protocol_validation_size():
    rows = cr.table()
    assert row(rows, "billsum", "selected")["total"] == 28_400      # A5 (validation min(1000, N), random-k)
    assert row(rows, "billsum", "other")["total"] == 19_400         # A5 22,400 minus random-k (3,000)
    assert cr.totals(rows)["billsum"] == 67_200


def test_billsum_alternative_readings_explain_the_disagreement():
    a5 = cr.table(random_k_other=True)
    assert row(a5, "billsum", "other")["total"] == 22_400 and cr.totals(a5)["billsum"] == 73_200
    a3 = cr.table(billsum_val=300)
    # A3 used 300 validation items at every N (3,600 per provider); the formula caps at N (3,200)
    assert row(a3, "billsum", "selected")["validation"] == 3_200
    assert 23_200 - row(a3, "billsum", "selected")["total"] == 400  # A3 = 23,200 with 3 x 300 items


def test_components_and_flags():
    sel = cr.count(cr.B77, True)
    assert sel["retrieved"] == 3080 * 8 * 3 and sel["validation"] == 4 * (200 + 1000 + 1000)
    assert sel["repeats"] == 1848 and sel["n77_exploratory"] == 0
    assert cr.count(cr.B77, True, with_n77=True)["n77_exploratory"] == 9240
    assert cr.count(cr.BILLSUM, True, with_n77=True)["n77_exploratory"] == 0  # BANKING77 only
    assert cr.count(cr.B77, True, random_k_selected=False)["random_k"] == 0
