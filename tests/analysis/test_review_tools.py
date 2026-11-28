"""Repeat-run summary and the type-I / BillSum MDG helpers."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_repeat_summary_reports_spread_and_rejects_mixed_runs():
    rr = load("repeat_runs", "scripts/repeat_runs.py")
    base = {"run_id": "r", "git_commit": "c", "lr": 1e-4, "stop_step": 5, "pool_hash": "p", "val_loss": 0.5}
    s = rr.summarise([{**base, "accuracy": a} for a in (0.80, 0.82, 0.84)])
    assert s["accuracy"]["sd"] == pytest.approx(0.02) and s["accuracy"]["range"] == pytest.approx(0.04)
    assert "rouge_l" not in s
    with pytest.raises(ValueError, match="git_commit"):
        rr.summarise([{**base, "accuracy": 1}, {**base, "accuracy": 1, "git_commit": "d"}])


def test_billsum_mdg_interpolates_between_simulated_gaps():
    t1 = load("type1_check", "analysis/type1_check.py")
    mdg = t1.billsum_mdg(str(ROOT / "analysis/reference_outputs/sim2_billsum.json"))
    assert 0.006 < mdg["0.002"] < 0.008 and 0.008 < mdg["0.004"] < 0.010 and 0.020 < mdg["0.008"] < 0.030


def test_type1_runs_on_a_tiny_simulation():
    t1 = load("type1_check", "analysis/type1_check.py")
    rows = t1.type1(sims=5)
    assert len(rows) == 6 and all(0 <= r["rate_at_0.0500"] <= 1 for r in rows)
