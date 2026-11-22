"""Tests for src/eval/schema.py: record validation, JSONL round trip, run alignment."""

import json

import pytest

from src.eval.schema import (
    AlignmentError,
    PredictionRecord,
    SchemaError,
    align_runs,
    load_predictions,
    write_predictions,
)


def rec(**overrides):
    base = dict(
        run_id="qwen-lora-n200-s1", system="Qwen/Qwen3-4B-Instruct-2507", arm="finetuned",
        technique="lora", task="banking77", n_budget=200, seed=1, item_id="t0",
        prediction="card_arrival", parsed="card_arrival", gold="card_arrival", parse_ok=True,
        confidence=0.9, latency_ms=12.5, input_tokens=40, output_tokens=3, extra={},
    )
    base.update(overrides)
    return PredictionRecord.from_dict(base)


def test_round_trip(tmp_path):
    records = [rec(item_id=f"t{i}") for i in range(3)]
    path = tmp_path / "p.jsonl"
    write_predictions(path, records)
    assert load_predictions(path) == records


@pytest.mark.parametrize(
    "overrides",
    [
        {"arm": "local"},
        {"technique": "zero_shot", "n_budget": 5},
        {"seed": True},  # bool is not an int here
        {"n_budget": -1},
        {"parse_ok": True, "parsed": None},
        {"parse_ok": False},  # parsed still set
        {"confidence": 1.5},
        {"input_tokens": 2.0},
        {"item_id": ""},
        {"gold": {1, 2}},  # not JSON-serialisable
    ],
)
def test_invalid_records_are_rejected(overrides):
    with pytest.raises(SchemaError):
        rec(**overrides)


def test_nullable_fields_and_empty_parse():
    r = rec(seed=None, confidence=None, latency_ms=None, input_tokens=None, output_tokens=None,
            parsed=[], parse_ok=True, technique="zero_shot", n_budget=0, arm="api")
    assert r.parsed == [] and r.parse_ok


def test_unknown_and_missing_fields():
    d = rec().to_dict()
    with pytest.raises(SchemaError, match="unknown"):
        PredictionRecord.from_dict({**d, "cost_usd": 0.1})
    d.pop("gold")
    with pytest.raises(SchemaError, match="missing"):
        PredictionRecord.from_dict(d)


def test_loader_reports_line_number(tmp_path):
    path = tmp_path / "p.jsonl"
    good = json.dumps(rec().to_dict())
    bad = json.dumps({**rec(item_id="t1").to_dict(), "arm": "x"})
    path.write_text(good + "\n" + bad + "\n")
    with pytest.raises(SchemaError, match=r"p\.jsonl:2"):
        load_predictions(path)


def test_duplicate_items_and_mixed_run_fields(tmp_path):
    path = tmp_path / "p.jsonl"
    with pytest.raises(SchemaError, match="duplicate"):
        write_predictions(path, [rec(), rec()])
    with pytest.raises(SchemaError, match="mixes seed"):
        write_predictions(path, [rec(), rec(item_id="t1", seed=2)])


def test_align_orders_items_and_checks_sets():
    a = [rec(run_id="A", item_id=i) for i in ("t2", "t0", "t1")]
    b = [rec(run_id="B", item_id=i, arm="api", technique="icl_retrieved", seed=None) for i in ("t1", "t2", "t0")]
    aligned = align_runs(a + b)
    assert aligned.item_ids == ["t0", "t1", "t2"]
    assert [r.item_id for r in aligned.runs["B"]] == ["t0", "t1", "t2"]
    assert aligned.gold == ["card_arrival"] * 3

    with pytest.raises(AlignmentError, match="1 extra items.*1 missing items"):
        align_runs(a + [rec(run_id="B", item_id=i) for i in ("t0", "t1", "t9")])
    with pytest.raises(AlignmentError, match="no records"):
        align_runs(a, run_ids=["A", "C"])


def test_align_rejects_gold_mismatch_and_mixed_tasks():
    a = [rec(run_id="A", item_id="t0")]
    with pytest.raises(AlignmentError, match="gold"):
        align_runs(a + [rec(run_id="B", item_id="t0", gold="other")])
    with pytest.raises(AlignmentError, match="tasks"):
        align_runs(a + [rec(run_id="B", item_id="t0", task="billsum")])
