import json

import jsonschema
import pytest
from conftest import BagOfWordsEmbedder, FakeClient

from src.api.records import RECORD_FIELDS, read_records, validate_record
from src.api.run import BudgetExceeded, run_config, side_path


def _run(client, out, pool, test, ctx, **kw):
    args = dict(client=client, task="banking77", template="classification", context=ctx,
                technique="icl_retrieved", n_budget=len(pool), k=3, seed=None, pool=pool, test_items=test,
                out_path=out, budget_usd=1.0, embedder=BagOfWordsEmbedder(), poll_interval_s=0,
                sleep=lambda s: None)
    args.update(kw)
    return run_config(**args)


def test_sync_run_writes_valid_records(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    client = FakeClient(specs["anthropic"])
    summary = _run(client, out, toy_pool, toy_test, cls_context)
    recs = read_records(out)
    assert summary["written"] == len(toy_test) == len(recs) == client.calls
    for r in recs:
        validate_record(r)
        assert tuple(r) == RECORD_FIELDS
        assert r["arm"] == "api" and r["system"] == "claude-haiku-5-5"
        assert r["seed"] is None and r["parse_ok"] is True and r["parsed"] == "card_arrival"
        assert len(r["extra"]["retrieved_ids"]) == 3
        assert r["extra"]["model_id"] == "claude-haiku-5-5" and "run_date" in r["extra"]
    assert {r["item_id"] for r in recs} == {t.id for t in toy_test}
    assert side_path(out, ".estimate.json").exists()
    assert len(side_path(out, ".raw.jsonl").read_text().splitlines()) == len(toy_test)


def test_retrieved_shots_put_nearest_last(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    _run(FakeClient(specs["openai"]), out, toy_pool, toy_test[:1], cls_context)
    rec = read_records(out)[0]
    # Query "my new card has not arrived": p1 ("my card has not arrived yet") is the nearest.
    assert rec["extra"]["retrieved_ids"][-1] == "p1"


def test_resume_skips_finished_items(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    first = FakeClient(specs["google"])
    _run(first, out, toy_pool, toy_test[:2], cls_context)
    assert first.calls == 2
    second = FakeClient(specs["google"])
    summary = _run(second, out, toy_pool, toy_test, cls_context)
    assert second.calls == len(toy_test) - 2
    assert summary["skipped_done"] == 2
    third = FakeClient(specs["google"])
    _run(third, out, toy_pool, toy_test, cls_context)
    assert third.calls == 0
    ids = [r["item_id"] for r in read_records(out)]
    assert sorted(ids) == sorted(t.id for t in toy_test)  # no duplicates


def test_failed_item_is_retried_on_resume(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    flaky = FakeClient(specs["anthropic"], fail_ids=[toy_test[0].input])
    summary = _run(flaky, out, toy_pool, toy_test, cls_context, technique="zero_shot")
    assert summary["errors"] == 1 and summary["written"] == len(toy_test) - 1
    healthy = FakeClient(specs["anthropic"])
    _run(healthy, out, toy_pool, toy_test, cls_context, technique="zero_shot")
    assert healthy.calls == 1
    assert len(read_records(out)) == len(toy_test)


def test_torn_last_line_is_redone(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    _run(FakeClient(specs["anthropic"]), out, toy_pool, toy_test[:2], cls_context)
    with open(out, "a") as fh:
        fh.write('{"item_id": "t2", "trunc')
    c = FakeClient(specs["anthropic"])
    _run(c, out, toy_pool, toy_test[:3], cls_context)
    assert c.calls == 1


def test_budget_cap_blocks_run(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    client = FakeClient(specs["google"])
    with pytest.raises(BudgetExceeded):
        _run(client, out, toy_pool, toy_test, cls_context, budget_usd=1e-9)
    assert client.calls == 0 and client.submits == 0
    assert not out.exists()
    est = json.loads(side_path(out, ".estimate.json").read_text())
    assert est["gate_usd"] > est["budget_usd"]


def test_budget_gate_uses_batch_price_in_batch_mode(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    s = _run(FakeClient(specs["openai"]), out, toy_pool, toy_test, cls_context, dry_run=True, mode="batch")
    e = s["estimate"]
    assert e["gate_usd"] == pytest.approx(e["usd_batch_ceiling"])
    assert e["usd_batch_ceiling"] == pytest.approx(e["usd_standard_ceiling"] * 0.5)


def test_dry_run_makes_no_calls(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    client = FakeClient(specs["anthropic"])
    s = _run(client, out, toy_pool, toy_test, cls_context, dry_run=True)
    assert client.calls == 0 and s["dry_run"] and not out.exists()


def test_batch_mode_collects_out_of_order_results(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    client = FakeClient(specs["anthropic"])
    s = _run(client, out, toy_pool, toy_test, cls_context, mode="batch", technique="icl_random", seed=7)
    recs = read_records(out)
    assert client.submits == 1 and client.calls == 0 and s["written"] == len(toy_test)
    assert {r["item_id"] for r in recs} == {t.id for t in toy_test}
    for r in recs:
        validate_record(r)
        assert r["latency_ms"] is None and r["seed"] == 7 and r["technique"] == "icl_random"
    assert not side_path(out, ".batch.json").exists()


def test_resume_polls_in_flight_batch_instead_of_resubmitting(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    client = FakeClient(specs["anthropic"])
    # Simulate a crash after submission: a batch exists at the provider and in the state file.
    from src.api.prompts import build_prompt

    reqs = [(f"r{i}", build_prompt("classification", [], t.input, cls_context)) for i, t in enumerate(toy_test)]
    bid = client.batch_submit(reqs, 32)
    side_path(out, ".batch.json").write_text(json.dumps(
        {"batch_id": bid, "custom_ids": {f"r{i}": t.id for i, t in enumerate(toy_test)}}))
    s = _run(client, out, toy_pool, toy_test, cls_context, mode="batch", technique="zero_shot")
    assert client.submits == 1  # not resubmitted
    assert s["resumed_batch"] == bid
    assert len(read_records(out)) == len(toy_test)


def test_random_k_requires_seed(specs, toy_pool, toy_test, cls_context, tmp_path):
    with pytest.raises(ValueError, match="seed"):
        _run(FakeClient(specs["anthropic"]), tmp_path / "x.jsonl", toy_pool, toy_test, cls_context,
             technique="icl_random", seed=None)


def test_schema_rejects_bad_records(specs, toy_pool, toy_test, cls_context, tmp_path):
    out = tmp_path / "run.jsonl"
    _run(FakeClient(specs["anthropic"]), out, toy_pool, toy_test[:1], cls_context)
    good = read_records(out)[0]
    for mutate in (
        lambda r: r.pop("gold"),
        lambda r: r.update(arm="slm"),
        lambda r: r.update(technique="few_shot"),
        lambda r: r.update(confidence=1.5),
        lambda r: r.update(unexpected=1),
        lambda r: r.update(seed=3),  # seed only allowed for icl_random
    ):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(jsonschema.ValidationError):
            validate_record(bad)


def test_records_match_eval_schema_when_available(specs, toy_pool, toy_test, cls_context, tmp_path):
    schema = pytest.importorskip("src.eval.schema", reason="src/eval/schema.py not merged into this branch yet")
    out = tmp_path / "run.jsonl"
    _run(FakeClient(specs["anthropic"]), out, toy_pool, toy_test[:2], cls_context)
    validator = next((getattr(schema, n) for n in ("validate_record", "validate") if hasattr(schema, n)), None)
    if validator is None:
        pytest.skip("src.eval.schema exposes no validate_record/validate function")
    for r in read_records(out):
        validator(r)
