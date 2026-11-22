"""End-to-end offline run of the ICL CLI: toy data, toy embedder, fake generator."""

import json

import pytest

from src.icl import cli
from src.train import tuning
from icl_helpers import FakeGenerator, ToyEmbedder, toy_loader  # noqa: F401


def _run(tmp_path, gen, *args, embedder=None):
    common = ["--frozen-arms", str(tmp_path / "arms.json"), "--results", str(tmp_path / "r.jsonl"),
              "--predictions-dir", str(tmp_path / "pred"), "--cache-dir", str(tmp_path / "cache"),
              "--model", "qwen3-4b"]
    return cli.run(list(args) + common, embedder=embedder or ToyEmbedder(), loader=toy_loader, generator=gen)


HF = "Qwen/Qwen3-4B-Instruct-2507"


def test_tune_then_final_retrieved(tmp_path):
    g = FakeGenerator(min_shots=8)
    t = _run(tmp_path, g, "--technique", "icl_retrieved", "--n", "200", "--sampling", "stratified",
             "--replicate", "0", "--tune")
    key = tuning.arm_key("banking77", HF, "icl_retrieved", 200, "stratified")
    assert t["status"] == "tuned" and t["arm_key"] == key
    assert t["selection"]["k"] == 8 and t["selection"]["ks"] == [4, 8, 16, 32] and t["selection"]["val_size"] == 200
    assert t["selection"]["at_grid_edge"] is None
    assert set(t["selection"]["per_k"]) == {"4", "8", "16", "32"} and t["selection"]["per_k"]["4"] < 0.5
    assert len(g.calls) == 4 and all(len(c[0]) == 200 and c[2] is None for c in g.calls)  # 4 k values, free decoding
    assert tuning.load_arm(str(tmp_path / "arms.json"), key)["k"] == 8
    assert t["prompt_tokens"] > 0 and "generate_seconds" in t

    g2 = FakeGenerator(min_shots=8)
    f = _run(tmp_path, g2, "--technique", "icl_retrieved", "--n", "200", "--sampling", "stratified", "--replicate", "1")
    assert f["status"] == "ok" and f["k"] == 8 and f["n_eval"] == 1000 and f["accuracy"] == 1.0
    assert f["n_unparseable"] == 0 and f["seed"] == 101 and f["pool_hash"] != t["pool_hash"]
    assert f["frozen_arm"]["arm_key"] == key and f["decoding"] == "free" and f["eval_split"] == "validation"
    assert len(g2.calls) == 1 and g2.calls[0][1] == 32  # one pass, 32 new tokens like the fine-tuned arm

    recs = [json.loads(line) for line in (tmp_path / "pred" / f"{f['run_id']}.jsonl").read_text().splitlines()]
    assert len(recs) == 1000
    assert all(r["arm"] == "baseline" and r["technique"] == "icl_retrieved" and r["system"] == HF for r in recs)
    assert all(r["extra"]["arm_detail"] == "slm_icl" and r["extra"]["k"] == 8 and len(r["extra"]["shot_ids"]) == 8
               and r["extra"]["max_cosine_to_pool"] is not None for r in recs)
    schema = pytest.importorskip("src.eval.schema")
    parsed = [schema.PredictionRecord.from_dict(r) for r in recs]
    schema.check_run_consistency(parsed)
    lines = [json.loads(line) for line in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert [x["status"] for x in lines] == ["tuned", "ok"]


def test_random_arm_seeded_and_arm_key_includes_technique(tmp_path):
    g = FakeGenerator(min_shots=1)
    _run(tmp_path, g, "--technique", "icl_random", "--n", "200", "--replicate", "0", "--tune", "--ks", "1,2,3,4")
    arms = tuning.load_arms(str(tmp_path / "arms.json"))
    assert list(arms) == [tuning.arm_key("banking77", HF, "icl_random", 200, "random")]
    f = _run(tmp_path, FakeGenerator(min_shots=1), "--technique", "icl_random", "--n", "200", "--replicate", "1")
    recs = [json.loads(line) for line in (tmp_path / "pred" / f"{f['run_id']}.jsonl").read_text().splitlines()]
    assert all(r["seed"] == 101 and r["extra"]["max_cosine_to_pool"] is None for r in recs)
    f2 = _run(tmp_path, FakeGenerator(min_shots=1), "--technique", "icl_random", "--n", "200", "--replicate", "1",
              "--random-seed", "5")
    recs2 = [json.loads(line) for line in (tmp_path / "pred" / f"{f2['run_id']}.jsonl").read_text().splitlines()]
    assert recs2[0]["seed"] == 5 and [r["extra"]["shot_ids"] for r in recs] != [r["extra"]["shot_ids"] for r in recs2]


def test_zero_shot_final_needs_no_tuning_and_has_n_budget_0(tmp_path):
    g = FakeGenerator(min_shots=1, junk_every=4)
    f = _run(tmp_path, g, "--technique", "zero_shot", "--replicate", "1")
    assert f["status"] == "ok" and f["k"] == 0 and f["frozen_arm"] is None and f["n"] == 0
    assert f["n_unparseable"] == 250 and f["accuracy"] < 0.5  # junk counts as wrong, never dropped
    rec = json.loads((tmp_path / "pred" / f"{f['run_id']}.jsonl").read_text().splitlines()[0])
    assert rec["n_budget"] == 0 and rec["seed"] is None and rec["extra"]["shot_ids"] == []
    assert rec["parse_ok"] is False and rec["parsed"] is None
    assert not (tmp_path / "arms.json").exists()
    with pytest.raises(ValueError):
        _run(tmp_path, g, "--technique", "zero_shot", "--replicate", "0", "--tune")


def test_constrained_is_a_separate_labelled_run_on_the_frozen_arm(tmp_path):
    _run(tmp_path, FakeGenerator(), "--technique", "icl_retrieved", "--n", "200", "--replicate", "0", "--tune")
    g = FakeGenerator()
    c = _run(tmp_path, g, "--technique", "icl_retrieved", "--n", "200", "--replicate", "1", "--constrained")
    free = _run(tmp_path, FakeGenerator(), "--technique", "icl_retrieved", "--n", "200", "--replicate", "1")
    assert c["decoding"] == "constrained" and c["run_id"].endswith("__constrained") and c["run_id"] != free["run_id"]
    assert g.calls[0][2] is not None and "card_arrival" in g.calls[0][2] and len(g.calls[0][2]) == 77
    rec = json.loads((tmp_path / "pred" / f"{c['run_id']}.jsonl").read_text().splitlines()[0])
    assert rec["extra"]["decoding"] == "constrained"
    with pytest.raises(ValueError):
        _run(tmp_path, g, "--technique", "icl_retrieved", "--n", "200", "--replicate", "0", "--tune", "--constrained")


def test_guards(tmp_path):
    g = FakeGenerator()
    with pytest.raises(ValueError):  # final on the tuning replicate
        _run(tmp_path, g, "--technique", "icl_random", "--n", "200", "--replicate", "0")
    with pytest.raises(ValueError):  # tune on a later replicate
        _run(tmp_path, g, "--technique", "icl_random", "--n", "200", "--replicate", "1", "--tune")
    with pytest.raises(KeyError):  # final before tuning: never selected on the fly
        _run(tmp_path, g, "--technique", "icl_random", "--n", "200", "--replicate", "1")
    with pytest.raises(NotImplementedError):  # BillSum loader not wired
        _run(tmp_path, g, "--task", "billsum", "--technique", "icl_random", "--n", "200", "--replicate", "1")
    assert g.calls == []


def test_retune_does_not_spend_generation_and_never_reselects(tmp_path):
    args = ("--technique", "icl_retrieved", "--n", "200", "--replicate", "0", "--tune")
    _run(tmp_path, FakeGenerator(), *args)
    g = FakeGenerator(min_shots=1)
    again = _run(tmp_path, g, *args)
    assert again["status"] == "already_frozen" and g.calls == []
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 1


def test_arm_sampling_reuses_stratified_arm_for_random_pool(tmp_path):
    t = _run(tmp_path, FakeGenerator(), "--technique", "icl_retrieved", "--n", "200", "--sampling", "stratified",
             "--replicate", "0", "--tune")
    f = _run(tmp_path, FakeGenerator(), "--technique", "icl_retrieved", "--n", "200", "--sampling", "random",
             "--replicate", "1", "--arm-sampling", "stratified")
    assert f["k"] == t["selection"]["k"] and f["sampling"] == "random"
    assert f["run_id"] != t["run_id"]


def test_select_k_tie_break_and_edges():
    s = cli.select_k({4: 0.5, 8: 0.9, 16: 0.9, 32: 0.2})
    assert s["k"] == 8 and s["at_grid_edge"] is None
    assert cli.select_k({4: 0.9, 8: 0.9, 16: 0.1, 32: 0.1})["at_grid_edge"] == "low"
    assert cli.select_k({4: 0.1, 8: 0.2, 16: 0.3, 32: 0.4})["at_grid_edge"] == "high"
