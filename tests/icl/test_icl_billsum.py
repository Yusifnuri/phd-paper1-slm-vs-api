"""BillSum local-ICL runner, offline: toy bills, toy embedder, fake generator (no GPU, no network)."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.api.prompts import Example
from src.icl import cli, retrievers
from src.icl.render import get_task
from src.train import tuning
from icl_helpers import FakeSumGenerator, ToyEmbedder, toy_billsum_loader

HF = "Qwen/Qwen3-4B-Instruct-2507"


def _tok(text, add_special_tokens=False):  # whitespace "tokenizer" for the gold-length covariate
    return {"input_ids": text.split()}


def _run(tmp_path, gen, *args, embedder=None):
    common = ["--task", "billsum", "--frozen-arms", str(tmp_path / "arms.json"), "--results", str(tmp_path / "r.jsonl"),
              "--predictions-dir", str(tmp_path / "pred"), "--cache-dir", str(tmp_path / "cache"),
              "--model", "qwen3-4b"]
    return cli.run(list(args) + common, embedder=embedder or ToyEmbedder(), loader=toy_billsum_loader,
                   generator=gen, tokenizer=_tok)


KEY = tuning.arm_key("billsum", HF, "icl_retrieved", 200, "random")


def test_tune_selects_k_by_validation_rouge_l_then_final(tmp_path):
    g = FakeSumGenerator(min_shots=2)
    t = _run(tmp_path, g, "--technique", "icl_retrieved", "--n", "200", "--replicate", "0", "--tune")
    sel = t["selection"]
    assert t["status"] == "tuned" and t["arm_key"] == KEY
    assert sel["ks"] == [1, 2, 3] and sel["k"] == 2 and sel["val_size"] == 200  # val(N) = min(1000, N)
    assert sel["per_k"]["2"] == pytest.approx(1.0) and sel["per_k"]["3"] == pytest.approx(1.0)  # tie -> smaller k
    assert sel["per_k"]["1"] < 0.5 and sel["val_rouge_l"] == pytest.approx(1.0) and "val_accuracy" not in sel
    assert sel["metric"] == "rouge_l" and sel["retriever"] == "bge-head" and sel["at_grid_edge"] is None
    assert len(g.calls) == 3 and all(len(c[0]) == 200 and c[1] == 1024 and c[2] is None for c in g.calls)
    assert tuning.load_arm(str(tmp_path / "arms.json"), KEY)["k"] == 2

    g2 = FakeSumGenerator(min_shots=2, empty_every=10, cap_every=7)
    f = _run(tmp_path, g2, "--technique", "icl_retrieved", "--n", "200", "--replicate", "1")
    assert f["status"] == "ok" and f["k"] == 2 and f["n_eval"] == 1000 and f["retriever"] == "bge-head"
    assert len(g2.calls) == 1 and g2.calls[0][1] == 1024
    assert f["n_unparseable"] == 100 and f["n_hit_cap"] > 0 and f["eval_split"] == "validation"
    assert 0.8 < f["rouge_l"] < 0.95  # 10% empty outputs score 0, never dropped
    assert f["gen_tokens_median"] == 5 and f["token_ratio_mean"] is not None and f["pool_hash"] != t["pool_hash"]

    recs = [json.loads(x) for x in (tmp_path / "pred" / f"{f['run_id']}.jsonl").read_text().splitlines()]
    assert f["run_id"].startswith("billsum__") and len(recs) == 1000
    r0 = recs[0]
    assert r0["task"] == "billsum" and r0["arm"] == "baseline" and r0["item_id"].startswith("billsum-train-")
    assert r0["parse_ok"] is False and r0["parsed"] is None and r0["extra"]["rouge_l"] == 0.0
    r1 = recs[1]
    assert r1["parsed"] == r1["prediction"] == r1["gold"] and r1["extra"]["rouge_l"] == pytest.approx(1.0)
    assert r1["extra"]["gold_tokens"] == 4 and r1["extra"]["token_ratio"] == pytest.approx(5 / 4)
    assert r1["extra"]["pred_words"] == 4 and r1["extra"]["hit_cap"] is False
    assert r1["extra"]["k"] == 2 and len(r1["extra"]["shot_ids"]) == 2 and r1["extra"]["retriever"] == "bge-head"
    assert recs[7]["extra"]["hit_cap"] is True and recs[7]["extra"]["finish_reason"] == "length"
    schema = pytest.importorskip("src.eval.schema")
    schema.check_run_consistency([schema.PredictionRecord.from_dict(r) for r in recs])
    lines = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    assert [x["status"] for x in lines] == ["tuned", "ok"]


def test_bm25_retriever_is_frozen_and_finals_must_use_it(tmp_path):
    t = _run(tmp_path, FakeSumGenerator(min_shots=1), "--technique", "icl_retrieved", "--n", "200",
             "--replicate", "0", "--tune", "--retriever", "bm25", "--ks", "1,2")
    assert t["selection"]["retriever"] == "bm25" and t["retriever"] == "bm25" and t["embedder"] is None
    with pytest.raises(ValueError, match="frozen with retriever 'bm25'"):
        _run(tmp_path, FakeSumGenerator(), "--technique", "icl_retrieved", "--n", "200", "--replicate", "1",
             "--retriever", "tfidf")
    f = _run(tmp_path, FakeSumGenerator(min_shots=1), "--technique", "icl_retrieved", "--n", "200", "--replicate", "1")
    assert f["retriever"] == "bm25" and f["embedder"] is None


def test_compare_retrievers_flag_writes_one_line_and_never_generates(tmp_path):
    g = FakeSumGenerator()
    c = _run(tmp_path, g, "--technique", "icl_retrieved", "--n", "200", "--replicate", "0",
             "--compare-retrievers", "--compare-size", "40")
    assert g.calls == [] and c["status"] == "retriever_comparison" and c["eval_split"] == "validation"
    cmp = c["comparison"]
    assert set(cmp["retrievers"]) == {"bge-head", "bm25", "tfidf"} and cmp["n_queries"] == 40 and cmp["best"] in cmp["retrievers"]
    assert cmp["retrievers"]["bm25"]["topk"] > cmp["random_reference"]["topk"]  # topic words are shared
    assert not (tmp_path / "arms.json").exists()  # comparing freezes nothing
    assert [json.loads(x)["status"] for x in (tmp_path / "r.jsonl").read_text().splitlines()] == ["retriever_comparison"]
    with pytest.raises(ValueError):  # replicate 0 only
        _run(tmp_path, g, "--technique", "icl_retrieved", "--n", "200", "--replicate", "1", "--compare-retrievers")


def test_billsum_guards(tmp_path):
    g = FakeSumGenerator()
    with pytest.raises(ValueError):  # BillSum has no classes
        _run(tmp_path, g, "--technique", "icl_random", "--n", "200", "--replicate", "0", "--tune", "--sampling", "stratified")
    with pytest.raises(ValueError):
        _run(tmp_path, g, "--technique", "icl_random", "--n", "200", "--replicate", "1", "--constrained")
    with pytest.raises(ValueError):  # a retriever only exists for icl_retrieved
        _run(tmp_path, g, "--technique", "icl_random", "--n", "200", "--replicate", "0", "--tune", "--retriever", "bm25")
    with pytest.raises(KeyError):  # final before tuning
        _run(tmp_path, g, "--technique", "icl_retrieved", "--n", "200", "--replicate", "1")
    with pytest.raises(ValueError):  # BANKING77 keeps bge only
        cli.run(["--task", "banking77", "--technique", "icl_retrieved", "--n", "200", "--replicate", "0", "--tune",
                 "--retriever", "bm25", "--model", "qwen3-4b", "--frozen-arms", str(tmp_path / "a.json")])
    assert g.calls == []


def test_zero_shot_and_random_billsum_finals(tmp_path):
    z = _run(tmp_path, FakeSumGenerator(), "--technique", "zero_shot", "--replicate", "1", "--eval-limit", "30")
    assert z["k"] == 0 and z["n"] == 0 and z["n_eval"] == 30 and z["retriever"] is None
    rec = json.loads((tmp_path / "pred" / f"{z['run_id']}.jsonl").read_text().splitlines()[0])
    assert rec["n_budget"] == 0 and rec["extra"]["shot_ids"] == []
    _run(tmp_path, FakeSumGenerator(min_shots=1), "--technique", "icl_random", "--n", "200", "--replicate", "0", "--tune")
    f = _run(tmp_path, FakeSumGenerator(min_shots=1), "--technique", "icl_random", "--n", "200", "--replicate", "1")
    assert f["technique"] == "icl_random" and f["rouge_l"] == pytest.approx(1.0)


def test_default_max_model_len_fits_long_billsum_prompts():
    assert cli.default_max_model_len(get_task("banking77")) == 8192
    assert cli.default_max_model_len(get_task("billsum")) > 4 * 8000 + 1024


def test_runner_sources_never_reference_the_test_split():
    root = Path(__file__).resolve().parents[2] / "src"
    for f in (root / "icl" / "cli.py", root / "icl" / "retrievers.py", root / "train" / "summarise.py"):
        src = f.read_text()
        assert "load_us_test_subset_for_final_eval" not in src and "_US_TEST_SPLIT" not in src, f
        assert "final_eval" not in src.replace("no final_eval exists", ""), f


# ---------------------------------------------------------------- retrievers

POOL = ["tax credit for solar energy panels", "highway funding for rural roads and bridges",
        "school lunch program for children", "veterans health care benefits and hospitals"]


def test_tfidf_and_bm25_rank_the_topical_bill_first():
    queries = ["rural roads bridges funding", "solar energy tax", "hospitals for veterans"]
    for fn in (retrievers.tfidf_sims, retrievers.bm25_sims):
        m = fn(POOL, queries)
        assert m.shape == (3, 4) and list(m.argmax(axis=1)) == [1, 0, 3]
    assert retrievers.bm25_sims(POOL, ["zzz unseen"]).sum() == 0  # out-of-vocabulary only: no score
    t = retrievers.tfidf_sims(POOL, POOL)
    assert np.allclose(np.diag(t), 1.0, atol=1e-5)  # a text is identical to itself under cosine


def test_compare_retrievers_scores_retrieved_summaries_and_ties_go_to_bge_head():
    pool = [Example(f"p{i}", t, s) for i, (t, s) in enumerate(zip(POOL, ["solar tax", "road money", "school food", "vet care"]))]
    q = [Example("q0", "solar panels tax", "solar tax"), Example("q1", "roads and bridges", "road money")]
    good = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    bad = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    exact = lambda preds, golds: [float(p == g) for p, g in zip(preds, golds)]  # noqa: E731
    r = retrievers.compare_retrievers(pool, q, sims={"bge-head": bad, "bm25": good, "tfidf": good}, k=1, rouge_fn=exact)
    assert r["best"] == "bm25" and r["retrievers"]["bm25"]["top1"] == 1.0 and r["retrievers"]["bge-head"]["top1"] == 0.0
    tie = retrievers.compare_retrievers(pool, q, sims={"bge-head": good, "bm25": good, "tfidf": good}, k=1, rouge_fn=exact)
    assert tie["best"] == "bge-head"
    top2 = retrievers.compare_retrievers(pool, q, sims={"bge-head": good, "bm25": good, "tfidf": good}, k=2, rouge_fn=exact)
    assert top2["retrievers"]["bge-head"]["top1"] == 1.0 and top2["retrievers"]["bge-head"]["topk"] == 0.5
    with pytest.raises(ValueError):
        retrievers.compare_retrievers([], q, sims={})


def test_compare_retrievers_with_real_rouge_and_toy_embedder(tmp_path):
    pool = [Example(f"p{i}", f"{t} bill", f"This bill concerns {t.split()[0]}") for i, t in enumerate(POOL)]
    q = [Example("q", "highway rural roads bill", "This bill concerns highway")]
    r = retrievers.compare_retrievers(pool, q, embedder=ToyEmbedder(), cache_dir=str(tmp_path), k=1)
    assert r["retrievers"]["bm25"]["top1"] == 1.0 and r["retrievers"]["tfidf"]["top1"] == 1.0
