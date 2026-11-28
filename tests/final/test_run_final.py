"""run_final end to end, offline: temporary repository, toy data, toy embedder and fake generator."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "icl"))
from icl_helpers import FakeGenerator, ToyEmbedder, toy_billsum_loader, toy_loader  # noqa: E402

from src.final import ledger, run_final as rf  # noqa: E402
from src.final.final_eval import TestIsolationError  # noqa: E402
from src.train import data  # noqa: E402

KNN_KEY = "banking77|knn:toy@0|knn|n200|stratified"
ICL_KEY = "banking77|fake/model|icl_retrieved|n200|random"
ARMS = {KNN_KEY: {"k": 5}, ICL_KEY: {"k": 8}}


@pytest.fixture
def frozen(tmp_path):
    p = tmp_path / "frozen_arms.json"
    p.write_text(json.dumps(ARMS), encoding="utf-8")
    return p


def factory(config, pool, adapter_dir, tmp):
    if config["technique"] == "knn":
        return rf.knn_predictor(config, pool, embedder=ToyEmbedder(), cache_dir=str(tmp / "emb"))
    return rf.icl_predictor(config, pool, embedder=ToyEmbedder(), generator=FakeGenerator(),
                            cache_dir=str(tmp / "emb"))


def dry(config, frozen, tmp_path, out="out"):
    return rf.dry_run(config, frozen, tmp_path / out, train_loader=toy_loader, limit=30,
                      predictor_factory=lambda c, p, a: factory(c, p, a, tmp_path))


def test_config_list_counts_replicates_and_random_pool_condition():
    cfgs = rf.list_configs(ARMS)
    knn = [c for c in cfgs if c["arm_key"] == KNN_KEY]
    assert len(knn) == 8 + 3 and sum(c["sampling"] == "random" for c in knn) == 3
    assert len([c for c in cfgs if c["arm_key"] == ICL_KEY]) == 8
    assert len({ledger.config_hash(c) for c in cfgs}) == len(cfgs)
    assert all(c["replicate"] >= 1 for c in cfgs)
    with pytest.raises(rf.FinalRunError):
        rf.make_config(KNN_KEY, ARMS[KNN_KEY], 0)


def test_written_configs_are_found_by_hash_prefix(frozen, tmp_path):
    out = tmp_path / "final_configs.json"
    assert rf.write_final_configs(frozen, out) == 19
    h = sorted(json.loads(out.read_text())["configs"])[0]
    assert ledger.config_hash(rf.lookup_config(out, h[:20])) == h
    with pytest.raises(rf.FinalRunError):
        rf.lookup_config(out, "")


@pytest.mark.parametrize("key", [KNN_KEY, ICL_KEY])
def test_dry_run_writes_records_the_contrast_builder_accepts(key, frozen, tmp_path):
    config = rf.make_config(key, ARMS[key], 1)
    res = dry(config, frozen, tmp_path)
    assert res["split"] == "validation-dryrun" and res["n_eval"] == 30 and 0 <= res["accuracy"] <= 1
    lines = [json.loads(x) for x in (tmp_path / "out" / "predictions" / res["predictions"]).read_text().splitlines()]
    pool = data.make_splits(1200, 1, sampling=config["sampling"], labels=toy_loader()[1]).pool(200)
    assert {r["extra"]["pool_hash"] for r in lines} == {data.indices_hash(pool)}
    assert all(r["extra"]["replicate"] == 1 and r["item_id"].startswith("banking77-heldout-") for r in lines)
    from src.eval.schema import PredictionRecord
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
    import build_contrasts as bc
    recs = [PredictionRecord.from_dict(r) for r in lines]
    assert bc.replicate_of(recs[0]) == 1 and bc.score_of(recs[0], "accuracy") in (0.0, 1.0)


def test_a_configuration_that_produced_a_result_cannot_run_again(frozen, tmp_path):
    config = rf.make_config(KNN_KEY, ARMS[KNN_KEY], 2)
    dry(config, frozen, tmp_path)
    with pytest.raises(rf.FinalRunError, match="already produced"):
        dry(config, frozen, tmp_path)


def test_gates_apply_and_nothing_is_read_without_tags(frozen, tmp_path):
    config = rf.make_config(KNN_KEY, ARMS[KNN_KEY], 1)
    repo = tmp_path / "bare"
    repo.mkdir()
    calls = []
    paths = rf.Paths(repo, repo / "final_configs.json", frozen, repo / "reads.jsonl",
                     repo / "fp.json", repo / "out")
    paths.final_configs.write_text("{}")
    with pytest.raises(TestIsolationError):
        rf.run_config(config, paths, loader=lambda: calls.append(1), train_loader=toy_loader,
                      predictor_factory=lambda c, p, a: factory(c, p, a, tmp_path))
    assert not calls and not paths.ledger.exists()


def test_billsum_icl_dry_run_scores_rouge_and_near_duplicates(tmp_path):
    pytest.importorskip("rouge_score")
    key = "billsum|fake/model|icl_retrieved|n200|random"
    arms = {key: {"k": 1, "retriever": "bm25"}}
    frozen = tmp_path / "fa.json"
    frozen.write_text(json.dumps(arms))

    class Gen:
        def generate(self, prompts, max_new_tokens, choices=None):
            from src.icl.generate import Generation
            return [Generation(text="water", input_tokens=5, output_tokens=1, finish_reason="stop") for _ in prompts]

    res = rf.dry_run(rf.make_config(key, arms[key], 1), frozen, tmp_path / "o", train_loader=toy_billsum_loader,
                     limit=5, predictor_factory=lambda c, p, a: rf.icl_predictor(
                         c, p, generator=Gen(), tokenizer=lambda t, **k: {"input_ids": t.split()}))
    rec = json.loads((tmp_path / "o" / "predictions" / res["predictions"]).read_text().splitlines()[0])
    assert "rouge_l" in res and 0 <= rec["extra"]["max_jaccard_to_pool"] <= 1


def test_adapter_manifest_must_match(tmp_path):
    key = "banking77|fake/model|lora|n200|stratified"
    arm = {"lr": 4e-4, "stop_step": 10}
    config = rf.make_config(key, arm, 1)
    pool = rf.load_pool(config, toy_loader)
    d = tmp_path / "ad"
    d.mkdir()
    with pytest.raises(rf.FinalRunError, match="run_manifest"):
        rf.check_adapter(d, config, pool)
    good = {"arm_key": key, "arm": arm, "replicate": 1, "pool_hash": data.indices_hash(pool.indices),
            "stop_step": 10}
    (d / "run_manifest.json").write_text(json.dumps({**good, "replicate": 2}))
    with pytest.raises(rf.FinalRunError, match="replicate"):
        rf.check_adapter(d, config, pool)
    (d / "run_manifest.json").write_text(json.dumps(good))
    (d / "adapter_model.safetensors").write_bytes(b"x")
    assert list(rf.check_adapter(d, config, pool)) == ["adapter_model.safetensors"]
