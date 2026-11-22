"""Offline tests for the kNN baseline: toy embedder and synthetic data, no network."""

import json

import numpy as np
import pytest

from src.baselines import knn
from src.train import tuning


def test_vote_weighted_and_tiebreaks():
    labels = ["a", "b", "b", "c"]
    row = np.array([0.9, 0.5, 0.45, 0.1])
    assert knn.vote(row, labels, 1) == "a"
    assert knn.vote(row, labels, 3) == "b"            # 0.95 > 0.9
    assert knn.vote(np.array([0.5, 0.5, 0.0, 0.0]), labels, 2) == "a"  # equal weight: nearest first
    assert knn.vote(np.array([0.3, 0.2, 0.1, -0.9]), ["a", "b", "b", "a"], 4) == "b"  # negatives count 0
    assert knn.top_k(np.array([0.1, 0.9, 0.9]), 2).tolist() == [1, 2]  # stable by pool order


def test_predict_shape_check_and_accuracy():
    with pytest.raises(ValueError):
        knn.predict(np.zeros((2, 3)), ["a", "b"], 1)
    assert knn.accuracy(["a", "b"], ["a", "c"]) == 0.5
    with pytest.raises(ValueError):
        knn.accuracy([], [])


def test_select_k_prefers_smaller_on_ties():
    sims = np.eye(3, dtype=np.float32)
    sel = knn.select_k(sims, ["a", "b", "c"], ["a", "b", "c"], ks=(1, 5, 16))
    assert sel["k"] == 1 and sel["val_accuracy"] == 1.0 and sel["at_grid_edge"] is None
    assert set(sel["per_k"]) == {1, 5, 16}


class ToyEmbedder:
    """Bag-of-characters vectors: deterministic, needs no model download."""
    name = "toy@0"

    def encode(self, texts):
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out


def _loader():
    words = {"card_arrival": "card arrive when", "exchange_rate": "exchange rate euro",
             "lost_or_stolen_card": "lost stolen phone"}
    texts, labels = [], []
    for j in range(1200):
        lab = list(words)[j % 3]
        texts.append(f"{words[lab]} {'x' * (j % 7)}")
        labels.append(lab)
    return texts, labels


def _run(tmp_path, *args):
    common = ["--frozen-arms", str(tmp_path / "arms.json"), "--results", str(tmp_path / "r.jsonl"),
              "--predictions-dir", str(tmp_path / "pred"), "--cache-dir", str(tmp_path / "cache")]
    return knn.run(list(args) + common, embedder=ToyEmbedder(), loader=_loader)


def test_run_tune_then_final_records_and_schema(tmp_path):
    t = _run(tmp_path, "--n", "200", "--sampling", "stratified", "--replicate", "0", "--tune")
    assert t["status"] == "tuned" and t["selection"]["k"] in knn.KS and t["selection"]["val_size"] == 200
    f = _run(tmp_path, "--n", "200", "--sampling", "stratified", "--replicate", "1")
    assert f["status"] == "ok" and f["n_eval"] == 1000 and f["accuracy"] > 0.9 and f["seed"] == 101
    assert f["pool_hash"] != t["pool_hash"]  # replicate 1 has a fresh pool
    recs = [json.loads(l) for l in (tmp_path / "pred" / f"{f['run_id']}.jsonl").read_text().splitlines()]
    assert len(recs) == 1000 and all(r["arm"] == "baseline" and r["technique"] == "knn" for r in recs)
    schema = pytest.importorskip("src.eval.schema")
    for r in recs[:20]:
        schema.PredictionRecord.from_dict(r)
    # random-pool final reusing the stratified arm; no random arm otherwise
    g = _run(tmp_path, "--n", "200", "--sampling", "random", "--replicate", "1", "--arm-sampling", "stratified")
    assert g["k"] == f["k"]
    with pytest.raises(KeyError):
        _run(tmp_path, "--n", "200", "--sampling", "random", "--replicate", "1")


def test_run_guards_replicates(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, "--n", "200", "--replicate", "1", "--tune")
    with pytest.raises(ValueError):
        _run(tmp_path, "--n", "200", "--replicate", "0")


def test_retune_conflicts_only_if_choice_changes(tmp_path):
    _run(tmp_path, "--n", "200", "--replicate", "0", "--tune")
    arms = tuning.load_arms(str(tmp_path / "arms.json"))
    assert len(arms) == 1 and next(iter(arms)).startswith("banking77|knn:toy@0|knn|n200|random")


def test_embedder_revision_is_pinned():
    from src.api.retrieval import DEFAULT_EMBEDDING_REVISION, SentenceTransformerEmbedder
    assert len(DEFAULT_EMBEDDING_REVISION) == 40
    assert SentenceTransformerEmbedder().name.endswith("@" + DEFAULT_EMBEDDING_REVISION)
