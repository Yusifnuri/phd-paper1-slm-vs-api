"""Tests for src/eval/metrics.py on small hand-checkable inputs."""

import sqlite3
import time

import numpy as np
import pytest

from src.eval import metrics as M


# Classification -------------------------------------------------------------


def test_accuracy_counts_unparseable_as_wrong():
    r = M.accuracy(["a", None, "b", "c"], ["a", "b", "b", "b"])
    assert r.per_item.tolist() == [1, 0, 1, 0]
    assert r.aggregate == 0.5


GOLD = ["a", "a", "b", "b", "c"]
PRED = ["a", "b", "b", "b", None]
# By hand: F1(a) = 2*1/(2+0+1) = 2/3, F1(b) = 2*2/(4+1+0) = 4/5, F1(c) = 0.
MACRO_BY_HAND = (2 / 3 + 4 / 5 + 0) / 3


def test_macro_f1_hand_example():
    r = M.macro_f1(PRED, GOLD)
    assert r.aggregate == pytest.approx(MACRO_BY_HAND)
    assert r.per_item.tolist() == [1, 0, 1, 1, 0]
    assert r.details["codes"].tolist() == [[0, 0], [0, 1], [1, 1], [1, 1], [2, -1]]


def test_macro_f1_matches_sklearn_with_fixed_labels():
    f1_score = pytest.importorskip("sklearn.metrics").f1_score
    pred = [p if p is not None else "__none__" for p in PRED]
    expected = f1_score(GOLD, pred, labels=["a", "b", "c"], average="macro", zero_division=0)
    assert M.macro_f1(PRED, GOLD).aggregate == pytest.approx(expected)


def test_macro_f1_out_of_set_prediction_adds_no_class():
    r = M.macro_f1(["a", "zzz"], ["a", "b"])
    assert r.aggregate == pytest.approx((1.0 + 0.0) / 2)
    with pytest.raises(ValueError):
        M.macro_f1(["a"], ["q"], labels=["a"])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="lengths"):
        M.accuracy(["a"], ["a", "b"])


# NER ------------------------------------------------------------------------


def test_typed_ner_f1_counts_and_none():
    gold = [[("PER", "Ann"), ("LOC", "Baku")], [], [("ORG", "UN")]]
    pred = [[("PER", "ann"), ("ORG", "Baku")], [], None]
    r = M.typed_ner_f1(pred, gold)
    assert r.details["counts"].tolist() == [[1, 1, 1], [0, 0, 0], [0, 0, 1]]
    # tp=1, fp=1, fn=2 -> P=1/2, R=1/3, F1=0.4
    assert r.aggregate == pytest.approx(0.4)
    assert r.per_item.tolist() == pytest.approx([0.5, 1.0, 0.0])


def test_seqeval_span_f1_matches_seqeval_and_repairs_length():
    seqeval_metrics = pytest.importorskip("seqeval.metrics")
    gold = [["B-PER", "I-PER", "O", "B-LOC"], ["O", "B-ORG"]]
    pred = [["B-PER", "I-PER", "O", "O"], ["O", "B-ORG", "O"]]  # second one too long
    r = M.seqeval_span_f1(pred, gold)
    assert r.details["length_repairs"] == 1
    assert r.aggregate == pytest.approx(seqeval_metrics.f1_score(gold, [pred[0], pred[1][:2]]))
    # tp=2, fp=0, fn=1 -> F1 = 4/5
    assert r.aggregate == pytest.approx(0.8)
    assert r.details["counts"].tolist() == [[1, 0, 1], [1, 0, 0]]


# Summarisation --------------------------------------------------------------


def test_rouge_l_hand_example():
    pytest.importorskip("rouge_score")
    r = M.rouge_l(["the cat sat", "dog", None], ["the cat sat on the mat", "dog", "something"])
    # LCS 3, P = 3/3, R = 3/6 -> F = 2/3; identical -> 1; empty prediction -> 0
    assert r.per_item.tolist() == pytest.approx([2 / 3, 1.0, 0.0])


def test_bertscore_is_lazy():
    # The module imported above without bert_score; calling without it gives a clear error.
    try:
        import bert_score  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="bert-score"):
            M.bertscore(["a"], ["a"])
    else:
        pytest.skip("bert_score installed; not running the model in tests")


# Text-to-SQL ----------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "toy.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (id INTEGER, name TEXT, age INTEGER)")
    conn.executemany("INSERT INTO singer VALUES (?, ?, ?)", [(1, "Ann", 30), (2, "Bo", 25), (3, "Cy", 30)])
    conn.commit()
    conn.close()
    return str(path)


def test_execution_accuracy(db):
    gold = [
        "SELECT name FROM singer WHERE age = 30",
        "SELECT name FROM singer ORDER BY age",
        "SELECT count(*) FROM singer",
        "SELECT name FROM singer",
        "SELECT name FROM singer",
        "SELECT name FROM singer",
    ]
    pred = [
        "SELECT name FROM singer WHERE age = 30 ORDER BY name DESC",  # same rows, other order: ok
        "SELECT name FROM singer ORDER BY age DESC",                  # gold is ordered: wrong
        "SELECT count(id) FROM singer",                               # same result: ok
        "SELEC name FROM singer",                                     # syntax error
        None,                                                         # unparseable
        "DELETE FROM singer",                                         # read-only: fails
    ]
    r = M.execution_accuracy(pred, gold, [db] * len(gold))
    assert r.per_item.tolist() == [1, 0, 1, 0, 0, 0]
    assert r.details["errors"]["unparsed"] == 1
    assert sum(r.details["errors"].values()) == 3
    assert M.run_sql(db, "SELECT count(*) FROM singer") == [(3,)]  # the DELETE did not run


def test_execution_accuracy_multiset_not_set(db):
    # Duplicates matter: {30, 30, 25} is not {30, 25}.
    r = M.execution_accuracy(["SELECT DISTINCT age FROM singer"], ["SELECT age FROM singer"], [db])
    assert r.per_item.tolist() == [0]


def test_execution_timeout(db):
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    start = time.monotonic()
    r = M.execution_accuracy([endless], ["SELECT 1"], [db], timeout_s=0.2)
    assert time.monotonic() - start < 5
    assert r.per_item.tolist() == [0] and r.details["errors"] == {"timeout": 1}


def test_failing_gold_raises(db):
    with pytest.raises(RuntimeError, match="gold query"):
        M.execution_accuracy(["SELECT 1"], ["SELECT nope FROM singer"], [db])


# Format compliance and calibration -----------------------------------------


def test_format_compliance():
    assert M.format_compliance([True, False, True, True]).aggregate == 0.75


def calibrated_toy():
    # 10 items at confidence 0.8 with 8 correct, 10 at 0.3 with 3 correct.
    conf = [0.8] * 10 + [0.3] * 10
    correct = [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7
    return conf, correct


def test_ece_zero_when_perfectly_calibrated():
    conf, correct = calibrated_toy()
    assert M.expected_calibration_error(conf, correct).aggregate == pytest.approx(0.0)
    assert M.expected_calibration_error(conf, correct, n_bins=2, binning="equal_mass").aggregate == pytest.approx(0.0)


def test_ece_hand_example():
    # All at 0.9 with half correct: one bin, gap 0.4.
    r = M.expected_calibration_error([0.9] * 10, [1, 0] * 5)
    assert r.aggregate == pytest.approx(0.4)
    assert r.per_item.sum() == pytest.approx(r.aggregate)
    assert r.details["bins"] == [{"bin": 13, "count": 10, "accuracy": 0.5, "confidence": 0.9}]
    # Two groups: 0.9 with 10/10 correct (gap 0.1) and 0.6 with 0/10 correct (gap 0.6): ECE = 0.35.
    r = M.expected_calibration_error([0.9] * 10 + [0.6] * 10, [1] * 10 + [0] * 10)
    assert r.aggregate == pytest.approx(0.35)


def test_equal_width_bin_edges():
    bins = M._bin_assignments(np.array([0.0, 1 / 15, 1 / 15 + 1e-9, 0.5, 1.0]), 15, "equal_width")
    assert bins.tolist() == [0, 0, 1, 7, 14]


def test_equal_mass_bins_have_equal_counts():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=150)
    r = M.expected_calibration_error(conf, rng.integers(0, 2, 150), binning="equal_mass")
    assert [b["count"] for b in r.details["bins"]] == [10] * 15


def test_brier_hand_example():
    # (0.8 - 1)^2 = 0.04, (0.3 - 0)^2 = 0.09 -> mean 0.065
    assert M.brier_score([0.8, 0.3], [1, 0]).aggregate == pytest.approx(0.065)
    with pytest.raises(ValueError):
        M.brier_score([1.2], [1])
