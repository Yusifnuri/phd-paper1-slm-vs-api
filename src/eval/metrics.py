"""
Task metrics. Each metric returns a MetricResult with per-item scores (aligned
with the input order) and the corpus-level aggregate.

Conventions:

- An unparseable prediction is passed as None and scored as wrong (0) or as
  an empty output, never dropped, so every system is scored on every item.
- For corpus-level metrics that are not a mean of per-item scores (macro-F1,
  micro-F1, ECE), `per_item` is still given, and `details` holds what the
  bootstrap needs to recompute the aggregate on resampled items
  (see stats.micro_f1_statistic and stats.macro_f1_statistic).
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .ner_typed import (  # noqa: F401  (re-exported)
    Entity,
    bio_tags,
    bio_to_entities,
    lenient_sentence_f1,
    micro_prf,
    parse_api_entities,
    prf_from_counts,
    sentence_counts,
)

BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"


@dataclass
class MetricResult:
    """Per-item scores plus the aggregate value of one metric."""

    name: str
    per_item: np.ndarray
    aggregate: float
    details: Dict[str, Any] = field(default_factory=dict)


def _check_lengths(*seqs: Sequence) -> int:
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"inputs have different lengths: {[len(s) for s in seqs]}")
    return lengths.pop()


# Classification and sentiment ------------------------------------------------


def accuracy(pred: Sequence[Any], gold: Sequence[Any]) -> MetricResult:
    """Exact-match accuracy; None predictions count as wrong."""
    _check_lengths(pred, gold)
    per_item = np.array([p is not None and p == g for p, g in zip(pred, gold)], dtype=float)
    return MetricResult("accuracy", per_item, float(per_item.mean()) if len(per_item) else float("nan"))


def macro_f1_from_codes(gold_codes: np.ndarray, pred_codes: np.ndarray, n_classes: int) -> float:
    """Macro-F1 over classes 0..n_classes-1 from integer codes; pred code -1 means no valid class."""
    gold_codes = np.asarray(gold_codes)
    pred_codes = np.asarray(pred_codes)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred_codes == c) & (gold_codes == c))
        fp = np.sum((pred_codes == c) & (gold_codes != c))
        fn = np.sum((pred_codes != c) & (gold_codes == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f1s))


def macro_f1(
    pred: Sequence[Any], gold: Sequence[Any], labels: Optional[Sequence[Any]] = None
) -> MetricResult:
    """Unweighted mean of per-class F1 over the label set (default: labels seen in gold).

    A prediction outside the label set (or None) counts as a miss for the gold
    class and adds no new class. This differs from sklearn's default, which
    averages over the union of gold and predicted labels; pass `labels` to
    sklearn to reproduce this value. per_item holds exact-match correctness.
    """
    _check_lengths(pred, gold)
    if labels is None:
        labels = sorted(set(gold), key=str)
    index = {lab: i for i, lab in enumerate(labels)}
    unknown_gold = sorted({str(g) for g in gold if g not in index})
    if unknown_gold:
        raise ValueError(f"gold labels not in label set: {unknown_gold[:5]}")
    gold_codes = np.array([index[g] for g in gold], dtype=int)
    pred_codes = np.array([index.get(p, -1) if p is not None else -1 for p in pred], dtype=int)
    value = macro_f1_from_codes(gold_codes, pred_codes, len(labels))
    per_item = (gold_codes == pred_codes).astype(float)
    return MetricResult(
        "macro_f1",
        per_item,
        value,
        {"labels": list(labels), "codes": np.stack([gold_codes, pred_codes], axis=1)},
    )


# NER ------------------------------------------------------------------------


def typed_ner_f1(
    pred: Sequence[Optional[Sequence[Entity]]],
    gold: Sequence[Sequence[Entity]],
    typed: bool = True,
    case_sensitive: bool = False,
) -> MetricResult:
    """Typed micro-F1 over (type, surface) multisets; None predictions count as no entities.

    per_item is sentence-level F1 (1.0 when gold and prediction are both
    empty); details['counts'] is an (N, 3) array of tp, fp, fn per sentence.
    """
    _check_lengths(pred, gold)
    pred = [p if p is not None else [] for p in pred]
    res = micro_prf(gold, pred, typed=typed, case_sensitive=case_sensitive)
    counts = np.array(res["per_sentence"], dtype=int).reshape(-1, 3)
    per_item = np.array(
        [1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn) for tp, fp, fn in counts]
    )
    return MetricResult(
        "typed_ner_micro_f1",
        per_item,
        float(res["f1"]),
        {"precision": res["precision"], "recall": res["recall"], "counts": counts},
    )


def seqeval_span_f1(
    pred_tags: Sequence[Optional[Sequence[str]]], gold_tags: Sequence[Sequence[str]]
) -> MetricResult:
    """seqeval span micro-F1 for token-aligned BIO outputs (default, non-strict mode).

    A predicted tag sequence of the wrong length is truncated or padded with
    'O' to the gold length, and None becomes all 'O'; the number of such
    repairs is reported in details['length_repairs'].
    """
    from seqeval.metrics import f1_score
    from seqeval.metrics.sequence_labeling import get_entities

    _check_lengths(pred_tags, gold_tags)
    fixed, repairs = [], 0
    for p, g in zip(pred_tags, gold_tags):
        p = list(p) if p is not None else []
        if len(p) != len(g):
            repairs += 1
            p = (p + ["O"] * len(g))[: len(g)]
        fixed.append(p)
    counts = []
    for p, g in zip(fixed, gold_tags):
        gc, pc = Counter(get_entities(list(g))), Counter(get_entities(p))
        hit = sum((gc & pc).values())
        counts.append((hit, sum(pc.values()) - hit, sum(gc.values()) - hit))
    counts = np.array(counts, dtype=int).reshape(-1, 3)
    per_item = np.array(
        [1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn) for tp, fp, fn in counts]
    )
    value = float(f1_score([list(g) for g in gold_tags], fixed, zero_division=0))
    return MetricResult(
        "seqeval_span_f1", per_item, value, {"counts": counts, "length_repairs": repairs}
    )


# Summarisation --------------------------------------------------------------


def rouge_l(
    pred: Sequence[Optional[str]], gold: Sequence[str], variant: str = "rougeL", use_stemmer: bool = True
) -> MetricResult:
    """ROUGE-L F-measure per item via Google's rouge-score; variant 'rougeLsum' splits on newlines."""
    from rouge_score import rouge_scorer

    if variant not in ("rougeL", "rougeLsum"):
        raise ValueError(f"variant must be 'rougeL' or 'rougeLsum', got {variant!r}")
    _check_lengths(pred, gold)
    scorer = rouge_scorer.RougeScorer([variant], use_stemmer=use_stemmer)
    per_item = np.array([scorer.score(g, p or "")[variant].fmeasure for p, g in zip(pred, gold)])
    return MetricResult(variant, per_item, float(per_item.mean()) if len(per_item) else float("nan"))


def bertscore(
    pred: Sequence[Optional[str]],
    gold: Sequence[str],
    model_type: str = BERTSCORE_MODEL,
    rescale_with_baseline: bool = True,
    lang: str = "en",
    batch_size: int = 16,
    device: Optional[str] = None,
) -> MetricResult:
    """BERTScore F1 per item (deberta-xlarge-mnli, baseline-rescaled); imports bert_score lazily."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "bertscore needs the optional 'bert-score' package (and torch); "
            "see requirements/eval.txt"
        ) from e
    _check_lengths(pred, gold)
    _, _, f1 = bert_score_fn(
        [p or "" for p in pred],
        list(gold),
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
        device=device,
    )
    per_item = f1.detach().cpu().numpy().astype(float)
    return MetricResult(
        "bertscore_f1",
        per_item,
        float(per_item.mean()),
        {"model_type": model_type, "rescale_with_baseline": rescale_with_baseline},
    )


# Text-to-SQL ----------------------------------------------------------------

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


class SQLTimeout(Exception):
    """A query exceeded its execution time limit."""


def run_sql(db_path: str, sql: str, timeout_s: float = 30.0) -> List[tuple]:
    """Execute one query on a read-only SQLite database and return all rows, or raise."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    deadline = time.monotonic() + timeout_s
    timed_out = []

    def _progress() -> int:
        if time.monotonic() > deadline:
            timed_out.append(True)
            return 1
        return 0

    try:
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(_progress, 1000)
        try:
            return conn.execute(sql).fetchall()
        except sqlite3.OperationalError as e:
            if timed_out:
                raise SQLTimeout(f"query exceeded {timeout_s}s") from e
            raise
    finally:
        conn.close()


def results_match(pred_rows: Sequence[tuple], gold_rows: Sequence[tuple], ordered: bool) -> bool:
    """Compare result sets: exact row order if ordered, else as multisets of rows."""
    if ordered:
        return list(pred_rows) == list(gold_rows)
    return Counter(map(tuple, pred_rows)) == Counter(map(tuple, gold_rows))


def execution_accuracy(
    pred_sql: Sequence[Optional[str]],
    gold_sql: Sequence[str],
    db_paths: Sequence[str],
    timeout_s: float = 30.0,
) -> MetricResult:
    """Execution accuracy: per item, do predicted and gold queries return the same result?

    Rows are compared as multisets unless the gold query contains ORDER BY, in
    which case row order must match too; column order always matters. A
    prediction that is None, fails, or times out scores 0. A gold query that
    fails raises, because that is a data error. details['errors'] counts
    prediction failures by kind.
    """
    _check_lengths(pred_sql, gold_sql, db_paths)
    per_item, errors = [], Counter()
    for i, (p, g, db) in enumerate(zip(pred_sql, gold_sql, db_paths)):
        try:
            gold_rows = run_sql(db, g, timeout_s)
        except Exception as e:
            raise RuntimeError(f"gold query for item {i} failed on {db}: {e}") from e
        if p is None or not p.strip():
            errors["unparsed"] += 1
            per_item.append(0.0)
            continue
        try:
            pred_rows = run_sql(db, p, timeout_s)
        except SQLTimeout:
            errors["timeout"] += 1
            per_item.append(0.0)
            continue
        except (sqlite3.Error, sqlite3.Warning, ValueError) as e:
            errors[type(e).__name__] += 1
            per_item.append(0.0)
            continue
        per_item.append(float(results_match(pred_rows, gold_rows, bool(_ORDER_BY.search(g)))))
    per_item = np.array(per_item)
    return MetricResult(
        "execution_accuracy",
        per_item,
        float(per_item.mean()) if len(per_item) else float("nan"),
        {"errors": dict(errors)},
    )


# Format compliance ----------------------------------------------------------


def format_compliance(parse_ok: Sequence[bool]) -> MetricResult:
    """Share of outputs that parsed into the required format."""
    per_item = np.array([bool(x) for x in parse_ok], dtype=float)
    return MetricResult("format_compliance", per_item, float(per_item.mean()) if len(per_item) else float("nan"))


# Calibration ----------------------------------------------------------------


def _bin_assignments(conf: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    if binning == "equal_width":
        # Right-closed bins (lo, hi], with 0.0 in the first bin.
        return np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    if binning == "equal_mass":
        order = np.argsort(conf, kind="stable")
        bins = np.empty(len(conf), dtype=int)
        for b, idx in enumerate(np.array_split(order, n_bins)):
            bins[idx] = b
        return bins
    raise ValueError(f"binning must be 'equal_width' or 'equal_mass', got {binning!r}")


def _check_calibration_inputs(confidence: Sequence[float], correct: Sequence[Any]):
    _check_lengths(confidence, correct)
    conf = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    if np.isnan(conf).any() or (conf < 0).any() or (conf > 1).any():
        raise ValueError("confidence must be in [0, 1] with no missing values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("correct must be 0/1 or boolean")
    return conf, y


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[Any], n_bins: int = 15, binning: str = "equal_width"
) -> MetricResult:
    """ECE with equal-width or equal-mass bins (default 15 equal-width).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|. per_item holds each
    item's share of that sum (|acc_b - conf_b| / N), so per_item sums to ECE.
    """
    conf, y = _check_calibration_inputs(confidence, correct)
    n = len(conf)
    bins = _bin_assignments(conf, n_bins, binning)
    per_item = np.zeros(n)
    table = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        gap = abs(y[mask].mean() - conf[mask].mean())
        per_item[mask] = gap / n
        table.append(
            {"bin": b, "count": int(mask.sum()), "accuracy": float(y[mask].mean()),
             "confidence": float(conf[mask].mean())}
        )
    return MetricResult(
        f"ece_{binning}_{n_bins}", per_item, float(per_item.sum()), {"bins": table}
    )


def brier_score(confidence: Sequence[float], correct: Sequence[Any]) -> MetricResult:
    """Brier score of the confidence in the predicted answer: mean of (confidence - correct)^2."""
    conf, y = _check_calibration_inputs(confidence, correct)
    per_item = (conf - y) ** 2
    return MetricResult("brier", per_item, float(per_item.mean()) if len(per_item) else float("nan"))
