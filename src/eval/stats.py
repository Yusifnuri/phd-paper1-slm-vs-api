"""
Paired statistics for the pre-registered hypotheses.

- H1 / H6: exact McNemar on paired correctness; Holm within task.
- H1 (non-binary metrics), H5: hierarchical paired bootstrap over items and
  seeds; TOST equivalence from a bootstrap CI.
- H3: two-group permutation test of a difference in mean fine-tuning gains.
- H2: bootstrap CI for a ratio of means (cost per correct answer).

All resampling functions take an explicit integer `seed` and are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.stats import binom

from .metrics import macro_f1_from_codes

DEFAULT_RESAMPLES = 10_000


# McNemar --------------------------------------------------------------------


@dataclass
class McNemarResult:
    """Exact McNemar test with the full 2x2 table."""

    both_correct: int
    a_only: int  # A correct, B wrong (discordant)
    b_only: int  # A wrong, B correct (discordant)
    both_wrong: int
    p_value: float

    @property
    def n_discordant(self) -> int:
        """Number of discordant pairs."""
        return self.a_only + self.b_only


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """Exact two-sided McNemar test: binomial(n_discordant, 0.5) on the smaller discordant count."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"need two 1-D arrays of equal length, got {a.shape} and {b.shape}")
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    p = 1.0 if n == 0 else float(min(1.0, 2.0 * binom.cdf(min(n10, n01), n, 0.5)))
    return McNemarResult(
        both_correct=int(np.sum(a & b)), a_only=n10, b_only=n01,
        both_wrong=int(np.sum(~a & ~b)), p_value=p,
    )


# Holm -----------------------------------------------------------------------


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted_sorted, 1.0)
    return out


def holm_reject(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of hypotheses rejected by Holm at family-wise level alpha."""
    return holm_adjust(p_values) <= alpha


# Hierarchical paired bootstrap ---------------------------------------------


@dataclass
class BootstrapResult:
    """Observed difference A - B, its percentile CI, and the bootstrap distribution."""

    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    p_value: float
    n_resamples: int
    distribution: np.ndarray = field(repr=False)

    def ci(self, level: float) -> tuple:
        """Percentile CI at another level from the stored distribution."""
        return percentile_ci(self.distribution, level)


def percentile_ci(distribution: np.ndarray, level: float = 0.95) -> tuple:
    """Percentile interval of a bootstrap distribution, ignoring NaNs."""
    alpha = 1.0 - level
    lo, hi = np.nanpercentile(distribution, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _as_seed_item(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim < 2:
        raise ValueError("scores must have shape (n_items,), (n_seeds, n_items) or (n_seeds, n_items, k)")
    return x


def _mean_statistic(x: np.ndarray) -> float:
    return float(x.mean())


def micro_f1_statistic(x: np.ndarray) -> float:
    """Bootstrap statistic for (n_seeds, n_items, 3) tp/fp/fn counts: mean over seeds of micro-F1."""
    tp, fp, fn = x[..., 0].sum(axis=1), x[..., 1].sum(axis=1), x[..., 2].sum(axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1.mean())


def macro_f1_statistic(n_classes: int) -> Callable[[np.ndarray], float]:
    """Bootstrap statistic for (n_seeds, n_items, 2) gold/pred codes: mean over seeds of macro-F1."""

    def stat(x: np.ndarray) -> float:
        """Mean over seeds of macro-F1 on resampled codes."""
        return float(np.mean([macro_f1_from_codes(s[:, 0], s[:, 1], n_classes) for s in x]))

    return stat


def hierarchical_paired_bootstrap(
    scores_a,
    scores_b,
    statistic: Optional[Callable[[np.ndarray], float]] = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap of statistic(A) - statistic(B), resampling items then seeds.

    scores_x has shape (n_items,), (n_seeds, n_items) or (n_seeds, n_items, k);
    item columns must be aligned between A and B. Each resample draws one set
    of item indices shared by A and B (the pairing), then draws seeds with
    replacement separately for each system (seeds are not paired across
    systems; a system with one run keeps it). The default statistic is the
    mean over seeds and items. p_value is the two-sided bootstrap
    approximation 2 * min(P(d <= 0), P(d >= 0)), capped at 1.
    """
    a, b = _as_seed_item(scores_a), _as_seed_item(scores_b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"A has {a.shape[1]} items and B has {b.shape[1]}")
    if a.shape[2:] != b.shape[2:]:
        raise ValueError("A and B have different trailing dimensions")
    n_items = a.shape[1]
    rng = np.random.default_rng(seed)
    stat = statistic or _mean_statistic
    estimate = stat(a) - stat(b)

    if statistic is None and a.ndim == 2:
        # Vectorised: an item resample is a multinomial count vector.
        diffs = np.empty(n_resamples)
        chunk = max(1, min(n_resamples, 2_000_000 // max(n_items, 1)))
        done = 0
        while done < n_resamples:
            m = min(chunk, n_resamples - done)
            w_items = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=m) / n_items
            per_seed_a = w_items @ a.T  # (m, S_a)
            per_seed_b = w_items @ b.T
            w_sa = rng.multinomial(a.shape[0], np.full(a.shape[0], 1.0 / a.shape[0]), size=m) / a.shape[0]
            w_sb = rng.multinomial(b.shape[0], np.full(b.shape[0], 1.0 / b.shape[0]), size=m) / b.shape[0]
            diffs[done : done + m] = (per_seed_a * w_sa).sum(1) - (per_seed_b * w_sb).sum(1)
            done += m
    else:
        diffs = np.empty(n_resamples)
        for i in range(n_resamples):
            items = rng.integers(0, n_items, n_items)
            sa = rng.integers(0, a.shape[0], a.shape[0])
            sb = rng.integers(0, b.shape[0], b.shape[0])
            diffs[i] = stat(a[sa][:, items]) - stat(b[sb][:, items])

    lo, hi = percentile_ci(diffs, ci_level)
    p = float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))
    return BootstrapResult(estimate, lo, hi, ci_level, p, n_resamples, diffs)


# TOST -----------------------------------------------------------------------


@dataclass
class TostResult:
    """Equivalence decision from a confidence interval and a margin."""

    equivalent: bool
    margin: float
    ci_low: float
    ci_high: float


def tost_from_ci(ci_low: float, ci_high: float, margin: float) -> TostResult:
    """TOST: equivalent iff the (1 - 2*alpha) CI lies strictly inside (-margin, +margin).

    For alpha = 0.05 pass the 90% CI, e.g. BootstrapResult.ci(0.90).
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    return TostResult(bool(-margin < ci_low and ci_high < margin), margin, ci_low, ci_high)


def tost_bootstrap(result: BootstrapResult, margin: float, alpha: float = 0.05) -> TostResult:
    """TOST at level alpha from a bootstrap result, using its (1 - 2*alpha) percentile CI."""
    lo, hi = result.ci(1 - 2 * alpha)
    return tost_from_ci(lo, hi, margin)


# Permutation test (H3) ------------------------------------------------------


@dataclass
class PermutationResult:
    """Observed difference in group means and its permutation p-value."""

    estimate: float
    p_value: float
    n_permutations: int
    n_group1: int
    n_group2: int


def permutation_test_gain_difference(
    gains: Sequence[float],
    group: Sequence[bool],
    n_permutations: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> PermutationResult:
    """Two-sided permutation test of mean(gain | group) - mean(gain | not group).

    `gains` are per-item fine-tuning gains (e.g. fine-tuned minus base score)
    and `group` marks the latent-knowledge stratum. p = (1 + #{|d*| >= |d|}) /
    (1 + n_permutations), so p is never 0.
    """
    g = np.asarray(gains, dtype=float)
    mask = np.asarray(group, dtype=bool)
    if g.shape != mask.shape or g.ndim != 1:
        raise ValueError("gains and group must be 1-D and the same length")
    n1 = int(mask.sum())
    if n1 == 0 or n1 == len(g):
        raise ValueError("both groups must be non-empty")
    observed = g[mask].mean() - g[~mask].mean()
    rng = np.random.default_rng(seed)
    total, n = g.sum(), len(g)
    count = 0
    for _ in range(n_permutations):
        idx = rng.permutation(n)[:n1]
        s1 = g[idx].sum()
        d = s1 / n1 - (total - s1) / (n - n1)
        count += abs(d) >= abs(observed) - 1e-12
    return PermutationResult(float(observed), (count + 1) / (n_permutations + 1), n_permutations, n1, n - n1)


# Ratio of means (H2) --------------------------------------------------------


@dataclass
class RatioResult:
    """Ratio of means with a percentile bootstrap CI."""

    estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int
    n_undefined: int  # resamples whose denominator summed to zero


def bootstrap_ratio_of_means(
    numerator: Sequence[float],
    denominator: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci_level: float = 0.95,
    seed: int = 0,
) -> RatioResult:
    """Bootstrap CI for sum(numerator) / sum(denominator), resampling items as pairs.

    For cost per correct answer pass per-item cost and per-item correctness.
    Resamples with a zero denominator are undefined, counted, and left out of
    the percentiles.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.shape != den.shape or num.ndim != 1:
        raise ValueError("numerator and denominator must be 1-D and the same length")
    if den.sum() == 0:
        raise ValueError("denominator sums to zero: the ratio is undefined")
    n = len(num)
    rng = np.random.default_rng(seed)
    ratios = np.empty(n_resamples)
    chunk = max(1, min(n_resamples, 2_000_000 // max(n, 1)))
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=m)
        d = w @ den
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[done : done + m] = np.where(d != 0, (w @ num) / np.where(d != 0, d, 1), np.nan)
        done += m
    undefined = int(np.isnan(ratios).sum())
    lo, hi = percentile_ci(ratios, ci_level)
    return RatioResult(float(num.sum() / den.sum()), lo, hi, ci_level, n_resamples, undefined)


def summarize(result) -> Dict[str, object]:
    """Flatten a result dataclass to a dict without the bootstrap distribution."""
    return {k: v for k, v in vars(result).items() if k != "distribution"}
