"""Tests for src/eval/stats.py against values derived by hand or by simulation."""

from math import comb

import numpy as np
import pytest

from src.eval import metrics as M
from src.eval import stats as S


# McNemar --------------------------------------------------------------------


def test_mcnemar_hand_value():
    # Table: 5 both correct, 8 only A, 2 only B, 3 both wrong.
    a = [1] * 5 + [1] * 8 + [0] * 2 + [0] * 3
    b = [1] * 5 + [0] * 8 + [1] * 2 + [0] * 3
    r = S.mcnemar_exact(a, b)
    assert (r.both_correct, r.a_only, r.b_only, r.both_wrong) == (5, 8, 2, 3)
    assert r.n_discordant == 10
    # p = 2 * P(X <= 2), X ~ Binomial(10, 1/2) = 2 * (C(10,0) + C(10,1) + C(10,2)) / 2^10 = 112 / 1024
    assert r.p_value == pytest.approx(2 * (comb(10, 0) + comb(10, 1) + comb(10, 2)) / 2**10)
    assert r.p_value == pytest.approx(0.109375)
    assert S.mcnemar_exact(b, a).p_value == pytest.approx(r.p_value)  # symmetric


def test_mcnemar_edge_cases():
    assert S.mcnemar_exact([1, 0], [1, 0]).p_value == 1.0  # no discordant pairs
    assert S.mcnemar_exact([1] * 3, [0] * 3).p_value == pytest.approx(2 / 8)  # 2 * (1/2)^3
    assert S.mcnemar_exact([1, 0], [0, 1]).p_value == 1.0  # capped at 1


# Holm -----------------------------------------------------------------------


def test_holm_textbook_example():
    # Sorted p: 0.005, 0.01, 0.03, 0.04; times 4, 3, 2, 1 -> 0.02, 0.03, 0.06, 0.04;
    # running maximum -> 0.02, 0.03, 0.06, 0.06; mapped back to input order.
    p = [0.01, 0.04, 0.03, 0.005]
    assert S.holm_adjust(p).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert S.holm_reject(p, 0.05).tolist() == [True, False, False, True]


def test_holm_caps_at_one_and_single_test_unchanged():
    assert S.holm_adjust([0.6, 0.7]).tolist() == [1.0, 1.0]
    assert S.holm_adjust([0.03]).tolist() == [0.03]


# Hierarchical paired bootstrap ---------------------------------------------


def test_bootstrap_constant_shift_is_exact():
    b = np.random.default_rng(1).normal(size=300)
    r = S.hierarchical_paired_bootstrap(b + 0.3, b, n_resamples=500)
    assert r.estimate == pytest.approx(0.3)
    assert (r.ci_low, r.ci_high) == pytest.approx((0.3, 0.3))


def test_bootstrap_coverage_of_known_difference():
    # True mean difference 0.5. Over 200 simulated data sets the 95% CI should
    # cover it about 95% of the time; accept 0.90 to 0.99.
    rng = np.random.default_rng(42)
    covered = 0
    for rep in range(200):
        base = rng.normal(size=200)
        a = base + 0.5 + rng.normal(scale=0.5, size=200)
        b = base + rng.normal(scale=0.5, size=200)
        r = S.hierarchical_paired_bootstrap(a, b, n_resamples=1000, seed=rep)
        covered += r.ci_low <= 0.5 <= r.ci_high
    assert 0.90 <= covered / 200 <= 0.99


def test_bootstrap_is_seeded_and_default_size():
    a, b = np.random.default_rng(0).uniform(size=(2, 100))
    r1 = S.hierarchical_paired_bootstrap(a, b, seed=7)
    r2 = S.hierarchical_paired_bootstrap(a, b, seed=7)
    assert r1.n_resamples == 10_000 and len(r1.distribution) == 10_000
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)


def test_bootstrap_vectorised_and_loop_paths_agree():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, size=(3, 400)).astype(float)  # 3 seeds
    b = rng.integers(0, 2, size=(1, 400)).astype(float)  # e.g. an API run, one "seed"
    fast = S.hierarchical_paired_bootstrap(a, b, n_resamples=4000, seed=0)
    slow = S.hierarchical_paired_bootstrap(a, b, statistic=lambda x: float(x.mean()), n_resamples=4000, seed=0)
    assert fast.estimate == pytest.approx(slow.estimate)
    assert fast.ci_low == pytest.approx(slow.ci_low, abs=0.01)
    assert fast.ci_high == pytest.approx(slow.ci_high, abs=0.01)


def test_bootstrap_seed_resampling_widens_ci():
    # Same item means, but A's seeds disagree: the CI must be wider than with one seed.
    rng = np.random.default_rng(5)
    base = rng.uniform(size=500)
    a_one = base[None, :]
    a_seeds = np.stack([base - 0.1, base, base + 0.1])
    b = rng.uniform(size=500)
    one = S.hierarchical_paired_bootstrap(a_one, b, n_resamples=3000)
    three = S.hierarchical_paired_bootstrap(a_seeds, b, n_resamples=3000)
    assert three.estimate == pytest.approx(one.estimate)
    assert (three.ci_high - three.ci_low) > (one.ci_high - one.ci_low)


def test_bootstrap_p_value():
    rng = np.random.default_rng(0)
    b = rng.normal(size=500)
    assert S.hierarchical_paired_bootstrap(b + 1.0 + rng.normal(size=500), b, n_resamples=2000).p_value < 0.001
    same = S.hierarchical_paired_bootstrap(b + rng.normal(scale=1e-3, size=500), b + rng.normal(scale=1e-3, size=500), n_resamples=2000)
    assert same.p_value > 0.05


def test_bootstrap_micro_f1_statistic_matches_metric():
    gold = [[("PER", "Ann")], [("LOC", "Baku"), ("ORG", "UN")], []]
    pred_a = [[("PER", "Ann")], [("LOC", "Baku")], []]
    pred_b = [[], [("LOC", "Baku")], [("PER", "x")]]
    ca = M.typed_ner_f1(pred_a, gold).details["counts"]
    cb = M.typed_ner_f1(pred_b, gold).details["counts"]
    r = S.hierarchical_paired_bootstrap(ca[None], cb[None], statistic=S.micro_f1_statistic, n_resamples=200)
    expected = M.typed_ner_f1(pred_a, gold).aggregate - M.typed_ner_f1(pred_b, gold).aggregate
    assert r.estimate == pytest.approx(expected)


def test_bootstrap_macro_f1_statistic_matches_metric():
    gold = ["a", "a", "b", "b", "c"]
    pa = M.macro_f1(["a", "b", "b", "b", None], gold, labels=["a", "b", "c"])
    pb = M.macro_f1(["a", "a", "b", "b", "c"], gold, labels=["a", "b", "c"])
    r = S.hierarchical_paired_bootstrap(
        pa.details["codes"][None], pb.details["codes"][None], statistic=S.macro_f1_statistic(3), n_resamples=200
    )
    assert r.estimate == pytest.approx(pa.aggregate - pb.aggregate)


def test_bootstrap_rejects_misaligned_items():
    with pytest.raises(ValueError, match="items"):
        S.hierarchical_paired_bootstrap(np.zeros(5), np.zeros(6))


# TOST -----------------------------------------------------------------------


def test_tost_from_ci():
    assert S.tost_from_ci(-0.5, 0.8, margin=1.0).equivalent
    assert not S.tost_from_ci(-0.5, 1.2, margin=1.0).equivalent
    assert not S.tost_from_ci(-1.0, 0.0, margin=1.0).equivalent  # bound on the margin is not inside
    with pytest.raises(ValueError):
        S.tost_from_ci(0, 0, margin=0)


def test_tost_bootstrap_uses_90_percent_ci():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 100, size=1000)  # scores in points
    a = base + rng.normal(scale=0.5, size=1000)
    b = base + rng.normal(scale=0.5, size=1000)
    r = S.hierarchical_paired_bootstrap(a, b, n_resamples=2000)
    t = S.tost_bootstrap(r, margin=1.0)
    assert (t.ci_low, t.ci_high) == pytest.approx(r.ci(0.90))
    assert t.equivalent
    shifted = S.hierarchical_paired_bootstrap(a + 2.0, b, n_resamples=2000)
    assert not S.tost_bootstrap(shifted, margin=1.0).equivalent


# Permutation test -----------------------------------------------------------


def test_permutation_exact_small_case():
    # gains 1,2,3,4 with groups {1,2} vs {3,4}: observed d = 1.5 - 3.5 = -2.
    # The 6 equally likely splits give d = -2, -1, 0, 0, 1, 2, so P(|d| >= 2) = 2/6.
    r = S.permutation_test_gain_difference([1, 2, 3, 4], [True, True, False, False], n_permutations=20_000)
    assert r.estimate == -2.0
    assert r.p_value == pytest.approx(1 / 3, abs=0.015)
    assert (r.n_group1, r.n_group2) == (2, 2)


def test_permutation_separated_and_null():
    sep = S.permutation_test_gain_difference([1.0] * 10 + [0.0] * 10, [True] * 10 + [False] * 10, n_permutations=5000)
    assert sep.estimate == 1.0 and sep.p_value < 0.01
    null = S.permutation_test_gain_difference([0.2] * 20, [True] * 10 + [False] * 10, n_permutations=500)
    assert null.p_value == 1.0
    with pytest.raises(ValueError):
        S.permutation_test_gain_difference([1, 2], [True, True])


# Ratio of means -------------------------------------------------------------


def test_ratio_of_means_cost_per_correct():
    # 100 items at $0.02 each, 50 correct: cost per correct = 2.00 / 50 = $0.04.
    cost = np.full(100, 0.02)
    correct = np.array([1, 0] * 50)
    r = S.bootstrap_ratio_of_means(cost, correct, n_resamples=4000)
    assert r.estimate == pytest.approx(0.04)
    assert r.ci_low < 0.04 < r.ci_high
    # With constant cost the ratio is 0.02 / mean(correct); mean(correct) stays well
    # inside (0.3, 0.7) for n = 100, so the CI lies inside (0.02/0.7, 0.02/0.3).
    assert 0.02 / 0.7 < r.ci_low and r.ci_high < 0.02 / 0.3
    assert r.n_undefined == 0


def test_ratio_of_means_undefined_denominator():
    with pytest.raises(ValueError):
        S.bootstrap_ratio_of_means([1.0, 1.0], [0, 0])
    r = S.bootstrap_ratio_of_means([1.0, 1.0], [1, 0], n_resamples=1000)
    assert r.n_undefined > 0  # resamples drawing only the zero item


def test_summarize_drops_distribution():
    r = S.hierarchical_paired_bootstrap(np.ones(5), np.zeros(5), n_resamples=10)
    assert "distribution" not in S.summarize(r) and S.summarize(r)["estimate"] == 1.0
