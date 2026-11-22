"""Confirmatory analysis code (numpy + scipy only).

Implements, as specified in docs/foundation/STATISTICAL_ANALYSIS_PLAN.md:
  1. crossed items x replicates paired contrast (ANOVA variance components, Satterthwaite df),
     optional finite-population correction (sections 1-2 of the plan; identical to
     power_sim2.py::a3, proven numerically in tests/analysis/test_inference.py);
  2. Holm adjustment over the 6-contrast family;
  3. two-stage percentile bootstrap (replicates, then items) with a fixed seed;
  4. minimal detectable gap from variance components;
  5. cost model: cost per correct, marginal and per-correct break-even Q*, utilisation u,
     Fieller interval or an explicit 'no break-even in range' result;
  6. the pre-registered decision rule (never outputs 'equivalent').

Sign convention: contrasts are d = arm_FT - arm_ICL; positive means FT better.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, norm, t as tdist

# Fixed seed for the bootstrap (arbitrary constant, chosen once; the plan requires only "fixed").
BOOTSTRAP_SEED = 20261122


# --------------------------------------------------------------------------------------
# 1. Crossed items x replicates contrast
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossedResult:
    mean: float
    var: float          # variance of the mean difference
    se: float
    df: float           # Satterthwaite df, unclamped (as a3 returns it)
    t: float
    p: float            # two-sided
    ci_low: float       # at `level`
    ci_high: float
    level: float
    n: int
    R: int
    ms_item: float
    ms_rep: float
    ms_res: float
    s2_item: float      # truncated at 0
    s2_rep: float       # truncated at 0
    s2_res: float
    f: Optional[float]

    def ci(self, level: float) -> tuple:
        """t interval at another level (e.g. 0.992 for the Bonferroni-level interval)."""
        crit = tdist.ppf(1 - (1 - level) / 2, max(self.df, 1.0))
        return (self.mean - crit * self.se, self.mean + crit * self.se)


def variance_components(d: np.ndarray) -> Dict[str, float]:
    """ANOVA (method of moments) components for d[i, r] = mu + a_i + b_r + e_ir, truncated at 0."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 2:
        raise ValueError("d must be a 2-D array [items, replicates]")
    n, R = d.shape
    if n < 2 or R < 2:
        raise ValueError("need at least 2 items and 2 replicates")
    dbar = d.mean(); di = d.mean(1); dr = d.mean(0)
    ms_i = R * ((di - dbar) ** 2).sum() / (n - 1)
    ms_r = n * ((dr - dbar) ** 2).sum() / (R - 1)
    res = d - di[:, None] - dr[None, :] + dbar
    ms_e = (res ** 2).sum() / ((n - 1) * (R - 1))
    return dict(n=n, R=R, mean=float(dbar), ms_item=float(ms_i), ms_rep=float(ms_r), ms_res=float(ms_e),
                s2_item=float(max((ms_i - ms_e) / R, 0.0)), s2_rep=float(max((ms_r - ms_e) / n, 0.0)),
                s2_res=float(ms_e))


def _rep_factor(R: int, f: Optional[float]) -> float:
    if f is None:
        return 1.0 / R
    if not (0.0 <= f < 1.0):
        raise ValueError("finite-population fraction f = N/M must be in [0, 1)")
    return 1.0 / R + f / (1.0 - f)


def _var_df(s_i: float, s_r: float, s_e: float, n: int, R: int, f: Optional[float]):
    """Variance of the mean and Satterthwaite df; same guards (1e-30) as power_sim2.a3."""
    cr = _rep_factor(R, f)
    v = s_i / n + s_r * cr + s_e / (n * R)
    den = (s_i / n) ** 2 / (n - 1) + (s_r * cr) ** 2 / (R - 1) + (s_e / (n * R)) ** 2 / ((n - 1) * (R - 1))
    return v, v ** 2 / (den + 1e-30)


def crossed_contrast(d: np.ndarray, f: Optional[float] = None, level: float = 0.95) -> CrossedResult:
    """Paired crossed contrast. d[i, r] = per-item, per-replicate difference between two arms
    (same items, same pool in replicate r). Var(mean) = s2_item/n + s2_rep*(1/R [+ f/(1-f)]) + s2_res/(nR).

    f=None: primary analysis (conditional estimand, plan section 2). f=N/M: super-population
    sensitivity analysis (sigma2_rep/R replaced by sigma2_rep*(1/R + f/(1-f)))."""
    c = variance_components(d)
    n, R = c["n"], c["R"]
    v, df = _var_df(c["s2_item"], c["s2_rep"], c["s2_res"], n, R, f)
    se = math.sqrt(v)
    tstat = c["mean"] / math.sqrt(v + 1e-18)   # a3's guard, kept so the equality is exact
    dfc = max(df, 1.0)
    p = float(2 * tdist.sf(abs(tstat), dfc))
    crit = tdist.ppf(1 - (1 - level) / 2, dfc)
    return CrossedResult(mean=c["mean"], var=v, se=se, df=df, t=tstat, p=p,
                         ci_low=c["mean"] - crit * se, ci_high=c["mean"] + crit * se, level=level,
                         n=n, R=R, ms_item=c["ms_item"], ms_rep=c["ms_rep"], ms_res=c["ms_res"],
                         s2_item=c["s2_item"], s2_rep=c["s2_rep"], s2_res=c["s2_res"], f=f)


# --------------------------------------------------------------------------------------
# 2. Holm over the confirmatory family
# --------------------------------------------------------------------------------------
FAMILY_SIZE = 6


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError(f"p-values must be finite and in [0, 1], got {p.tolist()}")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def holm_family(p_by_contrast: Dict[str, float], alpha: float = 0.05,
                expected_size: int = FAMILY_SIZE) -> Dict[str, Dict[str, float]]:
    """Holm over the named contrasts. The pre-registered family has 6 members
    ({BANKING77, BillSum} x N in {200, 1000, 3000}); a different size raises, because a silently
    smaller family would change the multiplicity correction."""
    if len(p_by_contrast) != expected_size:
        raise ValueError(f"family must have {expected_size} contrasts, got {len(p_by_contrast)}")
    names = list(p_by_contrast)
    adj = holm_adjust([p_by_contrast[k] for k in names])
    return {k: dict(p=float(p_by_contrast[k]), p_holm=float(a), reject=bool(a < alpha))
            for k, a in zip(names, adj)}


# --------------------------------------------------------------------------------------
# 3. Two-stage percentile bootstrap
# --------------------------------------------------------------------------------------
def two_stage_bootstrap(x: np.ndarray, n_boot: int = 10000, seed: int = BOOTSTRAP_SEED,
                        level: float = 0.95) -> Dict[str, object]:
    """Percentile bootstrap of the grand mean of x[i, r] (per-arm correctness/score, or a difference).
    Stage 1 resamples the R replicates with replacement; stage 2 resamples the n items with
    replacement (the same resampled items are used for every replicate, which keeps the item
    pairing). Descriptive only; with R < 8 the number of distinct replicate resamples is small
    (C(2R-1, R), returned) and the plan requires saying so."""
    x = np.asarray(x, dtype=float)
    n, R = x.shape
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, R, R), minlength=R)   # stage 1
        row_mean = x @ counts / R
        out[b] = row_mean[rng.integers(0, n, n)].mean()           # stage 2
    lo, hi = np.percentile(out, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return dict(mean=float(x.mean()), ci_low=float(lo), ci_high=float(hi), level=level, n_boot=n_boot,
                seed=seed, distinct_replicate_resamples=math.comb(2 * R - 1, R),
                small_R_warning=R < 8, draws=out)


# --------------------------------------------------------------------------------------
# 4. Minimal detectable gap
# --------------------------------------------------------------------------------------
def power_at(delta: float, s2_item: float, s2_rep: float, s2_res: float, n: int, R: int,
             alpha: float, f: Optional[float] = None, two_sided: bool = True) -> float:
    """Power of the crossed t test for true gap `delta`, treating the components as known: ncp =
    delta/SE, df = Satterthwaite df evaluated at the components (plug-in)."""
    v, df = _var_df(s2_item, s2_rep, s2_res, n, R, f)
    df = max(df, 1.0)
    ncp = abs(delta) / math.sqrt(v)
    a = alpha / 2 if two_sided else alpha
    crit = tdist.ppf(1 - a, df)
    pw = nct.sf(crit, df, ncp)
    if two_sided:
        pw += nct.cdf(-crit, df, ncp)
    # scipy's nct returns NaN for extreme ncp; power is 1 to machine precision there
    return 1.0 if not np.isfinite(pw) and ncp > crit else float(min(pw, 1.0))


def minimal_detectable_gap(s2_item: float, s2_rep: float, s2_res: float, n: int, R: int,
                           alpha: float = 0.0083, power: float = 0.80, f: Optional[float] = None,
                           two_sided: bool = True) -> float:
    """Smallest |delta| with power >= `power` (same units as the components' square root).
    alpha defaults to the first-step Holm-6 level 0.05/6 ~ 0.0083 used in the plan's power tables."""
    hi = 5 * math.sqrt(_var_df(s2_item, s2_rep, s2_res, n, R, f)[0]) or 1e-9
    while power_at(hi, s2_item, s2_rep, s2_res, n, R, alpha, f, two_sided) < power:
        hi *= 2
        if hi > 1e6:
            raise ValueError("no finite minimal detectable gap")
    return float(brentq(lambda d: power_at(d, s2_item, s2_rep, s2_res, n, R, alpha, f, two_sided) - power,
                        1e-12, hi, xtol=1e-10))


# --------------------------------------------------------------------------------------
# 5. Cost model
# --------------------------------------------------------------------------------------
def self_hosted_cost_per_query(gpu_seconds_per_query: float, rate_per_hour: float, u: float = 1.0) -> float:
    """Serving cost per query: measured GPU-seconds at saturation x rate, divided by utilisation u."""
    if not (0 < u <= 1):
        raise ValueError("utilisation u must be in (0, 1]")
    return gpu_seconds_per_query / 3600.0 * rate_per_hour / u


def cost_per_correct(c_train: float, cost_per_query: float, accuracy: float, volume: float) -> float:
    """(C_train + Q c) / (Q a) at deployment volume Q. Zero accuracy gives inf."""
    return math.inf if accuracy <= 0 else (c_train + volume * cost_per_query) / (volume * accuracy)


def break_even_marginal(c_train: float, c_api: float, c_ft: float, u: float = 1.0) -> float:
    """Q* = C_train / (c_API - c_FT/u); inf ('never') when c_API <= c_FT/u."""
    den = c_api - c_ft / u
    return c_train / den if den > 0 else math.inf


def break_even_per_correct(c_train: float, a_ft: float, a_api: float, c_api: float, c_ft: float,
                           u: float = 1.0) -> float:
    """Q* = C_train / (a_FT c_API / a_API - c_FT/u); inf when the denominator is <= 0."""
    den = a_ft * c_api / a_api - c_ft / u
    return c_train / den if den > 0 else math.inf


NO_BREAK_EVEN = "no break-even in range"


@dataclass(frozen=True)
class BreakEvenResult:
    status: str                 # 'finite' | NO_BREAK_EVEN | 'never'
    estimate: float             # point Q* (inf if denominator <= 0)
    ci_low: Optional[float]     # Fieller interval, only when status == 'finite'
    ci_high: Optional[float]
    p_denominator_le_0: float   # normal approximation
    denominator: float
    denominator_ci: tuple


def fieller_interval(num: float, den: float, var_num: float, var_den: float, cov: float = 0.0,
                     level: float = 0.95, df: Optional[float] = None):
    """Fieller interval for num/den. Returns (low, high) when the denominator is significantly
    different from 0 at `level`, otherwise None (the set is unbounded or the whole line)."""
    z = tdist.ppf(1 - (1 - level) / 2, df) if df is not None else norm.ppf(1 - (1 - level) / 2)
    a = den ** 2 - z ** 2 * var_den
    if a <= 0:
        return None
    b = num * den - z ** 2 * cov
    c = num ** 2 - z ** 2 * var_num
    disc = b ** 2 - a * c
    if disc < 0:
        return None
    r = math.sqrt(disc)
    return ((b - r) / a, (b + r) / a)


def break_even_with_ci(c_train: float, var_c_train: float, denominator: float, var_denominator: float,
                       cov: float = 0.0, level: float = 0.95, df: Optional[float] = None) -> BreakEvenResult:
    """Q* = C_train / D, D = c_API - c_FT/u (marginal form) or a_FT c_API/a_API - c_FT/u (per-correct form).
    The caller supplies the estimated D and its variance (from resampling replicates/items).
    - D's CI excludes 0 and D > 0: finite Q* with the Fieller interval (lower bound clipped at 0).
    - D's CI excludes 0 and D < 0: status 'never' (the API is cheaper at every volume).
    - D's CI contains 0: status 'no break-even in range'; no interval is reported (a percentile CI of an
      unbounded ratio is not computed), only P(D <= 0)."""
    sd = math.sqrt(var_denominator)
    p_le0 = float(norm.cdf(-denominator / sd)) if sd > 0 else float(denominator <= 0)
    zq = tdist.ppf(1 - (1 - level) / 2, df) if df is not None else norm.ppf(1 - (1 - level) / 2)
    dci = (denominator - zq * sd, denominator + zq * sd)
    est = c_train / denominator if denominator > 0 else math.inf
    if dci[0] <= 0 <= dci[1]:
        return BreakEvenResult(NO_BREAK_EVEN, est, None, None, p_le0, denominator, dci)
    if denominator < 0:
        return BreakEvenResult("never", math.inf, None, None, p_le0, denominator, dci)
    lo, hi = fieller_interval(c_train, denominator, var_c_train, var_denominator, cov, level, df)
    return BreakEvenResult("finite", est, max(lo, 0.0), hi, p_le0, denominator, dci)


# Scenario inputs of analysis/breakeven.py (A5's ranges, illustrative only).
BREAKEVEN_H = (0.4, 1.15)                 # GPU-hours for 5 runs
BREAKEVEN_INF_S = (20 / 3600 / 1000, 60 / 3600 / 1000)   # hours per item as written there
BREAKEVEN_API = (0.000126, 0.000169)
BREAKEVEN_RATE = (3.6, 6.0)


def breakeven_table() -> list:
    """Reproduces the table printed by analysis/breakeven.py (best / central / worst cases)."""
    H, INF, API = BREAKEVEN_H, BREAKEVEN_INF_S, BREAKEVEN_API
    cases = {"best": (H[0], INF[0], BREAKEVEN_RATE[0], API[1]),
             "central": (sum(H) / 2, sum(INF) / 2, 4.8, sum(API) / 2),
             "worst": (H[1], INF[1], BREAKEVEN_RATE[1], API[0])}
    rows = []
    for mode, fac in (("sync", 1.0), ("batch", 0.5)):
        for u in (1.0, 0.5, 0.2):
            for label, (h, hi_h, rate, api) in cases.items():
                c_train = h * rate
                c_ft = hi_h * rate / u      # hi_h is already hours per item, as in breakeven.py
                c_api = api * fac
                rows.append(dict(mode=mode, u=u, case=label, c_train=c_train, c_ft=c_ft, c_api=c_api,
                                 q_star=break_even_marginal(c_train, c_api, c_ft, 1.0)))
    return rows


# --------------------------------------------------------------------------------------
# 6. Pre-registered decision rule
# --------------------------------------------------------------------------------------
FT_BETTER = "FT better"
ICL_BETTER = "ICL better"
INCONCLUSIVE = "inconclusive"


def classify_contrast(result: CrossedResult, p_holm: float, margin: float, mdg: Optional[float] = None,
                      alpha: float = 0.05, allow_practically_equal: bool = True) -> Dict[str, object]:
    """Plan section 4 / R2.6. d = FT - ICL.
    (i) 'FT better' / 'ICL better' iff Holm-adjusted p < alpha (sign of the estimate);
    (iii) 'inconclusive' otherwise. The label 'equivalent' is never produced.
    Descriptive extras (outside the family, no p-value): practically_equal_descriptive iff p_holm >= alpha and
    the 95% CI lies inside (-margin, margin) (not for BillSum: allow_practically_equal=False);
    underpowered_a_priori iff mdg > margin (plan section 3: MDG from the validation-split runs vs m)."""
    if p_holm < alpha:
        label = FT_BETTER if result.mean > 0 else ICL_BETTER
    else:
        label = INCONCLUSIVE
    inside = (result.ci_low > -margin) and (result.ci_high < margin)
    return dict(label=label, mean=result.mean, p_holm=p_holm, ci_low=result.ci_low, ci_high=result.ci_high,
                ci_halfwidth=(result.ci_high - result.ci_low) / 2, margin=margin, mdg=mdg,
                ci_halfwidth_exceeds_mdg=None if mdg is None else (result.ci_high - result.ci_low) / 2 > mdg,
                practically_equal_descriptive=bool(allow_practically_equal and label == INCONCLUSIVE and inside),
                underpowered_a_priori=None if mdg is None else bool(mdg > margin))


def decide_family(results: Dict[str, CrossedResult], margins: Dict[str, float],
                  mdgs: Optional[Dict[str, float]] = None, alpha: float = 0.05,
                  no_equality: Sequence[str] = (),
                  expected_size: int = FAMILY_SIZE) -> Dict[str, Dict[str, object]]:
    """Holm over the 6 contrasts, then classify each. `margins[name]`: 0.03 (accuracy, BANKING77) or the
    ROUGE-L margin; `expected_size=4` gives the pre-registered sensitivity family without N=200 (L-15); contrast names in `no_equality` (BillSum) never receive the descriptive 'practically
    equal' flag."""
    holm = holm_family({k: r.p for k, r in results.items()}, alpha, expected_size)
    return {k: dict(classify_contrast(r, holm[k]["p_holm"], margins[k], None if mdgs is None else mdgs.get(k),
                                      alpha, k not in no_equality),
                    p=r.p, df=r.df, ci99_2=r.ci(0.992))
            for k, r in results.items()}
