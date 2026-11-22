"""Tests of analysis/inference.py against the existing simulation code and saved reference outputs."""

import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))              # src.eval.stats
sys.path.insert(0, str(ROOT / "analysis"))  # power_sim2 imports power_sim from here


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


inf = _load("inference")
ps2 = _load("power_sim2")
from src.eval.stats import holm_adjust as repo_holm  # noqa: E402


# ---- 1. crossed contrast equals power_sim2.a3 ---------------------------------------------------
@pytest.mark.parametrize("N,R,reg,delta", [(200, 5, "opt", 0.03), (1000, 8, "pess", 0.0), (3000, 8, "opt", 0.02),
                                           (1000, 5, "pess", 0.05)])
def test_crossed_equals_a3_binary(N, R, reg, delta):
    rng = np.random.default_rng(100 + N + R)
    for truth in ("cond", "super"):
        d = ps2.sim_b77(rng, 400, R, N, reg, delta, truth)
        for f in (None, N / ps2.M):
            z, df = ps2.a3(d, f)
            res = inf.crossed_contrast(d, f)
            assert res.t == pytest.approx(z, rel=1e-12, abs=1e-12)
            assert res.df == pytest.approx(df, rel=1e-10)


def test_crossed_equals_a3_continuous_and_a3_rejection_rule():
    rng = np.random.default_rng(5)
    n, R = 500, 5
    d = 0.01 + 0.04 * rng.standard_normal((n, 1)) + 0.004 * rng.standard_normal((1, R)) \
        + 0.055 * rng.standard_normal((n, R))
    z, df = ps2.a3(d)
    res = inf.crossed_contrast(d)
    assert res.t == pytest.approx(z, rel=1e-12) and res.df == pytest.approx(df, rel=1e-10)
    # same decision as rej(...) at several alphas (two-sided)
    for a in (0.05, 0.0083):
        assert bool(ps2.rej(z, df, a, True)) == (res.p < a)
    # CI is mean +- t_crit * se and contains the mean; 99.2% interval is wider
    assert res.ci_low < res.mean < res.ci_high
    lo2, hi2 = res.ci(0.992)
    assert lo2 < res.ci_low and hi2 > res.ci_high


def test_variance_formula_components_known_case():
    # d_ir = a_i + b_r exactly (no residual): s2_res = 0, components recovered by ANOVA
    a = np.array([1.0, -1.0, 0.5, -0.5]); b = np.array([2.0, 0.0, -2.0])
    d = a[:, None] + b[None, :]
    c = inf.variance_components(d)
    assert c["s2_res"] == pytest.approx(0.0, abs=1e-12)
    assert c["s2_item"] == pytest.approx(a.var(ddof=1)) and c["s2_rep"] == pytest.approx(b.var(ddof=1))
    res = inf.crossed_contrast(d)
    assert res.var == pytest.approx(a.var(ddof=1) / 4 + b.var(ddof=1) / 3)


def test_finite_population_correction_inflates_variance_only_through_rep_term():
    rng = np.random.default_rng(3)
    d = 0.02 + 0.05 * rng.standard_normal((300, 1)) + 0.02 * rng.standard_normal((1, 8)) \
        + 0.1 * rng.standard_normal((300, 8))
    base, corr = inf.crossed_contrast(d), inf.crossed_contrast(d, f=1000 / 9003)
    f = 1000 / 9003
    assert corr.var - base.var == pytest.approx(base.s2_rep * f / (1 - f))
    assert corr.mean == base.mean and corr.var >= base.var
    with pytest.raises(ValueError):
        inf.crossed_contrast(d, f=1.0)
    with pytest.raises(ValueError):
        inf.crossed_contrast(d[:, :1])


def test_type_i_error_near_nominal_conditional():
    # R=8, N=1000, opt regime, true delta 0: 600 sims (binomial SE about 0.9 points). Tolerance 2.5%-8.5%.
    rng = np.random.default_rng(11)
    rej = sum(inf.crossed_contrast(ps2.sim_b77(rng, 600, 8, 1000, "opt", 0.0, "cond")).p < 0.05
              for _ in range(600))
    assert 0.025 <= rej / 600 <= 0.085


# ---- 2. Holm -----------------------------------------------------------------------------------
def test_holm_matches_repo_and_hand_example():
    p = [0.001, 0.04, 0.03, 0.0100, 0.5, 0.2]
    assert np.allclose(inf.holm_adjust(p), repo_holm(p))
    assert inf.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.random(6) ** 3
        assert np.allclose(inf.holm_adjust(q), repo_holm(q))


def test_holm_family_requires_six():
    names = ["b77_200", "b77_1000", "b77_3000", "bs_200", "bs_1000", "bs_3000"]
    out = inf.holm_family(dict(zip(names, [0.001, 0.2, 0.9, 0.009, 0.01, 0.6])))
    assert out["b77_200"]["p_holm"] == pytest.approx(0.006) and out["b77_200"]["reject"]
    assert out["bs_200"]["p_holm"] == pytest.approx(0.045) and out["bs_200"]["reject"]
    assert out["bs_1000"]["p_holm"] == pytest.approx(0.045)  # monotone: max(4*0.01, 5*0.009)
    assert not out["b77_1000"]["reject"]
    with pytest.raises(ValueError):
        inf.holm_family({"a": 0.01, "b": 0.02})


# ---- 3. bootstrap ------------------------------------------------------------------------------
def test_bootstrap_fixed_seed_and_width():
    rng = np.random.default_rng(1)
    x = (rng.random((800, 5)) < 0.8).astype(float)
    a = inf.two_stage_bootstrap(x, n_boot=2000)
    b = inf.two_stage_bootstrap(x, n_boot=2000)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]           # fixed seed
    c = inf.two_stage_bootstrap(x, n_boot=2000, seed=1)
    assert (c["ci_low"], c["ci_high"]) != (a["ci_low"], a["ci_high"])
    assert a["ci_low"] < x.mean() < a["ci_high"]
    # iid binary data: half-width about 1.96*sqrt(p(1-p)/(n*R)); replicates add nothing here. Loose factor 1.5.
    se = math.sqrt(0.8 * 0.2 / x.size)
    half = (a["ci_high"] - a["ci_low"]) / 2
    assert 0.5 * 1.96 * se < half < 1.5 * 1.96 * se * math.sqrt(5)   # items are shared across replicates
    assert a["distinct_replicate_resamples"] == math.comb(9, 5) and a["small_R_warning"]


# ---- 4. minimal detectable gap -----------------------------------------------------------------
def test_mdg_properties_and_power_roundtrip():
    s = (0.04, 4e-4, 0.003)
    m = inf.minimal_detectable_gap(*s, 3080, 8)
    assert inf.power_at(m, *s, 3080, 8, 0.0083) == pytest.approx(0.8, abs=1e-6)
    assert inf.minimal_detectable_gap(*s, 3080, 5) > m                  # fewer replicates: larger gap
    assert inf.minimal_detectable_gap(*s, 3080, 8, alpha=0.025) < m     # looser alpha: smaller gap
    assert inf.minimal_detectable_gap(*s, 3080, 8, f=1000 / 9003) > m   # super-population needs more
    assert inf.power_at(0.0, *s, 3080, 8, 0.05) == pytest.approx(0.05, abs=1e-6)


# Tolerance 0.5 accuracy points: the reference MDG is a linear interpolation of 500-sim power cells on a
# 1-2 point grid (power MC error +-2 points ~ +-0.25 pt of gap; interpolation error similar), and ours
# is a plug-in with components averaged over 40 simulations. Observed differences were 0.1-0.35 points,
# ours slightly smaller (it ignores the estimation noise of the variance components).
@pytest.mark.parametrize("N,R,reg,dl", [(200, 8, "opt", 0.03), (1000, 8, "opt", 0.02),
                                        (1000, 8, "pess", 0.033), (3000, 8, "pess", 0.023),
                                        (1000, 5, "opt", 0.026), (200, 5, "pess", 0.07)])
def test_mdg_against_saved_reference(N, R, reg, dl):
    ref = json.loads((ROOT / "analysis/reference_outputs/sim2_mdg.json").read_text())
    row = next(r for r in ref if (r["N"], r["R"], r["reg"]) == (N, R, reg))["pw"]
    grid = sorted(float(g) for g in row)
    pw = [row[next(k for k in row if float(k) == g)]["cond_.0083"] for g in grid]
    ref_mdg = float(np.interp(0.8, pw, grid))
    rng = np.random.default_rng(1)
    comp = []
    for _ in range(40):
        c = inf.variance_components(ps2.sim_b77(rng, 3080, R, N, reg, dl, "cond"))
        comp.append([c["s2_item"], c["s2_rep"], c["s2_res"]])
    mine = inf.minimal_detectable_gap(*np.mean(comp, 0), 3080, R, alpha=0.0083)
    assert abs(mine - ref_mdg) < 0.005


# ---- 5. cost model -----------------------------------------------------------------------------
def test_breakeven_table_matches_script_output():
    out = subprocess.run([sys.executable, str(ROOT / "analysis/breakeven.py")], capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()[1:]
    theirs = []
    for line in out:
        mode, u, rest = line.split(" ", 2)
        for part in rest.split(" | "):
            case = part.split(":")[0]
            q = re.search(r"Q\*=(\S+)", part).group(1)
            c = re.search(r"C=\$([\d.]+)", part).group(1)
            theirs.append((mode, float(u), case, math.inf if q == "never" else float(q.replace(",", "")), float(c)))
    mine = inf.breakeven_table()
    assert len(mine) == len(theirs) == 18
    for r, (mode, u, case, q, c) in zip(mine, theirs):
        assert (r["mode"], r["u"], r["case"]) == (mode, u, case)
        assert r["c_train"] == pytest.approx(c, abs=0.05)               # script prints C to 0.1 dollar
        if math.isinf(q):
            assert math.isinf(r["q_star"])
        else:
            assert r["q_star"] == pytest.approx(q, abs=0.5)             # script prints Q* rounded to integers
    # spot values quoted in the plan
    by = {(r["mode"], r["u"], r["case"]): r["q_star"] for r in mine}
    assert round(by[("sync", 1.0, "best")], -2) == 9700 and round(by[("sync", 1.0, "central")], -3) == 40000


def test_cost_formulas():
    assert inf.cost_per_correct(10.0, 1e-4, 0.8, 1e5) == pytest.approx((10 + 10) / (1e5 * 0.8))
    assert inf.cost_per_correct(0, 1e-4, 0.0, 10) == math.inf
    assert inf.self_hosted_cost_per_query(36.0, 3.6, 0.5) == pytest.approx(36.0 / 3600 * 3.6 / 0.5)
    # marginal: at Q*, total costs are equal
    C, ca, cf, u = 5.0, 1.5e-4, 5e-5, 0.5
    q = inf.break_even_marginal(C, ca, cf, u)
    assert C + q * cf / u == pytest.approx(q * ca)
    assert inf.break_even_marginal(C, 1e-5, cf, u) == math.inf
    # per-correct: at Q*, cost per correct equal
    aft, aapi = 0.93, 0.90
    q2 = inf.break_even_per_correct(C, aft, aapi, ca, cf, u)
    assert inf.cost_per_correct(C, cf / u, aft, q2) == pytest.approx(inf.cost_per_correct(0, ca, aapi, q2))
    assert inf.break_even_per_correct(C, aft, aapi, 1e-5, cf, u) == math.inf


def test_fieller_and_no_break_even_result():
    # tight denominator: finite, interval brackets the estimate and satisfies the defining equation
    r = inf.break_even_with_ci(5.0, 0.1, 1e-4, (1e-5) ** 2)
    assert r.status == "finite" and r.ci_low < r.estimate < r.ci_high and r.p_denominator_le_0 < 1e-6
    z = 1.959963984540054
    for th in (r.ci_low, r.ci_high):
        assert (5.0 - th * 1e-4) ** 2 == pytest.approx(z**2 * (0.1 + th**2 * (1e-5) ** 2), rel=1e-6)
    # denominator CI contains 0: explicit "no break-even in range", no interval, P(D<=0) reported
    r = inf.break_even_with_ci(5.0, 0.1, 2e-5, (2e-5) ** 2)
    assert r.status == inf.NO_BREAK_EVEN == "no break-even in range"
    assert r.ci_low is None and r.ci_high is None and 0.1 < r.p_denominator_le_0 < 0.5
    # denominator clearly negative: 'never'
    r = inf.break_even_with_ci(5.0, 0.1, -1e-4, (1e-5) ** 2)
    assert r.status == "never" and r.estimate == math.inf and r.p_denominator_le_0 > 0.999
    # Fieller returns None when the denominator is not significant
    assert inf.fieller_interval(5.0, 1e-5, 0.1, (1e-5) ** 2) is None


# ---- 6. decision rule --------------------------------------------------------------------------
def _res(mean, se, df=30.0):
    from scipy.stats import t as tdist
    p = 2 * tdist.sf(abs(mean / se), df)
    crit = tdist.ppf(0.975, df)
    return inf.CrossedResult(mean=mean, var=se**2, se=se, df=df, t=mean / se, p=p, ci_low=mean - crit * se,
                             ci_high=mean + crit * se, level=0.95, n=100, R=8, ms_item=0, ms_rep=0, ms_res=0,
                             s2_item=0, s2_rep=0, s2_res=0, f=None)


def test_decision_rule_labels():
    results = {"b77_200": _res(0.06, 0.01), "b77_1000": _res(-0.05, 0.008), "b77_3000": _res(0.002, 0.003),
               "bs_200": _res(0.003, 0.02), "bs_1000": _res(0.0001, 0.0003), "bs_3000": _res(0.02, 0.005)}
    margins = {k: (0.03 if k.startswith("b77") else 0.01) for k in results}
    mdgs = {"b77_200": 0.05, "b77_1000": 0.02}
    out = inf.decide_family(results, margins, mdgs, no_equality=["bs_200", "bs_1000", "bs_3000"])
    assert out["b77_200"]["label"] == "FT better" and out["b77_1000"]["label"] == "ICL better"
    assert out["b77_3000"]["label"] == "inconclusive" and out["b77_3000"]["practically_equal_descriptive"]
    assert out["bs_200"]["label"] == "inconclusive" and not out["bs_200"]["practically_equal_descriptive"]
    assert out["bs_1000"]["label"] == "inconclusive" and not out["bs_1000"]["practically_equal_descriptive"]
    assert out["b77_200"]["underpowered_a_priori"] is True and out["b77_1000"]["underpowered_a_priori"] is False
    assert out["bs_3000"]["label"] == "FT better"
    assert {o["label"] for o in out.values()} <= {"FT better", "ICL better", "inconclusive"}
    assert "equivalent" not in json.dumps(out, default=str).lower().replace("practically_equal", "")


def test_unadjusted_significance_not_enough_under_holm():
    # the smallest of six p-values is multiplied by 6: p about 0.01 is below 0.05 but Holm p about 0.06
    res = {f"c{i}": _res(0.005, 0.01) for i in range(5)}
    res["c5"] = _res(0.027, 0.01)
    out = inf.decide_family(res, {k: 0.03 for k in res})
    assert out["c5"]["p"] < 0.05 and out["c5"]["p_holm"] >= 0.05 and out["c5"]["label"] == "inconclusive"


def test_holm_rejects_nan_and_out_of_range():
    import pytest
    from analysis.inference import holm_adjust
    with pytest.raises(ValueError):
        holm_adjust([0.001, 0.02, float("nan")])
    with pytest.raises(ValueError):
        holm_adjust([0.5, 1.5])


def test_decide_family_four_contrast_sensitivity():
    import numpy as np
    from analysis.inference import crossed_contrast, decide_family
    r = crossed_contrast(np.random.default_rng(0).normal(size=(50, 5)))
    out = decide_family({f"k{i}": r for i in range(4)}, {f"k{i}": 0.03 for i in range(4)}, expected_size=4)
    assert len(out) == 4
