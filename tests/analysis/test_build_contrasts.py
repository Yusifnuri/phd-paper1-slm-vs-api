"""Offline tests of analysis/build_contrasts.py and analysis/report_tables.py on synthetic prediction files."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load("build_contrasts")
rt = _load("report_tables")
from src.eval.schema import PredictionRecord, load_predictions, write_predictions  # noqa: E402


# ---- synthetic record factory -----------------------------------------------------------------
def rec(run_id, arm, item, rep, correct=None, rouge=None, task="banking77", n=1000, system="m", technique="lora",
        pool_hash=None, gold="a", extra_extra=None, seed=None):
    ok = True if correct is None else True
    parsed = gold if (correct or correct is None) else ("zzz" if correct is False else None)
    extra = {"pool_seed": rep, "pool_hash": pool_hash or f"hash{rep}"}
    if rouge is not None:
        extra["rouge_l"] = rouge
    extra.update(extra_extra or {})
    return PredictionRecord(run_id=run_id, system=system, arm=arm, technique=technique, task=task, n_budget=n,
                            seed=seed, item_id=item, prediction="x", parsed=parsed, gold=gold, parse_ok=parsed is not None,
                            confidence=None, latency_ms=None, input_tokens=None, output_tokens=None, extra=extra)


def make_arm(arm, system, technique, probs, rng, task="banking77", n=1000, runs=1, rouge=False, scale=0.0):
    """probs[i, r]: success probability (accuracy) or mean ROUGE-L. Returns a list of records."""
    out = []
    nI, R = probs.shape
    for r in range(R):
        for k in range(runs):
            rid = f"{arm}-{system}-{task}-n{n}-rep{r}-run{k}"
            for i in range(nI):
                if rouge:
                    v = float(np.clip(probs[i, r] + scale * rng.standard_normal(), 0, 1))
                    out.append(rec(rid, arm, f"i{i}", r, rouge=v, task=task, n=n, system=system, technique=technique,
                                   correct=True))
                else:
                    out.append(rec(rid, arm, f"i{i}", r, correct=bool(rng.random() < probs[i, r]), task=task, n=n,
                                   system=system, technique=technique))
    return out


def synth(gap, nI=2000, R=6, seed=0, task="banking77", n=1000, rouge=False):
    rng = np.random.default_rng(seed)
    item = rng.uniform(0.4, 0.8, size=(nI, 1))
    rep = 0.01 * rng.standard_normal((1, R))
    base = np.clip(item + rep, 0.02, 0.97) if not rouge else np.clip(0.3 + 0.1 * item + rep, 0.05, 0.9)
    ft = make_arm("finetuned", "ft", "lora", base + gap / 2, rng, task, n, rouge=rouge, scale=0.05)
    icl = make_arm("api", "gpt", "icl_retrieved", base - gap / 2, rng, task, n, rouge=rouge, scale=0.05)
    return ft, icl


# ---- round trip -------------------------------------------------------------------------------
@pytest.mark.parametrize("gap", [0.03, -0.04, 0.0])
def test_round_trip_recovers_injected_gap(gap):
    ft, icl = synth(gap)
    m = bc.build_matrix(ft, icl, "banking77", 1000, expected_replicates=range(6))
    assert m.d.shape == (2000, 6)
    assert abs(m.d.mean() - gap) < 0.02          # SE about 0.006; tolerance > 3 SE
    res = bc.inf.crossed_contrast(m.d)
    assert res.ci_low - 0.005 < gap < res.ci_high + 0.005


def test_round_trip_rouge_l_from_extra_field():
    ft, icl = synth(0.02, nI=800, R=5, task="billsum", rouge=True)
    m = bc.build_matrix(ft, icl, "billsum", 1000, metric="rouge_l")
    assert abs(m.d.mean() - 0.02) < 0.01


def test_round_trip_through_jsonl_files(tmp_path):
    ft, icl = synth(0.03, nI=300, R=4)
    f1, f2 = tmp_path / "ft.jsonl", tmp_path / "icl.jsonl"
    for p, recs in ((f1, ft), (f2, icl)):
        by_run = {}
        for r in recs:
            by_run.setdefault(r.run_id, []).append(r)
        with open(p, "w") as fh:
            for rr in by_run.values():
                for r in rr:
                    fh.write(json.dumps(r.to_dict()) + "\n")
    m = bc.build_matrix(load_predictions(f1), load_predictions(f2), "banking77", 1000)
    out = tmp_path / "m.npz"
    m.save(str(out))
    m2 = bc.ContrastMatrix.load(str(out))
    assert np.array_equal(m.d, m2.d) and m2.item_ids == m.item_ids and m2.pool_hashes == m.pool_hashes


def test_alignment_is_on_item_id_not_order():
    ft, icl = synth(0.0, nI=50, R=3)
    rng = np.random.default_rng(1)
    shuffled = list(icl)
    rng.shuffle(shuffled)
    a = bc.build_matrix(ft, icl, "banking77", 1000)
    b = bc.build_matrix(ft, shuffled, "banking77", 1000)
    assert np.array_equal(a.d, b.d)


# ---- stochastic arm (D-17) ---------------------------------------------------------------------
def test_stochastic_arm_enters_as_item_level_mean():
    rng = np.random.default_rng(3)
    nI, R = 40, 3
    ft = make_arm("finetuned", "ft", "lora", np.ones((nI, R)), rng)          # always correct
    icl = []
    for r in range(R):
        for k, right in enumerate((True, False)):                            # run 0 right, run 1 wrong
            rid = f"api-rep{r}-run{k}"
            icl += [rec(rid, "api", f"i{i}", r, correct=right) for i in range(nI)]
    m = bc.build_matrix(ft, icl, "banking77", 1000, icl_runs=2)
    assert np.allclose(m.d, 0.5)
    with pytest.raises(bc.ContrastError, match="expected 1"):
        bc.build_matrix(ft, icl, "banking77", 1000, icl_runs=1)


# ---- explicit failure checks --------------------------------------------------------------------
def _pair(nI=30, R=3):
    return synth(0.0, nI=nI, R=R)


def test_duplicate_item_raises():
    ft, icl = _pair()
    with pytest.raises(bc.ContrastError, match="duplicate"):
        bc.build_matrix(ft + [ft[0]], icl, "banking77", 1000)


def test_missing_item_raises():
    ft, icl = _pair()
    drop = [r for r in icl if not (r.item_id == "i3" and r.extra["pool_seed"] == 1)]
    with pytest.raises(bc.ContrastError, match="item set differs"):
        bc.build_matrix(ft, drop, "banking77", 1000)


def test_missing_replicate_raises():
    ft, icl = _pair()
    with pytest.raises(bc.ContrastError, match="replicates differ"):
        bc.build_matrix(ft, [r for r in icl if r.extra["pool_seed"] != 2], "banking77", 1000)
    with pytest.raises(bc.ContrastError, match="expected"):
        bc.build_matrix(ft, icl, "banking77", 1000, expected_replicates=range(4))


def test_pool_hash_mismatch_missing_and_reused():
    ft, icl = _pair()
    bad = [rec(r.run_id, r.arm, r.item_id, r.extra["pool_seed"], correct=True,
               pool_hash="other" if r.extra["pool_seed"] == 1 else None, system=r.system,
               technique=r.technique) for r in icl]
    with pytest.raises(bc.ContrastError, match="pool hashes differ"):
        bc.build_matrix(ft, bad, "banking77", 1000)
    nohash = [PredictionRecord(**{**r.to_dict(), "extra": {"pool_seed": r.extra["pool_seed"]}}) for r in icl]
    with pytest.raises(bc.ContrastError, match="pool_hash"):
        bc.build_matrix(ft, nohash, "banking77", 1000)
    reused = [PredictionRecord(**{**r.to_dict(), "extra": {**r.extra, "pool_hash": "same"}}) for r in icl]
    with pytest.raises(bc.ContrastError, match="share a pool hash"):
        bc.build_matrix(reused, reused, "banking77", 1000)


def test_missing_replicate_field_and_rouge_field_errors():
    ft, icl = _pair()
    noreps = [PredictionRecord(**{**r.to_dict(), "extra": {"pool_hash": r.extra["pool_hash"]}}) for r in ft]
    with pytest.raises(bc.ContrastError, match="replicate"):
        bc.build_matrix(noreps, icl, "banking77", 1000)
    with pytest.raises(bc.ContrastError, match="rouge_l"):
        bc.build_matrix(ft, icl, "banking77", 1000, metric="rouge_l")
    big = [PredictionRecord(**{**r.to_dict(), "extra": {**r.extra, "rouge_l": 45.0}}) for r in ft]
    with pytest.raises(bc.ContrastError, match="outside"):
        bc.build_matrix(big, big, "banking77", 1000, metric="rouge_l")


def test_gold_mismatch_and_mixed_arms_and_empty():
    ft, icl = _pair()
    g = [PredictionRecord(**{**r.to_dict(), "gold": "b", "parsed": "b"}) for r in icl]
    with pytest.raises(bc.ContrastError, match="gold differs"):
        bc.build_matrix(ft, g, "banking77", 1000)
    other = [rec(r.run_id + "x", "finetuned", r.item_id, r.extra["pool_seed"], correct=True, system="other")
             for r in ft[:5]]
    with pytest.raises(bc.ContrastError, match="mix several"):
        bc.build_matrix(ft + other, icl, "banking77", 1000)
    with pytest.raises(bc.ContrastError, match="no records"):
        bc.build_matrix(ft, icl, "banking77", 3000)


def test_unparseable_counts_as_wrong():
    r = rec("r", "api", "i", 0, correct=None)
    bad = PredictionRecord(**{**r.to_dict(), "parsed": None, "parse_ok": False})
    assert bc.score_of(bad, "accuracy") == 0.0 and bc.score_of(r, "accuracy") == 1.0


# ---- six-contrast table --------------------------------------------------------------------------
def _six(gaps, nI=400, R=5):
    ms = {}
    for t in bc.TASKS:
        for k, n in enumerate(bc.NS):
            ft, icl = synth(gaps[(t, n)], nI=nI, R=R, seed=10 * k + (t == "billsum"), task=t, n=n,
                            rouge=(t == "billsum"))
            ms[(t, n)] = bc.build_matrix(ft, icl, t, n, "accuracy" if t == "banking77" else "rouge_l")
    return ms


def test_analyze_labels_holm_and_never_equivalent():
    gaps = {(t, n): 0.0 for t in bc.TASKS for n in bc.NS}
    gaps[("banking77", 1000)] = 0.10      # clear FT win
    gaps[("billsum", 3000)] = -0.10       # clear ICL win
    rows = bc.analyze(_six(gaps), sensitivity_M={"banking77": 9003})
    by = {r["cell"]: r for r in rows}
    assert len(rows) == 6 and [r["cell"] for r in rows][:3] == ["banking77_N200", "banking77_N1000", "banking77_N3000"]
    assert by["banking77_N1000"]["label"] == "FT better"
    assert by["billsum_N3000"]["label"] == "ICL better"
    assert {by[c]["label"] for c in by} <= {"FT better", "ICL better", "inconclusive"}
    assert all(r["p_holm"] >= r["p"] for r in rows)
    assert all(r["practically_equal_descriptive"] is False for r in rows if r["task"] == "billsum")
    assert "p_corrected_unadjusted" in by["banking77_N200"] and "p_corrected_unadjusted" not in by["billsum_N200"]
    assert "equivalent" not in json.dumps(rows, default=str).lower()
    assert "stratified" in by["banking77_N200"]["note"]


def test_analyze_family_size_checks():
    ms = _six({(t, n): 0.0 for t in bc.TASKS for n in bc.NS}, nI=100, R=3)
    with pytest.raises(bc.ContrastError, match="exactly"):
        bc.analyze({k: v for k, v in ms.items() if k != ("billsum", 200)})
    rows = bc.analyze({k: v for k, v in ms.items() if k[1] != 200}, exclude_n200=True)
    assert len(rows) == 4
    with pytest.raises(bc.ContrastError, match="metric"):
        bad = dict(ms)
        m = ms[("billsum", 200)]
        bad[("billsum", 200)] = bc.ContrastMatrix("billsum", 200, "accuracy", m.item_ids, m.replicates, m.d, m.pool_hashes)
        bc.analyze(bad)


# ---- report tables ---------------------------------------------------------------------------------
def test_contrast_tables_markdown_and_csv(tmp_path):
    rows = bc.analyze(_six({(t, n): 0.01 for t in bc.TASKS for n in bc.NS}, nI=150, R=4))
    t = rt.contrast_table(rows)
    md, csv_text = rt.to_markdown(t), rt.to_csv(t)
    assert md.count("\n| ") == 6 and len(csv_text.strip().split("\n")) == 7
    assert "equivalent" not in md.lower() and "equivalent" not in csv_text.lower()
    paths = rt.write_tables(t, str(tmp_path), "contrasts")
    assert all(Path(p).exists() for p in paths)
    with pytest.raises(ValueError):
        rt.contrast_table([{**rows[0], "label": "equivalent"}])


def test_cost_table_closed_form_and_never():
    inp = [dict(cell="b77_N1000", c_train=2.0, c_api=0.0002, c_ft=0.00005, a_ft=0.9, a_api=0.8)]
    t = rt.cost_table(inp)
    assert len(t["rows"]) == 3
    r1 = t["rows"][0]                                 # u = 1
    assert r1[2] == f"{2.0 / (0.0002 - 0.00005):,.0f}"
    assert t["rows"][2][2] == "never"                 # c_ft/u = 0.00025 > c_api
    md = rt.to_markdown(t)
    assert "never" in md
    with pytest.raises(ValueError, match="missing"):
        rt.cost_table([dict(cell="x")])


def test_cost_table_fieller_status():
    inp = [dict(cell="c", c_train=2.0, c_api=0.0002, c_ft=0.00005, a_ft=0.9, a_api=0.8, u=[1.0],
                var_denominator=(0.001) ** 2, var_c_train=0.01)]
    t = rt.cost_table(inp)
    assert t["rows"][0][4] == "no break-even in range"
