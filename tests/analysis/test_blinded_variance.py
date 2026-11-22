"""The blinded variance script returns variance components, MDG, power and the trigger, never the effect."""

import dataclasses
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
spec = importlib.util.spec_from_file_location("blinded_variance", ROOT / "analysis" / "blinded_variance.py")
bv = importlib.util.module_from_spec(spec)
sys.modules["blinded_variance"] = bv
spec.loader.exec_module(bv)

FORBIDDEN = re.compile(r"mean|sign|delta|effect|diff|location|bar|average|avg|\bmu\b", re.I)


def make_d(n=1500, R=5, s_i=0.04, s_r=0.004, s_e=0.07, mu=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return mu + s_i * rng.standard_normal((n, 1)) + s_r * rng.standard_normal((1, R)) + s_e * rng.standard_normal((n, R))


def test_components_match_known_values():
    rep = bv.blinded_report(make_d(n=4000, R=8, s_i=0.05, s_r=0.01, s_e=0.1), "banking77", 1000)
    assert rep.s2_item == pytest.approx(0.05 ** 2, rel=0.15)
    assert rep.s2_res == pytest.approx(0.1 ** 2, rel=0.05)
    assert rep.s2_rep == pytest.approx(0.01 ** 2, abs=1.5e-4)
    assert rep.R_observed == 8 and rep.n_items == 4000


def test_opposite_sign_means_give_identical_output():
    a, b = make_d(mu=0.05), make_d(mu=-0.05)
    for task, n in (("banking77", 1000), ("banking77", 3000), ("banking77", 200), ("billsum", 1000)):
        ra, rb = bv.blinded_report(a, task, n), bv.blinded_report(b, task, n)
        assert ra == rb
        assert bv.format_report(ra) == bv.format_report(rb)
        assert json.dumps(ra.as_dict()) == json.dumps(rb.as_dict())
    assert bv.blinded_report(a) == bv.blinded_report(-a)          # exact negation too
    # and a large mean shift changes nothing
    assert bv.blinded_report(a) == bv.blinded_report(a + 7.0)


def test_no_field_or_signature_exposes_the_effect():
    names = [f.name for f in dataclasses.fields(bv.BlindedReport)]
    assert not [n for n in names if FORBIDDEN.search(n)], names
    for fn in (bv.blinded_report, bv.load_matrix, bv.overall_trigger, bv.format_report):
        ann = str(inspect.signature(fn).return_annotation)
        assert not FORBIDDEN.search(ann.replace("Optional", "")), (fn.__name__, ann)
        for p in inspect.signature(fn).parameters:
            assert not re.search(r"mean|sign", p, re.I), (fn.__name__, p)
    r = bv.blinded_report(make_d(mu=0.123456789), "banking77", 1000)
    # no reported value equals or encodes the shift
    for k, v in r.as_dict().items():
        assert "0.12345" not in str(v), k
    assert "0.12345" not in bv.format_report(r)
    assert set(r.as_dict()) == set(names)
    # the module calls only the two inference functions that take components and return power / a gap
    import ast
    used = {n.attr for n in ast.walk(ast.parse(Path(bv.__file__).read_text()))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "_inf"}
    assert used == {"power_at", "minimal_detectable_gap"}, used


def test_cli_output_has_no_effect(tmp_path, capsys):
    d = make_d(mu=0.0314159)
    np.save(tmp_path / "d.npy", d)
    np.savez(tmp_path / "d.npz", d=d, arm_labels=np.array(["ft", "icl"]), note="x")
    np.savetxt(tmp_path / "d.csv", d, delimiter=",")
    outs = []
    for ext in ("npy", "npz", "csv"):
        assert bv.main([str(tmp_path / f"d.{ext}"), "--task", "banking77", "--n-budget", "1000"]) == 0
        outs.append(capsys.readouterr().out)
    assert "0.0314" not in "".join(outs) and "3.14" not in "".join(outs)
    assert outs[0] == outs[1]
    bv.main([str(tmp_path / "d.npy"), "--task", "banking77", "--n-budget", "1000", "--json"])
    js = json.loads(capsys.readouterr().out)
    assert set(js) == {"cells", "r_extension"}
    assert not [k for k in js["cells"][0] if FORBIDDEN.search(k)]


def test_power_mdg_and_trigger_logic():
    # large residual noise: power for 3 points is low at R=8 -> trigger fires at N=1000
    noisy = bv.blinded_report(make_d(n=1000, R=5, s_i=0.05, s_r=0.03, s_e=0.3), "banking77", 1000)
    assert noisy.power_target_at_R_configured < 0.8 and noisy.trigger_fires is True
    assert noisy.mdg_at_R_configured > 0.03 and noisy.R_configured == 8 and noisy.R_max == 10
    # quiet: powered -> no trigger
    quiet = bv.blinded_report(make_d(n=3000, R=5, s_i=0.02, s_r=0.002, s_e=0.05), "banking77", 3000)
    assert quiet.power_target_at_R_configured > 0.8 and quiet.trigger_fires is False
    assert quiet.mdg_at_R_configured < 0.03
    # power and MDG are mutually consistent
    inf = bv._inf
    assert inf.power_at(noisy.mdg_at_R_configured, noisy.s2_item, noisy.s2_rep, noisy.s2_res, 1000, 8,
                        noisy.alpha) == pytest.approx(0.8, abs=1e-4)
    assert 0 < noisy.power_target_at_R_observed < noisy.power_target_at_R_configured < noisy.power_target_at_R_max <= 1


def test_n200_excluded_and_billsum_never_triggers():
    noisy = make_d(n=1000, R=5, s_i=0.05, s_r=0.03, s_e=0.3)
    r200 = bv.blinded_report(noisy, "banking77", 200)
    assert r200.trigger_applicable is False and r200.trigger_fires is None and "excluded" in r200.note
    rb = bv.blinded_report(noisy, "billsum", 1000)
    assert rb.trigger_fires is None and rb.R_configured == 5 and rb.R_max == 5
    reps = [bv.blinded_report(noisy, "banking77", n) for n in (200, 1000, 3000)]
    ov = bv.overall_trigger(reps)
    assert ov["cells_evaluated"] == [1000, 3000] and ov["extend_R_to_max"] is True
    # N=200 alone cannot fire the extension
    assert bv.overall_trigger([r200])["extend_R_to_max"] is None


def test_input_validation(tmp_path):
    with pytest.raises(ValueError):
        bv.blinded_report(np.ones(5))
    with pytest.raises(ValueError):
        bv.blinded_report(np.ones((10, 1)))
    with pytest.raises(ValueError):
        bv.blinded_report(np.array([[1.0, np.nan], [0.0, 1.0]]))
    np.savez(tmp_path / "x.npz", other=np.ones((3, 3)))
    with pytest.raises(ValueError, match="'d'"):
        bv.load_matrix(str(tmp_path / "x.npz"))
