"""Offline tests for the tuned procedure (tuning.py, config replicate fields, frozen arms, stage 2)."""

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.train import task_spec, tuning
from src.train.config import BASE_LR_GRID, TrainConfig, config_from_args, lr_grid

ROOT = Path(__file__).resolve().parents[2]
HF = "Qwen/Qwen3-4B-Instruct-2507"


# ------------------------------------------------------------------ schedule and validation size

def test_replicate_seeds_and_config_override():
    assert tuning.replicate_seeds(0) == (0, 100) and tuning.replicate_seeds(3) == (3, 103)
    c = TrainConfig(model="m", replicate=2, seed=7, pool_seed=9)
    assert (c.pool_seed, c.seed) == (2, 102)
    with pytest.raises(ValueError):
        tuning.replicate_seeds(-1)


def test_tune_only_on_replicate_zero():
    TrainConfig(model="m", tune=True, replicate=0)
    for r in (None, 1):
        with pytest.raises(ValueError):
            TrainConfig(model="m", tune=True, replicate=r)


def test_max_epochs_val_size_schedule():
    assert [tuning.max_epochs(n) for n in (200, 1000, 2999, 3000)] == [10, 10, 10, 5]
    assert [tuning.val_size(n) for n in (77, 200, 1000, 3000)] == [77, 200, 1000, 1000]
    # N=200, B=10: 10 epochs * 20 steps = 200 = S_min; N=1000, B=16: 10 * 63; N=3000: 5 * 188
    assert tuning.schedule_steps(200, 10, 200) == 200
    assert tuning.schedule_steps(1000, 16, 200) == 630
    assert tuning.schedule_steps(3000, 16, 200) == 940
    assert tuning.schedule_steps(50, 3, 200) == 200  # floor


def test_checkpoint_steps_even_and_end_at_schedule():
    assert tuning.checkpoint_steps(200) == [40, 80, 120, 160, 200]
    assert tuning.checkpoint_steps(630)[-1] == 630 and len(tuning.checkpoint_steps(630)) == 5
    assert tuning.checkpoint_steps(3) == [1, 2, 3]  # duplicates removed, never step 0
    with pytest.raises(ValueError):
        tuning.checkpoint_steps(0)


def test_schedule_and_stop_step_in_config():
    c = TrainConfig(model="m", n=1000, schedule_total=630, stop_step=252)
    assert c.schedule_length() == 630 and c.steps_to_train() == 252
    assert TrainConfig(model="m", n=1000, epochs=3).steps_to_train() == 200  # untuned: S_min floor (3*63=189)
    assert TrainConfig(model="m", n=5000, epochs=3).steps_to_train() == 3 * 313
    with pytest.raises(ValueError):
        TrainConfig(model="m", n=1000, schedule_total=100, stop_step=200)


def test_cli_parses_new_flags():
    c = config_from_args(["--model", "qwen3-4b", "--n", "200", "--replicate", "0", "--tune",
                          "--frozen-arms", "x.json", "--sampling", "stratified"])
    assert c.tune and c.replicate == 0 and c.frozen_arms == "x.json" and c.pool_seed == 0
    c2 = config_from_args(["--model", "qwen3-4b", "--n", "200", "--replicate", "1",
                           "--frozen-arms", "x.json", "--arm-sampling", "stratified"])
    assert c2.arm_sampling == "stratified" and c2.seed == 101


# ------------------------------------------------------------------ early stopping and selection

def _ck(step, acc, loss):
    return {"step": step, "accuracy": acc, "val_loss": loss}


def test_early_stopper_patience_two_with_loss_tiebreak():
    s = tuning.EarlyStopper("accuracy")
    assert not s.update(_ck(40, 0.5, 1.0))
    assert not s.update(_ck(80, 0.6, 0.9))
    assert not s.update(_ck(120, 0.6, 0.8))   # same accuracy, lower loss = improvement
    assert not s.update(_ck(160, 0.6, 0.85))  # worse: bad = 1
    assert s.update(_ck(200, 0.55, 0.7))      # worse: bad = 2 -> stop


def test_early_stopper_loss_metric():
    s = tuning.EarlyStopper("loss")
    assert not s.update(_ck(1, None, 2.0))
    assert not s.update(_ck(2, None, 1.5))
    assert not s.update(_ck(3, None, 1.6))
    assert s.update(_ck(4, None, 1.7))


def _trial(lr, cks, status="ok"):
    return {"lr": lr, "schedule_steps": 200, "status": status, "checkpoints": cks}


def test_select_best_checkpoint_across_trials():
    trials = [
        _trial(5e-5, [_ck(40, 0.40, 2.0), _ck(80, 0.50, 1.5)]),
        _trial(1e-4, [_ck(40, 0.55, 1.4), _ck(80, 0.70, 1.0), _ck(120, 0.70, 1.1)]),
        _trial(2e-4, [_ck(40, 0.65, 1.2), _ck(80, 0.70, 0.9)]),  # same acc, lower loss: wins
        _trial(4e-4, [], status="nonfinite"),
    ]
    sel = tuning.select(trials, BASE_LR_GRID, "accuracy", steps_per_epoch=20)
    assert sel["lr"] == 2e-4 and sel["stop_step"] == 80 and sel["epochs_star"] == 4.0
    assert sel["schedule_steps"] == 200 and sel["at_grid_edge"] is None and sel["n_trials"] == 4


def test_select_ties_keep_fewer_steps_and_flag_edges():
    trials = [_trial(1e-4, [_ck(40, 0.8, 1.0), _ck(80, 0.8, 1.0)])]
    sel = tuning.select(trials, BASE_LR_GRID, "accuracy", 20)
    assert sel["stop_step"] == 40 and sel["at_grid_edge"] == "low"
    sel = tuning.select([_trial(8e-4, [_ck(40, 0.8, 1.0)])], BASE_LR_GRID, "accuracy", 20)
    assert sel["at_grid_edge"] == "high"


def test_select_ignores_nonfinite_and_fails_if_nothing_finite():
    trials = [_trial(1e-4, [_ck(40, float("nan"), float("nan"))]),
              _trial(2e-4, [_ck(40, 0.3, 2.0)])]
    assert tuning.select(trials, BASE_LR_GRID, "accuracy", 20)["lr"] == 2e-4
    with pytest.raises(RuntimeError):
        tuning.select([_trial(1e-4, [_ck(40, float("nan"), 1.0)]), _trial(2e-4, [], "nonfinite")],
                      BASE_LR_GRID, "accuracy", 20)


def test_select_loss_metric_for_summarisation():
    trials = [_trial(1e-4, [_ck(40, None, 1.2)]), _trial(2e-4, [_ck(40, None, 1.1)])]
    assert tuning.select(trials, BASE_LR_GRID, "loss", 20)["lr"] == 2e-4


def test_lr_grid_shifted_after_pilot_g5_keeps_four_trials():
    assert lr_grid("lora") == (1e-4, 2e-4, 4e-4, 8e-4)  # pilot: top value 4e-4 chosen, 5e-5 worst
    assert len(lr_grid("rslora")) == 4 and lr_grid("rslora")[-1] == pytest.approx(2e-4)


# ------------------------------------------------------------------ frozen arms

def test_freeze_arm_is_write_once(tmp_path):
    path = str(tmp_path / "arms.json")
    key = tuning.arm_key("banking77", HF, "lora", 200, "stratified")
    h1 = tuning.freeze_arm(path, key, {"lr": 1e-4, "stop_step": 80})
    assert tuning.freeze_arm(path, key, {"stop_step": 80, "lr": 1e-4}) == h1  # same entry: no-op
    with pytest.raises(tuning.FrozenArmConflict):
        tuning.freeze_arm(path, key, {"lr": 2e-4, "stop_step": 80})
    assert tuning.load_arm(path, key)["lr"] == 1e-4
    with pytest.raises(KeyError):
        tuning.load_arm(path, "missing")
    assert json.loads(Path(path).read_text())[key]["stop_step"] == 80


def _arm(**kw):
    a = {"lr": 2e-4, "schedule_steps": 200, "stop_step": 120, "effective_batch": 10, "s_min": 200}
    a.update(kw)
    return a


def test_apply_frozen_arm_sets_procedure_and_guards(tmp_path):
    from src.train.train import apply_frozen_arm
    path = str(tmp_path / "arms.json")
    tuning.freeze_arm(path, tuning.arm_key(task_spec.TASK, HF, "lora", 200, "stratified"), _arm())
    cfg = TrainConfig(model="qwen3-4b", n=200, sampling="stratified", replicate=1, frozen_arms=path)
    new, info = apply_frozen_arm(cfg, HF)
    assert (new.lr, new.schedule_length(), new.steps_to_train()) == (2e-4, 200, 120)
    assert info["arm_key"].endswith("n200|stratified") and len(info["frozen_arms_sha256"]) == 64
    # random-pool final reusing the stratified arm
    rnd = TrainConfig(model="qwen3-4b", n=200, sampling="random", replicate=1, frozen_arms=path,
                      arm_sampling="stratified")
    assert apply_frozen_arm(rnd, HF)[0].stop_step == 120
    with pytest.raises(KeyError):  # no random arm frozen
        apply_frozen_arm(TrainConfig(model="qwen3-4b", n=200, replicate=1, frozen_arms=path), HF)
    with pytest.raises(ValueError):  # replicate 0 is the tuning replicate
        apply_frozen_arm(TrainConfig(model="qwen3-4b", n=200, sampling="stratified", replicate=0,
                                     frozen_arms=path), HF)
    with pytest.raises(ValueError):  # batch must match the tuned arm
        apply_frozen_arm(TrainConfig(model="qwen3-4b", n=200, sampling="stratified", replicate=1,
                                     frozen_arms=path, effective_batch=20), HF)
    assert apply_frozen_arm(TrainConfig(model="qwen3-4b", n=200), HF)[1] is None  # no file: untouched


def test_train_loop_has_inrun_checkpoints_not_retraining():
    src = (ROOT / "src/train/train.py").read_text(encoding="utf-8")
    assert "def select_trials" not in src and "EPOCH_GRID" not in src
    assert "on_checkpoint" in src and "cfg.schedule_length()" in src and "cfg.steps_to_train()" in src


# ------------------------------------------------------------------ stage 2 orchestrator

def _stage2():
    spec = importlib.util.spec_from_file_location("run_stage2", ROOT / "scripts/run_stage2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage2_dry_run_plans_tunes_then_finals(tmp_path):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_stage2.py"), "--dry-run",
         "--frozen-arms", str(tmp_path / "a.json"), "--results", str(tmp_path / "r.jsonl")],
        capture_output=True, text=True, cwd=ROOT, check=True).stdout
    assert out.count("--tune") == 4 and "14 steps to run" in out  # kNN 2 tunes + 5 finals, FT the same
    assert out.count("--replicate 0") == 4 and out.count("--arm-sampling stratified") == 2
    assert out.index("src.baselines.knn") < out.index("src.train.train")  # cheap baseline first


def _final(n, sampling, acc, unparse=10, first=3.0, last=0.5, rep=1):
    return {"status": "ok", "model": HF, "technique": "lora", "n": n, "sampling": sampling, "replicate": rep,
            "frozen_arm": {"arm_key": "k"}, "accuracy": acc, "n_eval": 1000, "n_unparseable": unparse,
            "first_train_loss": first, "final_train_loss": last}


def test_gate_report_pass_fail_and_incomplete():
    m = _stage2()
    arms = {tuning.arm_key(task_spec.TASK, HF, "lora", 200, "stratified"): {"at_grid_edge": None},
            tuning.arm_key(task_spec.TASK, HF, "lora", 1000, "random"): {"at_grid_edge": None}}
    rows = [_final(200, "stratified", 0.70, rep=1), _final(200, "stratified", 0.73, rep=2),
            _final(1000, "random", 0.82, rep=1), _final(1000, "random", 0.84, rep=2)]
    rep = m.gate_report(rows, arms, HF)
    assert rep["G1"]["pass"] and rep["G4"]["pass"] and rep["G5"]["pass"] and rep["G6"]["pass"]
    assert rep["G7"]["pass"] and rep["G2"]["pass"] is None
    assert rep["overall"] == "INCOMPLETE"  # kNN gates cannot be checked yet
    bad = rows[:3] + [_final(1000, "random", 0.90, unparse=90)]  # spread 8 pts, 9% invalid
    rep = m.gate_report(bad, arms, HF)
    assert rep["G1"]["pass"] is False and rep["G6"]["pass"] is False and rep["overall"] == "FAIL"
    assert m.gate_report([], {}, HF)["G4"]["pass"] is None
    edge = dict(arms)
    edge[next(iter(edge))] = {"at_grid_edge": "high"}
    assert m.gate_report(rows, edge, HF)["G5"]["pass"] is False


def _knn(n, sampling, acc, rep):
    return {"status": "ok", "model": "knn:x", "technique": "knn", "n": n, "sampling": sampling,
            "replicate": rep, "accuracy": acc, "n_eval": 1000, "n_unparseable": 0}


def test_gate_report_g2_g3_against_knn_on_matched_replicates():
    m = _stage2()
    arms = {tuning.arm_key(task_spec.TASK, HF, "lora", 200, "stratified"): {"at_grid_edge": None},
            tuning.arm_key(task_spec.TASK, HF, "lora", 1000, "random"): {"at_grid_edge": None}}
    ft = [_final(200, "stratified", 0.70, rep=1), _final(200, "stratified", 0.73, rep=2),
          _final(1000, "random", 0.82, rep=1), _final(1000, "random", 0.84, rep=2)]
    knn = [_knn(200, "stratified", 0.74, 1), _knn(200, "stratified", 0.76, 2),   # FT -3.5 pts: pass
           _knn(1000, "random", 0.90, 1), _knn(1000, "random", 0.90, 2),       # FT -7 pts: fail
           _knn(200, "random", 0.10, 1)]                                         # other condition: ignored
    rep = m.gate_report(ft + knn, arms, HF)
    assert rep["G2"]["pass"] and rep["G2"]["ft_minus_knn_mean"] == pytest.approx(-0.035)
    assert rep["G3"]["pass"] is False and rep["overall"] == "FAIL"
    knn[2]["accuracy"] = knn[3]["accuracy"] = 0.80
    assert m.gate_report(ft + knn, arms, HF)["overall"] == "PASS"
