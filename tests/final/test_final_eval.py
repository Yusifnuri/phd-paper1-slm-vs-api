"""Every refusal path of final_eval, using a temporary git repository and a fake loader."""
import datetime as dt
import json
import subprocess

import pytest

from src.final import ledger
from src.final.final_eval import TestIsolationError, final_eval

FIXED = lambda: dt.datetime(2026, 12, 20, 12, 0, tzinfo=dt.timezone.utc)  # noqa: E731


def run(env, **over):
    over.setdefault("now", FIXED)
    return final_eval(env.config, **env.kwargs(**over))


def refused(env, code, **over):
    with pytest.raises(TestIsolationError) as e:
        run(env, **over)
    assert e.value.code == code
    assert env.calls == 0, "loader must not be called when a check fails"


def git(env, *a):
    return subprocess.run(["git", "-C", str(env.repo), *a], check=True, capture_output=True, text=True).stdout


def test_happy_path_logs_one_line(env):
    res = run(env, run_id="r1")
    assert env.calls == 1 and res.data["hash"] == env.test_hash
    lines = ledger.read_ledger(env.ledger)
    assert len(lines) == 1
    rec = lines[0]
    assert rec["config_hash"] == ledger.config_hash(env.config)
    assert rec["test_hash"] == env.test_hash and rec["run_id"] == "r1"
    assert rec["utc"] == "2026-12-20T12:00:00Z"
    assert rec["git_commit"] == git(env, "rev-parse", "HEAD").strip()
    assert rec["status"] == "ok"


def test_refuses_missing_protocol_tag(env):
    git(env, "tag", "-d", "prereg-v1")
    refused(env, "tag_missing")


def test_refuses_missing_arms_tag(env):
    git(env, "tag", "-d", "arms-A")
    refused(env, "tag_missing")


def test_tag_names_are_parameters(env):
    refused(env, "tag_missing", protocol_tag="prereg-v2")
    git(env, "tag", "my-proto")
    git(env, "tag", "my-arms-a")
    run(env, protocol_tag="my-proto", arms_tags={"A": "my-arms-a"})
    assert env.calls == 1


def test_tag_must_be_ancestor_of_head(env):
    git(env, "checkout", "-q", "-b", "side")
    (env.repo / "x.txt").write_text("x")
    env.commit("side")
    git(env, "tag", "arms-side")
    git(env, "checkout", "-q", "master") if "master" in git(env, "branch") else git(env, "checkout", "-q", "main")
    refused(env, "tag_not_ancestor", arms_tags={"A": "arms-side"})


def test_family_b_needs_both_arms_tags(env):
    env.write_configs(family="B")
    env.commit("family b")
    refused(env, "tag_missing")  # arms-B absent
    git(env, "tag", "arms-B")
    run(env)
    assert env.calls == 1


def test_refuses_config_not_listed(env):
    env.config["lr"] = 0.001  # changed configuration -> different hash
    refused(env, "config_not_listed")


def test_refuses_uncommitted_configs_file(env):
    env.configs.write_text(env.configs.read_text() + " ")
    refused(env, "configs_modified")
    env.configs.unlink()
    git(env, "rm", "-q", "--cached", "final_configs.json")
    env.configs.write_text("{}")
    refused(env, "configs_not_committed")


def test_refuses_task_mismatch(env):
    refused(env, "task_mismatch", task="billsum")


def test_refuses_family_missing(env):
    env.write_configs(family=None)
    env.commit("no family")
    refused(env, "arm_family_missing")


def test_refuses_changed_frozen_arms(env):
    env.arms.write_text(json.dumps({"arm": {"lr": 0.001}}) + "\n")
    refused(env, "frozen_arms_changed")


def test_refuses_wrong_recorded_arms_hash(env):
    env.write_configs(arms_hash="0" * 64)
    env.commit("bad hash")
    refused(env, "frozen_arms_changed")


def test_refuses_missing_fingerprint_and_placeholder(env):
    env.fingerprint.write_text(json.dumps({"banking77": {"status": "PLACEHOLDER_NOT_FROZEN", "sha256": None}}))
    refused(env, "fingerprint_missing")
    env.fingerprint.unlink()
    refused(env, "fingerprint_missing")


def test_second_read_refused_then_allowed_with_infrastructure_reason(env):
    run(env)
    assert env.calls == 1
    with pytest.raises(TestIsolationError) as e:
        run(env)
    assert e.value.code == "already_read" and env.calls == 1
    # non-infrastructure reasons and missing detail are refused
    for reason, detail in (("bad_metric", "x"), ("crash", ""), ("crash", None)):
        with pytest.raises(TestIsolationError) as e:
            run(env, rerun_reason=reason, rerun_detail=detail)
        assert e.value.code == "rerun_reason_invalid"
    assert env.calls == 1
    run(env, rerun_reason="out_of_memory", rerun_detail="CUDA OOM at item 812, job killed", run_id="r2")
    assert env.calls == 2
    lines = ledger.read_ledger(env.ledger)
    assert len(lines) == 2 and lines[1]["rerun_reason"] == "out_of_memory" and lines[1]["prior_reads"] == 1
    assert lines[0]["rerun_reason"] is None


def test_rerun_reason_without_prior_read_refused(env):
    refused(env, "rerun_without_prior_read", rerun_reason="crash", rerun_detail="x")


def test_different_config_or_test_set_is_a_separate_read(env):
    run(env)
    other = dict(env.config, n=1000)
    env.write_configs(config=other)  # list both? rewrite lists only `other`
    env.commit("other config")
    final_eval(other, **env.kwargs(now=FIXED))
    assert env.calls == 2


def test_fingerprint_mismatch_logs_and_withholds(env):
    def bad_loader():
        env.calls += 1
        return {"texts": ["y"], "hash": "ee" * 32}

    with pytest.raises(TestIsolationError) as e:
        run(env, loader=bad_loader)
    assert e.value.code == "fingerprint_mismatch"
    lines = ledger.read_ledger(env.ledger)
    assert len(lines) == 1 and lines[0]["status"] == "fingerprint_mismatch"


def test_loader_failure_logs_nothing(env):
    def boom():
        raise OSError("network down")

    with pytest.raises(OSError):
        run(env, loader=boom)
    assert ledger.read_ledger(env.ledger) == []


def test_refuses_when_committed_ledger_was_edited(env):
    run(env)
    env.commit("ledger v1")
    lines = env.ledger.read_text().splitlines()
    env.ledger.write_text(lines[0].replace('"ok"', '"edited"') + "\n")  # rewrite history
    before = env.calls
    with pytest.raises(TestIsolationError) as e:
        run(env, rerun_reason="crash", rerun_detail="x")
    assert e.value.code == "ledger_invalid" and env.calls == before


def test_refuses_corrupt_ledger(env):
    env.ledger.write_text("not json\n")
    refused(env, "ledger_invalid")


def test_default_loaders_are_wired_but_not_called_on_refusal(env, monkeypatch):
    from src.final import final_eval as fe
    called = []
    monkeypatch.setitem(fe.DEFAULT_LOADERS, "banking77", lambda: called.append(1) or {"hash": env.test_hash})
    git(env, "tag", "-d", "arms-A")
    with pytest.raises(TestIsolationError):
        final_eval(env.config, **env.kwargs(loader=None))
    assert called == []
    git(env, "tag", "arms-A")
    final_eval(env.config, **env.kwargs(loader=None))
    assert called == [1]
    assert set(fe.DEFAULT_LOADERS) == {"banking77", "billsum"}
