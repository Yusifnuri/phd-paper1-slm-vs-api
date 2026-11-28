"""The single `final_eval` entry: the only gate through which a test split may be read.

Refuses to load test data unless ALL of these hold (pre-registration section 5.3, D-04, D-08):
  (a) the git tags exist, are annotated (they carry a tagger and date) and are ancestors of HEAD: the protocol tag (default "prereg-v1") and
      the arms tag of the configuration's family ("arms-A": fine-tuned, kNN, local ICL; "arms-B":
      API arm, which requires both arms tags);
  (b) the configuration hash is listed in a COMMITTED, unmodified `final_configs.json`;
  (c) the `frozen_arms.json` sha256 recorded in that file equals the current file hash;
  (d) the expected test-set hash is present in `results/test_split_fingerprint.json`;
  (e) (config hash, test hash) was not read before, unless a logged infrastructure-failure reason
      is passed (D-04); and the ledger is still append-only against git HEAD.
Only then is the injected loader called. The loaded rows' hash must equal the fingerprint; one
line is appended to `test_reads.jsonl` for every read that happened (also on a hash mismatch, which
then raises). A loader that raises before returning data logs nothing (no row was read).

`final_configs.json` schema:
  {"schema": 1, "frozen_arms_sha256": "<hex>",
   "configs": {"<config_hash>": {"task": "banking77|billsum", "arm_family": "A|B", "note": "..."}}}
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from . import ledger

DEFAULT_PROTOCOL_TAG = "prereg-v1"
DEFAULT_ARMS_TAGS = {"A": "arms-A", "B": "arms-B"}
# Family B (API arm) is frozen after A (D-08): its read needs both arms tags.
_REQUIRED_ARMS = {"A": ("A",), "B": ("A", "B")}


class TestIsolationError(RuntimeError):
    """A test-read precondition failed; nothing was read."""
    __test__ = False  # not a pytest class

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass
class FinalRead:
    data: Any
    record: Dict[str, Any]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _check_tag(repo: Path, tag: str) -> None:
    r = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if r.returncode != 0:
        raise TestIsolationError("tag_missing", f"git tag {tag!r} does not exist")
    if _git(repo, "cat-file", "-t", f"refs/tags/{tag}").stdout.strip() != "tag":
        raise TestIsolationError("tag_not_annotated", f"git tag {tag!r} is lightweight; use `git tag -a`")
    if _git(repo, "merge-base", "--is-ancestor", r.stdout.strip(), "HEAD").returncode != 0:
        raise TestIsolationError("tag_not_ancestor", f"git tag {tag!r} is not an ancestor of HEAD")


def _check_committed_clean(repo: Path, path: Path) -> None:
    rel = path.resolve().relative_to(repo.resolve()).as_posix()
    if _git(repo, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise TestIsolationError("configs_not_committed", f"{rel} is not tracked by git")
    if _git(repo, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise TestIsolationError("configs_modified", f"{rel} differs from the committed version")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_banking77_test(revision: Optional[str] = None) -> Dict[str, Any]:
    """Default BANKING77 test loader (3,080 rows). Called only after every check has passed."""
    from datasets import load_dataset
    from src.train import data

    ds = load_dataset(data.DATASET_ID, split="test", revision=revision or data.REVISION)
    if len(ds) != data.N_TEST:
        raise RuntimeError(f"expected {data.N_TEST} test rows, got {len(ds)}")
    names = ds.features["label"].names
    texts, labels = list(ds["text"]), [names[i] for i in ds["label"]]
    return {"texts": texts, "labels": labels, "n": len(texts),
            "hash": data.rows_fingerprint(texts, labels)}


def load_billsum_test() -> Dict[str, Any]:
    """Default BillSum loader: the guarded reader of src/train/billsum.py (seeded 1,000 subset)."""
    from src.train import billsum

    return billsum.load_us_test_subset_for_final_eval()


DEFAULT_LOADERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "banking77": load_banking77_test,
    "billsum": load_billsum_test,
}


def final_eval(
    config: Mapping[str, Any],
    *,
    task: str,
    repo_root: str | Path,
    final_configs_path: str | Path,
    frozen_arms_path: str | Path,
    ledger_path: str | Path,
    fingerprint_path: str | Path,
    protocol_tag: str = DEFAULT_PROTOCOL_TAG,
    arms_tags: Optional[Mapping[str, str]] = None,
    loader: Optional[Callable[[], Dict[str, Any]]] = None,
    run_id: Optional[str] = None,
    rerun_reason: Optional[str] = None,
    rerun_detail: Optional[str] = None,
    now: Optional[Callable[[], _dt.datetime]] = None,
) -> FinalRead:
    """Run every gate, then load the test split via `loader` and log the read. See module doc."""
    repo = Path(repo_root)
    tags = dict(DEFAULT_ARMS_TAGS)
    tags.update(arms_tags or {})
    cfg_hash = ledger.config_hash(dict(config))

    # (b) committed final_configs.json listing this configuration; its family decides the tags.
    fc_path = Path(final_configs_path)
    _check_committed_clean(repo, fc_path)
    fc = json.loads(fc_path.read_text(encoding="utf-8"))
    entry = fc.get("configs", {}).get(cfg_hash)
    if entry is None:
        raise TestIsolationError("config_not_listed", f"config hash {cfg_hash} is not in {fc_path.name}")
    if entry.get("task") not in (None, task):
        raise TestIsolationError("task_mismatch", f"{fc_path.name} lists this config for task "
                                 f"{entry.get('task')!r}, not {task!r}")
    family = entry.get("arm_family")
    if family not in _REQUIRED_ARMS:
        raise TestIsolationError("arm_family_missing", "final_configs entry needs arm_family 'A' or 'B'")

    # (a) tags.
    _check_tag(repo, protocol_tag)
    for fam in _REQUIRED_ARMS[family]:
        _check_tag(repo, tags[fam])

    # (c) frozen_arms.json hash.
    fa = Path(frozen_arms_path)
    if not fa.exists():
        raise TestIsolationError("frozen_arms_missing", f"{fa} does not exist")
    if fc.get("frozen_arms_sha256") != _file_sha256(fa):
        raise TestIsolationError("frozen_arms_changed", "sha256 of frozen_arms.json differs from the "
                                 "hash recorded in final_configs.json")

    # (d) expected test hash.
    fp_path = Path(fingerprint_path)
    if not fp_path.exists():
        raise TestIsolationError("fingerprint_missing", f"{fp_path} does not exist")
    fp = json.loads(fp_path.read_text(encoding="utf-8")).get(task, {})
    expected = fp.get("sha256")
    if not expected or str(fp.get("status", "")).startswith("PLACEHOLDER"):
        raise TestIsolationError("fingerprint_missing", f"no frozen test fingerprint for task {task!r}")

    # (e) ledger: append-only against HEAD, and no repeat read without an infrastructure reason.
    try:
        ledger.check_append_only(ledger_path, repo)
        prior = ledger.previous_reads(ledger_path, cfg_hash, expected)
    except ledger.LedgerError as e:
        raise TestIsolationError("ledger_invalid", str(e)) from e
    if prior:
        if rerun_reason is None:
            raise TestIsolationError("already_read", f"(config {cfg_hash[:12]}, test {expected[:12]}) was "
                                     "read before; a repeat needs a logged infrastructure-failure reason (D-04)")
        try:
            ledger.validate_rerun(rerun_reason, rerun_detail)
        except ValueError as e:
            raise TestIsolationError("rerun_reason_invalid", str(e)) from e
    elif rerun_reason is not None:
        raise TestIsolationError("rerun_without_prior_read", "a rerun reason was given but there is no prior read")

    # All checks passed: only now is the dataset touched.
    chosen = loader or DEFAULT_LOADERS.get(task)
    if chosen is None:
        raise TestIsolationError("no_loader", f"no loader for task {task!r}")
    loaded = chosen()
    actual = loaded.get("hash") if isinstance(loaded, dict) else None
    stamp = (now or (lambda: _dt.datetime.now(_dt.timezone.utc)))()
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip())
    record = {
        "config_hash": cfg_hash, "test_hash": actual, "task": task, "arm_family": family,
        "utc": stamp.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git(repo, "rev-parse", "HEAD").stdout.strip(),
        "git_dirty": dirty, "run_id": run_id or uuid.uuid4().hex,
        "protocol_tag": protocol_tag, "arms_tags": [tags[f] for f in _REQUIRED_ARMS[family]],
        "rerun_reason": rerun_reason, "rerun_detail": rerun_detail if rerun_reason else None,
        "prior_reads": len(prior),
    }
    if actual != expected:
        record["status"] = "fingerprint_mismatch"
        record["expected_test_hash"] = expected
        ledger.append_record(ledger_path, record)
        raise TestIsolationError("fingerprint_mismatch", "loaded test rows do not match the frozen "
                                 "fingerprint; the read is logged and the data is withheld")
    record["status"] = "ok"
    ledger.append_record(ledger_path, record)
    return FinalRead(data=loaded, record=record)
