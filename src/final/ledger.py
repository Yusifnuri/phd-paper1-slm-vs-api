"""Append-only test-read ledger (`test_reads.jsonl`) and configuration hashing.

One JSON object per line. Lines are only ever appended; `check_append_only` additionally proves
that the committed version of the file (git HEAD) is a byte prefix of the working copy.
Standard library only.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reasons that D-04 accepts for repeating a read of the same (config hash, test hash).
INFRA_REASONS = ("crash", "out_of_memory", "interrupted_job", "api_outage")


class LedgerError(RuntimeError):
    """The ledger is malformed or no longer append-only."""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(config: Dict[str, Any]) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace) of a run configuration."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def read_ledger(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise LedgerError(f"{p}: blank line {n}; the ledger holds one JSON object per line")
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise LedgerError(f"{p}: line {n} is not valid JSON: {e}") from e
        if not isinstance(rec, dict) or "config_hash" not in rec or "test_hash" not in rec:
            raise LedgerError(f"{p}: line {n} lacks config_hash/test_hash")
        out.append(rec)
    return out


def previous_reads(path: str | Path, cfg_hash: str, test_hash: str) -> List[Dict[str, Any]]:
    return [r for r in read_ledger(path)
            if r["config_hash"] == cfg_hash and r["test_hash"] == test_hash]


def append_record(path: str | Path, record: Dict[str, Any]) -> None:
    """Append one line (exclusive lock, fsync). Never rewrites or truncates."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(record) + "\n"
    with open(p, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        if fh.tell() > 0:  # file must end with a newline, else a torn line would merge with ours
            with open(p, "rb") as rd:
                rd.seek(-1, os.SEEK_END)
                if rd.read(1) != b"\n":
                    raise LedgerError(f"{p} does not end with a newline; refusing to append")
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _git(repo: str | Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True)


def check_append_only(path: str | Path, repo: str | Path) -> None:
    """If the ledger is tracked at HEAD, the committed bytes must be a prefix of the working copy."""
    p = Path(path).resolve()
    rel = os.path.relpath(p, Path(repo).resolve())
    shown = _git(repo, "show", f"HEAD:{rel}")
    if shown.returncode != 0:
        return  # not committed yet: nothing to compare against
    now = p.read_bytes() if p.exists() else b""
    if not now.startswith(shown.stdout):
        raise LedgerError(f"{rel} is not append-only: the committed version is not a prefix of the "
                          "working copy (lines were edited, reordered or removed)")


def validate_rerun(reason: Optional[str], detail: Optional[str]) -> None:
    if reason not in INFRA_REASONS:
        raise ValueError(f"rerun reason must be one of {INFRA_REASONS}, got {reason!r}")
    if not detail or not detail.strip():
        raise ValueError("a logged infrastructure-failure detail (what failed, where) is required")
