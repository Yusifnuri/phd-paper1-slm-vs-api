#!/usr/bin/env python3
"""Static test-isolation check: only the allowed readers may reference a test split.

    python scripts/verify_test_isolation.py            # exit 0 if clean, 1 with a list of violations

Parses every .py file under src/, scripts/ and analysis/ (AST, so comments and docstring prose do
not matter) and flags
  * a string constant that names a test split: "test", "ca_test", or a slice such as "test[:10]";
  * any identifier or import of the guarded readers (`load_us_test_subset_for_final_eval`,
    `_US_TEST_SPLIT`, `load_banking77_test`, `load_billsum_test`).
Files in ALLOWED_READERS are exempt: they are the reviewed places that touch a test split. Adding a
file to that set is a protocol decision, not a convenience.
Generalises tests/train/test_train_offline.py::test_no_test_split_reference_in_sources.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parent.parent

ALLOWED_READERS = frozenset({
    "src/train/billsum.py",            # the guarded reader (checked by tests/train/test_billsum.py)
    "src/final/final_eval.py",         # the single final_eval entry
    "scripts/freeze_test_split.py",    # BANKING77-only fingerprint script (older)
    "scripts/freeze_test_splits.py",   # fingerprints for BANKING77, BillSum subset, LEDGAR placeholder
    "scripts/verify_test_isolation.py",  # this file names the patterns
})
SCAN_DIRS = ("src", "scripts", "analysis")
GUARDED_NAMES = frozenset({"load_us_test_subset_for_final_eval", "_US_TEST_SPLIT",
                           "load_banking77_test", "load_billsum_test"})
_SPLIT_RE = re.compile(r"^(ca_)?test(\[.*\])?$")


def violations_in_source(source: str) -> List[str]:
    """Human-readable findings for one file's source text."""
    out: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _SPLIT_RE.match(node.value):
            out.append(f"line {node.lineno}: string {node.value!r} names a test split")
        elif isinstance(node, ast.Name) and node.id in GUARDED_NAMES:
            out.append(f"line {node.lineno}: reference to {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in GUARDED_NAMES:
            out.append(f"line {node.lineno}: reference to {node.attr}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[-1] in GUARDED_NAMES:
                    out.append(f"line {node.lineno}: import of {alias.name}")
        elif isinstance(node, ast.FunctionDef) and node.name in GUARDED_NAMES:
            out.append(f"line {node.lineno}: definition of {node.name}")
    return out


def scan(root: Path = ROOT, allowed: Iterable[str] = ALLOWED_READERS,
         dirs: Sequence[str] = SCAN_DIRS) -> List[str]:
    allowed = set(allowed)
    found: List[str] = []
    for d in dirs:
        for f in sorted((root / d).rglob("*.py")):
            rel = f.relative_to(root).as_posix()
            if rel in allowed:
                continue
            found += [f"{rel}: {v}" for v in violations_in_source(f.read_text(encoding="utf-8"))]
    return found


def main() -> int:
    bad = scan()
    if bad:
        print("TEST ISOLATION VIOLATIONS:")
        print("\n".join("  " + b for b in bad))
        return 1
    print("test isolation OK: only the allowed readers reference a test split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
