#!/usr/bin/env python3
"""Compare installed packages with requirements/train.txt. Prints pip-installable missing
packages (one per line) on stdout; exits 1 if an installed version differs from its pin."""
import importlib.metadata as md
import re
import sys
from pathlib import Path

req = Path(__file__).resolve().parent.parent / "requirements" / "train.txt"
bad, missing = [], []
verify_only = False
for line in req.read_text().splitlines():
    if line.startswith("# --- installed on the server"):
        verify_only = True
    line = line.split("#")[0].strip()
    if not line:
        continue
    m = re.match(r"([A-Za-z0-9_.-]+)==(\S+)$", line)
    if not m:  # unpinned (pytest): only needs to exist
        try:
            md.version(line)
        except md.PackageNotFoundError:
            missing.append(line)
        continue
    name, pin = m.groups()
    try:
        have = md.version(name).split("+")[0]
    except md.PackageNotFoundError:
        if not verify_only:
            missing.append(line)
        continue
    if have != pin:
        bad.append(f"{name}: installed {have}, pinned {pin}")
if bad:
    print("VERSION MISMATCH (not touching installed packages):\n  " + "\n  ".join(bad), file=sys.stderr)
    sys.exit(1)
print("\n".join(missing))
