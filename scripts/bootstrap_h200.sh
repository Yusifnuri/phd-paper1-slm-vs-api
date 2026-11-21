#!/usr/bin/env bash
# Idempotent setup for a fresh H200 server (Ubuntu 24.04, user with passwordless sudo).
# Needs only the packages in requirements/train.txt and your Hugging Face credentials.
# Safe to rerun after every server restart (the disk is wiped, so you will need to).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$HOME/venv-paper1}"

if ! python3 -c 'import venv, ensurepip' 2>/dev/null; then
  echo ">> python3-venv missing; installing via apt"
  sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating venv at $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> installing requirements/train.txt (no-op if already satisfied)"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "$REPO/requirements/train.txt"

echo; echo ">> GPU"
nvidia-smi || echo "WARNING: nvidia-smi failed"

echo; echo ">> library versions"
python - <<'PY'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for n in ("torch", "transformers", "peft", "bitsandbytes", "datasets", "accelerate", "huggingface_hub"):
    try: print(f"{n:16s}{md.version(n)}")
    except md.PackageNotFoundError: print(f"{n:16s}NOT INSTALLED")
try:
    import torch
    print("torch CUDA build", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available(): print("device", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", e)
PY

echo; echo ">> Hugging Face login check (token from HF_TOKEN or the login cache; never printed)"
if python - <<'PY'
from huggingface_hub import HfApi
try:
    print("logged in as:", HfApi().whoami()["name"])
except Exception:
    raise SystemExit(1)
PY
then :; else
  echo "NOT logged in. Run:   source $VENV/bin/activate && huggingface-cli login"
  echo "(newer huggingface_hub: 'hf auth login'; or export HF_TOKEN=...)"
fi

echo; echo ">> done. Next:"
echo "   source $VENV/bin/activate && cd $REPO"
echo "   python scripts/run_pilot.py --smoke"
