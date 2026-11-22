"""Local ICL arm (Qwen3-4B-Instruct-2507, Phi-4-mini-instruct) on vLLM. Thin wrapper: see src/icl/cli.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.icl.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
