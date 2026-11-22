import subprocess
import sys

from src.api.prompts import Prompt
from src.icl.generate import Generation, VLLMGenerator


def test_importing_the_package_does_not_import_vllm():
    code = ("import sys; import src.icl.generate, src.icl.cli, src.icl.render; "
            "assert 'vllm' not in sys.modules and 'torch' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(__import__('pathlib').Path(__file__).resolve().parents[2]))
    assert r.returncode == 0, r.stderr


def test_vllm_generator_is_lazy_and_needs_vllm_only_on_generate():
    g = VLLMGenerator("Qwen/Qwen3-4B-Instruct-2507")  # no engine, no import
    assert g.name == "Qwen/Qwen3-4B-Instruct-2507" and g._llm is None
    try:
        import vllm  # noqa: F401
    except ImportError:
        try:
            g.generate([Prompt("s", "u")], 32)
        except ImportError:
            return
        raise AssertionError("expected ImportError without vllm")
