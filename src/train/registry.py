"""Model registry. Adding a model is one entry in MODELS.

Target modules are the attention projections, as in the thesis. Phi-3/4 fuse q, k and v
into one `qkv_proj`, so listing q_proj/k_proj/v_proj for Phi-4-mini would silently
match nothing (see check_lora_targets). Module names were checked against the
transformers modeling files for Phi3, Qwen3 and Olmo3 on 2026-11-21.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    target_modules: Tuple[str, ...]
    note: str = ""


_STD = ("q_proj", "k_proj", "v_proj", "o_proj")

MODELS: Dict[str, ModelSpec] = {
    s.key: s
    for s in (
        ModelSpec("phi4-mini", "microsoft/Phi-4-mini-instruct", ("qkv_proj", "o_proj"), "fused qkv"),
        ModelSpec("qwen3-4b", "Qwen/Qwen3-4B-Instruct-2507", _STD),
        ModelSpec("olmo3-7b", "allenai/Olmo-3-7B-Instruct", _STD),
    )
}


def resolve(name: str) -> ModelSpec:
    """Find a spec by registry key or by Hugging Face id."""
    if name in MODELS:
        return MODELS[name]
    for spec in MODELS.values():
        if spec.hf_id == name:
            return spec
    raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)} (add one to MODELS)")
