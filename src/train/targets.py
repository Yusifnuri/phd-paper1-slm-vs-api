"""LoRA target-module check, ported from the thesis repo (src/training/run_utils.py).

PEFT only raises when *none* of the targets match; if some match, the rest are skipped
silently while adapter_config.json still lists them. That is how the thesis Phi-4-mini
adapters ended up on o_proj alone. We fail before training instead.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def unmatched_targets(module_names: Iterable[str], target_modules: Sequence[str]) -> List[str]:
    """Targets matching no module. PEFT rule: name equals the target or ends with '.target'."""
    names = list(module_names)
    return [
        t for t in target_modules
        if not any(n == t or n.endswith("." + t) for n in names)
    ]


def check_lora_targets_from_names(module_names: Iterable[str], target_modules: Sequence[str]) -> None:
    names = list(module_names)
    missing = unmatched_targets(names, target_modules)
    if missing:
        proj = sorted({n.rsplit(".", 1)[-1] for n in names if n and n.endswith("_proj")})
        raise ValueError(
            f"LoRA target_modules {missing} match no layer in this model. "
            f"Projection layers it does have: {proj}. Fix the registry entry."
        )


def check_lora_targets(model, target_modules: Sequence[str]) -> None:
    """Fail before training if any configured target is missing from the model."""
    check_lora_targets_from_names((n for n, _ in model.named_modules()), target_modules)
