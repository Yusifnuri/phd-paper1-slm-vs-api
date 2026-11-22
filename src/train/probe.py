"""Pure helpers of the BillSum memory/throughput probe (protocol section 8, stage 4). No torch."""

from __future__ import annotations

from typing import Dict, List, Sequence

from . import tuning


def longest_fraction(lengths: Dict[int, int], frac: float = 0.01) -> List[int]:
    """Indices of the longest `frac` of items (at least one), longest first; ties by index."""
    if not lengths or not 0 < frac <= 1:
        raise ValueError("need items and 0 < frac <= 1")
    k = max(1, int(round(len(lengths) * frac)))
    return sorted(lengths, key=lambda i: (-lengths[i], i))[:k]


def projected_seconds(n: int, mean_tokens: float, tokens_per_second: float, batch: int = 16,
                      s_min: int = 200) -> Dict[str, float]:
    """Upper-bound training time of ONE trial/final at N: the whole cosine schedule
    (max(E_max * ceil(N/B), S_min) steps, no early stopping) at the measured throughput.
    Throughput measured on the longest bills is conservative for the mean bill."""
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    steps = tuning.schedule_steps(n, batch, s_min)
    tokens = steps * batch * mean_tokens
    return {"n": n, "schedule_steps": steps, "tokens": tokens, "seconds": tokens / tokens_per_second}


def go_no_go(final_seconds_n3000: float) -> str:
    """Protocol section 8 / FEASIBILITY go/no-go on the Qwen BillSum N=3,000 final run (hours)."""
    h = final_seconds_n3000 / 3600
    if h <= 1.2:
        return "keep the design with cut 1 (BillSum tuning at N=3,000 reuses N=1,000)"
    if h <= 2.0:
        return "cuts 1 and 4 (BillSum N=3,000 at R=3, exploratory)"
    return "cuts 1, 5 and 6 (BillSum Qwen-only, N=3,000 dropped)"
