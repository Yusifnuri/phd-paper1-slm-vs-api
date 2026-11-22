"""Few-shot rendering and shot selection for the local ICL arm.

The rendered text comes from `src.api.prompts.build_prompt`, the same function the API arm
calls, with the same template and context. There is no second implementation of the prompt:
`tests/icl/test_icl_render.py` proves byte equality with `src.api.run.plan_items` (API arm) and
with `src.train.task_spec` (fine-tuned arm, zero-shot).

Shot order follows the API arm (src/api/run.py plan_items, protocol section 1): the retriever
returns most similar first and the list is reversed, so the nearest neighbour sits next to the
query. icl_random keeps the order of `src.api.retrieval.random_k`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.api import prompts
from src.api.prompts import Example, Prompt, build_prompt
from src.api.retrieval import Embedder, embed_cached, random_k
from src.train import task_spec

TECHNIQUES = ("zero_shot", "icl_retrieved", "icl_random")


@dataclass(frozen=True)
class IclTask:
    name: str
    template: str                 # key in src.api.prompts registry
    context: Dict[str, Any]       # passed unchanged to build_prompt
    ks: Tuple[int, ...]           # k grid for tuning (protocol section 1 and 4)
    max_new_tokens: int
    metric: str                   # "accuracy" | "rouge_l"
    parse: Callable[[str], Tuple[Any, bool]]  # text -> (parsed, parse_ok); parse_ok == parsed is not None


def _parse_banking77(text: str) -> Tuple[Any, bool]:
    parsed = task_spec.parse_label(text)  # same parser as the fine-tuned arm
    return parsed, parsed is not None


def _parse_billsum(text: str) -> Tuple[Any, bool]:
    return prompts.parse_free_text(text, {})


TASKS: Dict[str, IclTask] = {
    "banking77": IclTask(
        name="banking77", template="classification",
        context={"task_description": prompts.BANKING77_DESCRIPTION, "labels": list(task_spec.BANKING77_LABELS)},
        ks=(4, 8, 16, 32, 64), max_new_tokens=task_spec.MAX_NEW_TOKENS, metric="accuracy",
        parse=_parse_banking77),
    # BillSum: k <= 3 (protocol section 1). Output cap 1,024 for every arm (protocol section 5).
    # The CLI selects k by validation ROUGE-L (src/icl/cli.py).
    "billsum": IclTask(
        name="billsum", template="summarisation",
        context={"task_description": prompts.BILLSUM_DESCRIPTION},
        ks=(1, 2, 3), max_new_tokens=1024, metric="rouge_l", parse=_parse_billsum),
}


def get_task(name: str) -> IclTask:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}") from None


def render(task: IclTask | str, shots: Sequence[Example], query: str) -> Prompt:
    """The prompt (system text, user text) for one query. Identical to the API arm's."""
    t = get_task(task) if isinstance(task, str) else task
    return build_prompt(t.template, list(shots), query, t.context)


def chat_messages(prompt: Prompt) -> List[Dict[str, str]]:
    """Chat turns sent to the model: system, then user (same two turns as task_spec.build_messages)."""
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]


class ShotSelector:
    """Shots for a fixed list of items, for any k, with one embedding pass.

    zero_shot: no shots. icl_random: `random_k(pool, k, seed, item.id)` (per-item seeding, so
    independent of processing order). icl_retrieved: the k nearest pool items by cosine of the
    pinned bge embeddings (stable argsort, ties by pool order, as src.api.retrieval.Retriever),
    nearest last. `max_cosine` (per item, to the whole pool) is kept for the record.
    """

    def __init__(self, technique: str, pool: Sequence[Example], items: Sequence[Example], *,
                 embedder: Optional[Embedder] = None, cache_dir: Optional[str] = None,
                 seed: Optional[int] = None, sims: Optional[np.ndarray] = None):
        if technique not in TECHNIQUES:
            raise ValueError(f"technique must be one of {TECHNIQUES}")
        if technique != "zero_shot" and not pool:
            raise ValueError("empty pool")
        if technique == "icl_random" and seed is None:
            raise ValueError("icl_random needs a seed")
        if technique == "icl_retrieved" and embedder is None and sims is None:
            raise ValueError("icl_retrieved needs an embedder (or a precomputed similarity matrix)")
        ids = [e.id for e in pool]
        if len(set(ids)) != len(ids):
            raise ValueError("pool example ids must be unique")
        self.technique, self.pool, self.items, self.seed = technique, list(pool), list(items), seed
        self.sims: Optional[np.ndarray] = None
        self.max_cosine: List[Optional[float]] = [None] * len(self.items)
        if technique == "icl_retrieved" and sims is not None:
            # BillSum bm25 / tfidf (src/icl/retrievers.py): scores of every pool item for every item;
            # `max_cosine` then holds the largest retriever score (not a cosine) and is only descriptive.
            if sims.shape != (len(self.items), len(self.pool)):
                raise ValueError(f"sims shape {sims.shape} != ({len(self.items)}, {len(self.pool)})")
            self.sims = np.asarray(sims)
            self.max_cosine = [float(r.max()) for r in self.sims]
        elif technique == "icl_retrieved":
            def unit(m: np.ndarray) -> np.ndarray:
                nrm = np.linalg.norm(m, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                return m / nrm
            pv = unit(embed_cached(embedder, [e.input for e in self.pool], cache_dir))
            qv = unit(embed_cached(embedder, [e.input for e in self.items], cache_dir))
            self.sims = qv @ pv.T
            self.max_cosine = [float(r.max()) for r in self.sims]

    def shots(self, k: int) -> List[List[Example]]:
        """Shots per item, in prompt order."""
        if self.technique == "zero_shot" or k <= 0:
            return [[] for _ in self.items]
        if self.technique == "icl_random":
            return [random_k(self.pool, k, int(self.seed), it.id) for it in self.items]
        kk = min(k, len(self.pool))
        out = []
        for row in self.sims:  # type: ignore[union-attr]
            order = np.argsort(-row, kind="stable")[:kk]
            out.append([self.pool[i] for i in reversed(order)])  # nearest last
        return out
