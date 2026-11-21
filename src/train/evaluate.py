"""Validation-set generation, parsing, accuracy, and prediction records.

Records follow src/eval/schema.py (branch claude/eval-harness-3f9302, commit ae0aabd)
field for field. That module is not merged here, so records are built as plain dicts and
tests validate them against it when it is importable. Differences: none known.
Unparseable outputs get parsed=None, parse_ok=False and count as wrong.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from .config import TASK
from .data import item_id, left_pad, prompt_ids

SCHEMA_FIELDS = (
    "run_id", "system", "arm", "technique", "task", "n_budget", "seed", "item_id",
    "prediction", "parsed", "gold", "parse_ok", "confidence", "latency_ms",
    "input_tokens", "output_tokens", "extra",
)


def normalise(text: str) -> str:
    """First non-empty line, lower-cased, spaces and hyphens to underscores, quotes stripped."""
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            line = line.strip("`'\".,;: ")
            return re.sub(r"[\s\-]+", "_", line.lower())
    return ""


def parse_label(text: str, labels: Set[str]) -> Optional[str]:
    cand = normalise(text)
    return cand if cand in labels else None


def make_record(
    *, run_id: str, system: str, technique: str, n: int, seed: int, train_index: int,
    prediction: str, gold: str, labels: Set[str], input_tokens: int, output_tokens: int,
) -> Dict[str, Any]:
    parsed = parse_label(prediction, labels)
    return {
        "run_id": run_id, "system": system, "arm": "finetuned", "technique": technique,
        "task": TASK, "n_budget": n, "seed": seed, "item_id": item_id(train_index),
        "prediction": prediction, "parsed": parsed, "gold": gold, "parse_ok": parsed is not None,
        "confidence": None, "latency_ms": None,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "extra": {"split": "validation"},
    }


def accuracy(records: Sequence[Dict[str, Any]]) -> float:
    """Exact match of parsed to gold; unparseable counts as wrong."""
    return sum(r["parsed"] == r["gold"] for r in records) / len(records) if records else 0.0


def stop_token_ids(tokenizer: Any, completion_ids: Sequence[int]) -> List[int]:
    """EOS plus the end-of-turn token(s) the chat template appends to a completion.

    Phi-4-mini's eos is <|endoftext|> but its template closes turns with <|end|>; stopping
    only on eos would let the model run on past the label.
    """
    special = set(tokenizer.all_special_ids)
    ids = {i for i in completion_ids if i in special}
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def generate_predictions(
    model: Any, tokenizer: Any, queries: Sequence[str], stop_ids: Sequence[int],
    pad_id: int, batch_size: int, max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """Greedy batched generation. Returns per item: text, input_tokens, output_tokens."""
    import torch

    prompts = [prompt_ids(tokenizer, q) for q in queries]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # similar lengths together
    out: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    stops = set(stop_ids)
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            ids, mask = left_pad([prompts[i] for i in idx], pad_id)
            ids_t = torch.tensor(ids, device=device)
            gen = model.generate(
                input_ids=ids_t, attention_mask=torch.tensor(mask, device=device),
                max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=list(stop_ids), pad_token_id=pad_id,
            )[:, ids_t.shape[1]:].tolist()
            for i, g in zip(idx, gen):
                cut = next((k for k, t in enumerate(g) if t in stops), len(g))
                g = g[:cut]
                out[i] = {
                    "text": tokenizer.decode(g, skip_special_tokens=True),
                    "input_tokens": len(prompts[i]), "output_tokens": len(g),
                }
    return out  # type: ignore[return-value]


def evaluate(
    model: Any, tokenizer: Any, *, run_id: str, system: str, technique: str, n: int,
    seed: int, items: Sequence[tuple], labels: Set[str], stop_ids: Sequence[int],
    pad_id: int, batch_size: int, max_new_tokens: int,
) -> Dict[str, Any]:
    """items: (train_index, query, gold). Returns records, accuracy and timing."""
    t0 = time.perf_counter()
    gens = generate_predictions(
        model, tokenizer, [q for _, q, _ in items], stop_ids, pad_id, batch_size, max_new_tokens
    )
    records = [
        make_record(
            run_id=run_id, system=system, technique=technique, n=n, seed=seed, train_index=ti,
            prediction=g["text"], gold=gold, labels=labels,
            input_tokens=g["input_tokens"], output_tokens=g["output_tokens"],
        )
        for (ti, _, gold), g in zip(items, gens)
    ]
    return {
        "records": records,
        "accuracy": accuracy(records),
        "n_eval": len(records),
        "n_unparseable": sum(not r["parse_ok"] for r in records),
        "eval_seconds": time.perf_counter() - t0,
    }
