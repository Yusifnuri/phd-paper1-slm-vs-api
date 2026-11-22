"""Validation-set generation, parsing, accuracy, and prediction records.

Parsing is the shared task_spec.parse_label (case-insensitive, same as the API arm).
Records follow src/eval/schema.py (branch claude/eval-harness-3f9302, commit ae0aabd)
field for field. That module is not merged here, so records are built as plain dicts and
tests validate them against it when it is importable. Differences: none known.
Unparseable outputs get parsed=None, parse_ok=False and count as wrong.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from . import task_spec
from .data import item_id, left_pad, prompt_ids
from .task_spec import TASK

SCHEMA_FIELDS = (
    "run_id", "system", "arm", "technique", "task", "n_budget", "seed", "item_id",
    "prediction", "parsed", "gold", "parse_ok", "confidence", "latency_ms",
    "input_tokens", "output_tokens", "extra",
)


def parse_label(text: str, labels=None) -> Optional[str]:
    """Shared parser (task_spec.parse_label): canonical label or None. `labels` is ignored
    unless given as a custom lookup dict; kept for call compatibility."""
    return task_spec.parse_label(text, labels if isinstance(labels, dict) else None)


def make_record(
    *, run_id: str, system: str, technique: str, n: int, seed: int, train_index: int,
    prediction: str, gold: str, input_tokens: int, output_tokens: int,
) -> Dict[str, Any]:
    parsed = task_spec.parse_label(prediction)
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
    prompt_fn: Optional[Callable[[Any, str], List[int]]] = None, use_cache: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Greedy batched generation (left padding). Returns per item: text, input_tokens, output_tokens
    and `stopped` (a stop token ended the output; False means the max_new_tokens cap cut it).

    prompt_fn(tokenizer, query) -> prompt ids defaults to the BANKING77 prompt; the summarisation
    evaluation passes the BillSum one. use_cache=None leaves the model's setting untouched
    (BANKING77); the BillSum training model has use_cache off for gradient checkpointing, so
    src/train/summarise.py passes True.
    """
    import torch

    make_prompt = prompt_fn or prompt_ids
    prompts = [make_prompt(tokenizer, q) for q in queries]
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
            extra_kw = {} if use_cache is None else {"use_cache": use_cache}
            gen = model.generate(
                input_ids=ids_t, attention_mask=torch.tensor(mask, device=device),
                max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=list(stop_ids), pad_token_id=pad_id, **extra_kw,
            )[:, ids_t.shape[1]:].tolist()
            for i, gen_i in zip(idx, gen):
                cut = next((k for k, t in enumerate(gen_i) if t in stops), len(gen_i))
                g = gen_i[:cut]
                out[i] = {
                    "text": tokenizer.decode(g, skip_special_tokens=True),
                    "input_tokens": len(prompts[i]), "output_tokens": len(g),
                    "stopped": cut < len(gen_i),
                }
    return out  # type: ignore[return-value]


def evaluate(
    model: Any, tokenizer: Any, *, run_id: str, system: str, technique: str, n: int,
    seed: int, items: Sequence[tuple], stop_ids: Sequence[int],
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
            prediction=g["text"], gold=gold,
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
