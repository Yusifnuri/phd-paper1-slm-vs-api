"""BillSum summary generation and ROUGE-L evaluation for the fine-tuned model (validation only).

Optional in train.run (`--gen-eval`, default off) because it costs GPU time: greedy decoding,
batched with left padding, at most billsum.MAX_NEW_TOKENS (1,024) new tokens, stopping at the
chat end token (same stop ids as src/train/evaluate.py). One schema-valid record per item with
arm "finetuned", task "billsum", gold = the gold summary and parsed = the summary text with a
leading "Summary:" removed (same parser as the API and local-ICL arms, prompts.parse_free_text;
an empty output is unparseable and scores ROUGE-L 0, never dropped).

Per-item ROUGE-L F1 comes from src.eval.metrics.rouge_l (Google rouge-score, stemming). The
pure helpers here (length fields, aggregates) are shared with the local ICL runner so every arm
reports the same length covariates the statistical plan asks for (section 6): generated tokens,
gold tokens and their ratio, words, and whether the 1,024-token cap cut the output.

Model-free helpers are importable without torch; only the default generation function needs it.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.api.prompts import parse_free_text

from . import billsum, evaluate

ARM = "finetuned"


def word_count(text: Optional[str]) -> int:
    return len(text.split()) if text else 0


def rouge_l_scores(preds: Sequence[Optional[str]], golds: Sequence[str]) -> List[float]:
    """Per-item ROUGE-L F1 (rougeL, stemming) of predictions against gold summaries; None scores 0."""
    from src.eval.metrics import rouge_l  # lazy: needs the rouge-score package

    return [float(x) for x in rouge_l(list(preds), list(golds)).per_item]


def length_fields(pred_text: str, gold: str, output_tokens: Optional[int], gold_tokens: Optional[int],
                  hit_cap: Optional[bool]) -> Dict[str, Any]:
    """Length covariates of one summary (tokens are the model's own tokeniser; ratios are
    prediction over gold, None when the denominator is 0 or unknown)."""
    pw, gw = word_count(pred_text), word_count(gold)
    return {
        "gen_tokens": output_tokens, "gold_tokens": gold_tokens,
        "token_ratio": (output_tokens / gold_tokens) if output_tokens is not None and gold_tokens else None,
        "pred_words": pw, "gold_words": gw, "word_ratio": (pw / gw) if gw else None,
        "hit_cap": hit_cap,
    }


def make_record(*, run_id: str, system: str, technique: str, n: int, seed: int, train_index: int,
                prediction: str, gold: str, input_tokens: int, output_tokens: int,
                gold_tokens: Optional[int], hit_cap: bool, rouge_l: float) -> Dict[str, Any]:
    """One prediction record in the field order of src/eval/schema.py (validated by a test)."""
    parsed, ok = parse_free_text(prediction, {})
    return {
        "run_id": run_id, "system": system, "arm": ARM, "technique": technique,
        "task": billsum.TASK, "n_budget": n, "seed": seed, "item_id": billsum.item_id("train", train_index),
        "prediction": prediction, "parsed": parsed, "gold": gold, "parse_ok": ok,
        "confidence": None, "latency_ms": None,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "extra": {"split": "validation", "rouge_l": rouge_l, "finish_reason": "length" if hit_cap else "stop",
                  **length_fields(parsed or "", gold, output_tokens, gold_tokens, hit_cap)},
    }


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Run-level summary of records made by make_record (or the ICL runner's equivalent)."""
    if not records:
        raise ValueError("no records to aggregate")
    ex = [r["extra"] for r in records]

    def col(name: str) -> List[float]:
        return [e[name] for e in ex if e.get(name) is not None]

    toks, tr, wr = col("gen_tokens"), col("token_ratio"), col("word_ratio")
    return {
        "rouge_l": statistics.fmean(e["rouge_l"] for e in ex),
        "n_unparseable": sum(not r["parse_ok"] for r in records),
        "n_hit_cap": sum(bool(e.get("hit_cap")) for e in ex),
        "gen_tokens_mean": statistics.fmean(toks) if toks else None,
        "gen_tokens_median": statistics.median(toks) if toks else None,
        "token_ratio_mean": statistics.fmean(tr) if tr else None,
        "token_ratio_median": statistics.median(tr) if tr else None,
        "word_ratio_mean": statistics.fmean(wr) if wr else None,
        "pred_words_mean": statistics.fmean(col("pred_words")),
        "gold_words_mean": statistics.fmean(col("gold_words")),
    }


def summary_stop_ids(tokenizer: Any, labels_all: Sequence[str]) -> List[int]:
    """EOS plus the end-of-turn token the BillSum chat template appends to a completion."""
    _, comp = billsum.render_pair(tokenizer, "x", labels_all[0] if labels_all else "y")
    return evaluate.stop_token_ids(tokenizer, tokenizer(comp, add_special_tokens=False)["input_ids"])


def _default_generate(model: Any, tokenizer: Any, stop_ids: Sequence[int], pad_id: int,
                      batch_size: int, max_new_tokens: int) -> Callable[[Sequence[str]], List[Dict[str, Any]]]:
    def gen(queries: Sequence[str]) -> List[Dict[str, Any]]:
        return evaluate.generate_predictions(
            model, tokenizer, queries, stop_ids, pad_id, batch_size, max_new_tokens,
            prompt_fn=billsum.prompt_ids, use_cache=True)
    return gen


def evaluate_summaries(
    model: Any, tokenizer: Any, *, run_id: str, system: str, technique: str, n: int, seed: int,
    items: Sequence[tuple], stop_ids: Sequence[int], pad_id: int, batch_size: int,
    max_new_tokens: int = billsum.MAX_NEW_TOKENS,
    generate_fn: Optional[Callable[[Sequence[str]], List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """items: (train_index, bill text, gold summary). Returns records, aggregate stats and timing.

    generate_fn(queries) -> [{"text", "input_tokens", "output_tokens", "stopped"}] is injectable for
    offline tests; the default is evaluate.generate_predictions with the BillSum prompt.
    """
    t0 = time.perf_counter()
    fn = generate_fn or _default_generate(model, tokenizer, stop_ids, pad_id, batch_size, max_new_tokens)
    gens = fn([q for _, q, _ in items])
    if len(gens) != len(items):
        raise RuntimeError(f"generator returned {len(gens)} outputs for {len(items)} items")
    texts = [parse_free_text(g["text"], {})[0] for g in gens]
    rouge = rouge_l_scores(texts, [gold for _, _, gold in items])
    records = [
        make_record(
            run_id=run_id, system=system, technique=technique, n=n, seed=seed, train_index=ti,
            prediction=g["text"], gold=gold, input_tokens=g["input_tokens"], output_tokens=g["output_tokens"],
            gold_tokens=len(tokenizer(gold, add_special_tokens=False)["input_ids"]),
            hit_cap=not g.get("stopped", True), rouge_l=r)
        for (ti, _, gold), g, r in zip(items, gens, rouge)
    ]
    return {"records": records, **aggregate(records), "n_eval": len(records),
            "eval_seconds": time.perf_counter() - t0}
