"""Run one (provider, task, technique, n_budget) configuration over a test set.

Inputs are JSONL files with one object per line: {"id": str, "input": str, "output": str}.
- The pool file is the N-example training pool. Its "output" is the gold answer shown in the shots.
- The test file holds the evaluation items. Their "output" is the gold answer, copied into `gold`.

Output:
- `--out` (JSONL): one record per item, in the schema of src/api/records.py.
- `<out>.raw.jsonl`: the raw provider responses, keyed by item_id.
- `<out>.estimate.json`: the cost estimate that cleared the budget gate.
- `<out>.batch.json`: batch state, kept while a batch is in flight.

Safety:
- **Resumable.** Item ids already in `--out` are skipped, and so are items in a batch that was submitted but not yet collected. That batch is polled and collected instead of being resubmitted, so nothing is paid twice.
- **Budget gate.** The cost of the remaining items is estimated before any call: the ceiling, with every output at max_output_tokens. If it exceeds `--budget-usd`, the run raises `BudgetExceeded` and makes no call.
- **Explicit approval.** A paid run needs `--approve-spend`. Without it, the CLI prints the estimate and stops, like `--dry-run`.

Example (dry run, no network):
    python -m src.api.run --provider anthropic --task banking77 --template classification \\
        --technique icl_retrieved --n-budget 200 --k 32 --pool pool.jsonl --test test.jsonl \\
        --context-json banking77_context.json --out results/api/x.jsonl --budget-usd 5 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.api.clients import BATCH_FAILED, BatchFailed, ChatClient, Completion, make_client
from src.api.config import DEFAULT_CONFIG, ModelSpec, load_specs
from src.api.cost import Estimate, estimate_prompts
from src.api.prompts import Example, Prompt, build_prompt, get_template, parse_output
from src.api.records import append_records, done_item_ids
from src.api.retrieval import TECHNIQUES, Embedder, Retriever, random_k


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class PlannedItem:
    item: Example
    prompt: Prompt
    shot_ids: list[str]  # ids of the examples in the prompt, in prompt order


def load_examples(path: str | Path) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                d = json.loads(ln)
                out.append(Example(id=str(d["id"]), input=str(d["input"]), output=str(d["output"])))
    return out


def make_run_id(system: str, task: str, technique: str, n_budget: int, k: int, seed: int | None, mode: str) -> str:
    seed_part = f"s{seed}" if seed is not None else "s-"
    k_part = 0 if technique == "zero_shot" else k
    return f"api__{system}__{task}__{technique}__n{n_budget}__k{k_part}__{seed_part}__{mode}"


def side_path(out: Path, suffix: str) -> Path:
    return out.with_name(out.name + suffix)


def plan_items(
    *, template: str, context: dict[str, Any], technique: str, k: int, seed: int | None,
    pool: Sequence[Example], items: Sequence[Example], embedder: Embedder | None,
    cache_dir: str | Path | None, verbalised_confidence: bool,
) -> list[PlannedItem]:
    if technique not in TECHNIQUES:
        raise ValueError(f"technique must be one of {TECHNIQUES}")
    if technique == "icl_retrieved":
        if embedder is None:
            raise ValueError("icl_retrieved needs an embedder")
        retriever = Retriever(pool, embedder, cache_dir)
        # Most similar first from the retriever; reversed so the nearest neighbour sits next to the query.
        shots_per_item = [list(reversed(s)) for s in retriever.top_k_many([it.input for it in items], k)]
    elif technique == "icl_random":
        if seed is None:
            raise ValueError("icl_random needs a seed")
        shots_per_item = [random_k(pool, k, seed, it.id) for it in items]
    else:
        shots_per_item = [[] for _ in items]
    return [
        PlannedItem(it, build_prompt(template, shots, it.input, context, verbalised_confidence=verbalised_confidence),
                    [s.id for s in shots])
        for it, shots in zip(items, shots_per_item)
    ]


def _record(*, run_id: str, spec: ModelSpec, task: str, technique: str, n_budget: int, seed: int | None,
            planned: PlannedItem, completion: Completion, template: str, context: dict[str, Any],
            verbalised_confidence: bool, mode: str, k: int) -> dict[str, Any]:
    parsed, ok, conf = parse_output(template, completion.text, context, verbalised_confidence=verbalised_confidence)
    return {
        "run_id": run_id,
        "system": spec.model_id,
        "arm": "api",
        "technique": technique,
        "task": task,
        "n_budget": n_budget,
        "seed": seed if technique == "icl_random" else None,
        "item_id": planned.item.id,
        "prediction": completion.text,
        "parsed": parsed,
        "gold": planned.item.output,
        "parse_ok": ok,
        "confidence": conf,
        "latency_ms": completion.latency_ms,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "extra": {
            "provider": spec.provider,
            "model_id": spec.model_id,
            "model_id_returned": completion.model_id,
            "run_date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "k": k if technique != "zero_shot" else 0,
            "retrieved_ids": planned.shot_ids,
            "template": template,
            "verbalised_confidence": verbalised_confidence,
            "price_read_date": spec.price_read_date,
            **{f"provider_{k_}": v for k_, v in completion.extra.items()},
        },
    }


def _write_raw(out: Path, item_id: str, raw: dict[str, Any]) -> None:
    with open(side_path(out, ".raw.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"item_id": item_id, "raw": raw}, ensure_ascii=False, default=str) + "\n")


def run_config(
    *, client: ChatClient, task: str, template: str, context: dict[str, Any], technique: str,
    n_budget: int, k: int, seed: int | None, pool: Sequence[Example], test_items: Sequence[Example],
    out_path: str | Path, budget_usd: float, mode: str = "sync", embedder: Embedder | None = None,
    cache_dir: str | Path | None = None, verbalised_confidence: bool = False, dry_run: bool = False,
    poll_interval_s: float = 60.0, sleep=None,
) -> dict[str, Any]:
    """Run (or estimate) one configuration. Returns a summary dict."""
    if mode not in ("sync", "batch"):
        raise ValueError("mode must be 'sync' or 'batch'")
    if technique == "icl_random" and seed is None:
        raise ValueError("icl_random needs a seed")
    if technique != "icl_random":
        seed = None
    if len(pool) < n_budget and technique != "zero_shot":
        raise ValueError(f"pool has {len(pool)} examples but n_budget is {n_budget}")
    spec = client.spec
    tpl = get_template(template)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(spec.model_id, task, technique, n_budget, k, seed, mode)
    summary: dict[str, Any] = {"run_id": run_id, "written": 0, "skipped_done": 0, "errors": 0}

    # 1. Finish any batch already in flight before planning new work.
    batch_state_path = side_path(out, ".batch.json")
    in_flight: set[str] = set()
    if batch_state_path.exists():
        state = json.loads(batch_state_path.read_text(encoding="utf-8"))
        in_flight = set(state["custom_ids"].values())
        if not dry_run:
            summary["resumed_batch"] = state["batch_id"]
            n, errs = _finish_batch(client, state, out, test_items, pool, task, template, context, technique, n_budget,
                                    k, seed, embedder, cache_dir, verbalised_confidence, run_id,
                                    poll_interval_s, sleep)
            summary["written"] += n
            summary["errors"] += errs
            batch_state_path.unlink()
            in_flight = set()

    # 2. Skip finished items.
    done = done_item_ids(out)
    todo = [it for it in test_items if it.id not in done and it.id not in in_flight]
    summary["skipped_done"] = len(test_items) - len(todo)
    planned = plan_items(template=template, context=context, technique=technique, k=k, seed=seed, pool=pool,
                         items=todo, embedder=embedder, cache_dir=cache_dir,
                         verbalised_confidence=verbalised_confidence)

    # 3. Budget gate: estimate before any call.
    expected_chars = (sum(len(it.output) for it in test_items) / len(test_items)) if test_items else None
    est: Estimate = estimate_prompts([p.prompt for p in planned], spec, max_output_tokens=tpl.max_output_tokens,
                                     expected_output_chars=expected_chars, task=task, technique=technique,
                                     n_budget=n_budget, k=k)
    cap = est.cap_usd(batch=(mode == "batch"))
    est_record = {**est.to_dict(), "mode": mode, "budget_usd": budget_usd, "gate_usd": cap,
                  "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "run_id": run_id}
    side_path(out, ".estimate.json").write_text(json.dumps(est_record, indent=2), encoding="utf-8")
    summary["estimate"] = est_record
    if cap > budget_usd:
        raise BudgetExceeded(f"estimated ceiling ${cap:.6f} for {len(planned)} items exceeds --budget-usd ${budget_usd:.6f}")
    if dry_run or not planned:
        summary["dry_run"] = dry_run
        return summary

    # 4. Execute.
    common = dict(run_id=run_id, spec=spec, task=task, technique=technique, n_budget=n_budget, seed=seed,
                  template=template, context=context, verbalised_confidence=verbalised_confidence, k=k)
    if mode == "sync":
        for p in planned:
            try:
                comp = client.complete(p.prompt, tpl.max_output_tokens)
            except Exception as exc:  # noqa: BLE001 - logged; the item stays undone and is retried on resume
                summary["errors"] += 1
                print(f"[error] {p.item.id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            _write_raw(out, p.item.id, comp.raw)
            summary["written"] += append_records(out, [_record(planned=p, completion=comp, mode=mode, **common)])
    else:
        custom_ids = {f"r{i}": p.item.id for i, p in enumerate(planned)}
        batch_id = client.batch_submit([(cid, p.prompt) for cid, p in zip(custom_ids, planned)], tpl.max_output_tokens)
        state = {"batch_id": batch_id, "provider": spec.provider, "model_id": spec.model_id, "task": task,
                 "custom_ids": custom_ids,
                 "submitted": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        batch_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        summary["batch_id"] = batch_id
        by_id = {p.item.id: p for p in planned}
        n, errs = _collect_into(client, state, out, by_id, common, poll_interval_s, sleep)
        summary["written"] += n
        summary["errors"] += errs
        batch_state_path.unlink()
    return summary


def _collect_into(client, state, out, by_id, common, poll_interval_s, sleep) -> tuple[int, int]:
    kwargs = {"interval_s": poll_interval_s}
    if sleep is not None:
        kwargs["sleep"] = sleep
    status = client.batch_poll(state["batch_id"], **kwargs)
    if status == BATCH_FAILED:
        raise BatchFailed(f"batch {state['batch_id']} failed; state kept in .batch.json")
    written = errors = 0
    for res in client.batch_collect(state["batch_id"]):
        item_id = state["custom_ids"].get(res.custom_id)
        if item_id is None or item_id not in by_id:
            continue
        if res.completion is None:
            errors += 1
            print(f"[batch error] {item_id}: {res.error}", file=sys.stderr)
            continue
        _write_raw(out, item_id, res.completion.raw)
        written += append_records(out, [_record(planned=by_id[item_id], completion=res.completion,
                                                mode="batch", **common)])
    return written, errors


def _finish_batch(client, state, out, test_items, pool, task, template, context, technique, n_budget, k, seed,
                  embedder, cache_dir, verbalised_confidence, run_id, poll_interval_s, sleep) -> tuple[int, int]:
    """Re-plan the in-flight items (deterministic) so their records carry the same shot ids, then collect."""
    ids = set(state["custom_ids"].values())
    done = done_item_ids(out)
    items = [it for it in test_items if it.id in ids and it.id not in done]
    planned = plan_items(template=template, context=context, technique=technique, k=k, seed=seed, pool=pool,
                         items=items, embedder=embedder, cache_dir=cache_dir,
                         verbalised_confidence=verbalised_confidence)
    common = dict(run_id=run_id, spec=client.spec, task=task, technique=technique,
                  n_budget=n_budget, seed=seed, template=template, context=context,
                  verbalised_confidence=verbalised_confidence, k=k)
    return _collect_into(client, state, out, {p.item.id: p for p in planned}, common, poll_interval_s, sleep)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, help="key in configs/api_models.yaml, e.g. anthropic")
    ap.add_argument("--task", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--technique", required=True, choices=TECHNIQUES)
    ap.add_argument("--n-budget", type=int, required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=None, help="required for icl_random")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--context-json", default=None, help="JSON object with template context, e.g. labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--mode", choices=("sync", "batch"), default="batch")
    ap.add_argument("--confidence", action="store_true", help="ask for verbalised confidence")
    ap.add_argument("--cache-dir", default="data/cache/embeddings")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--approve-spend", action="store_true", help="required for any paid call")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N test items (pilot)")
    args = ap.parse_args(argv)

    dry_run = args.dry_run or not args.approve_spend
    if not dry_run:
        from dotenv import load_dotenv

        load_dotenv()
    spec = load_specs(args.config)[args.provider]
    client = make_client(spec)
    pool = load_examples(args.pool)[: args.n_budget]
    test_items = load_examples(args.test)
    if args.limit:
        test_items = test_items[: args.limit]
    context = json.loads(Path(args.context_json).read_text(encoding="utf-8")) if args.context_json else {}
    embedder = None
    if args.technique == "icl_retrieved":
        from src.api.retrieval import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
    try:
        summary = run_config(client=client, task=args.task, template=args.template, context=context,
                             technique=args.technique, n_budget=args.n_budget, k=args.k, seed=args.seed,
                             pool=pool, test_items=test_items, out_path=args.out, budget_usd=args.budget_usd,
                             mode=args.mode, embedder=embedder, cache_dir=args.cache_dir,
                             verbalised_confidence=args.confidence, dry_run=dry_run)
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, default=str))
    if dry_run and not args.dry_run:
        print("No calls made: pass --approve-spend to run for real.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
