"""Local ICL arm: open models served with vLLM, validation only (protocol sections 1, 4, 7).

    python scripts/run_icl.py --model qwen3-4b --technique icl_retrieved --n 200 --sampling stratified \
        --replicate 0 --tune --frozen-arms runs/stage2/frozen_arms.json
    python scripts/run_icl.py --model qwen3-4b --technique icl_retrieved --n 200 --sampling stratified \
        --replicate 1 --frozen-arms runs/stage2/frozen_arms.json
    python scripts/run_icl.py --model qwen3-4b --technique zero_shot --replicate 1 --frozen-arms ...

Mirrors src/baselines/knn.py. `--tune` (replicate 0 only) scores each k of the task's grid
(BANKING77 {4, 8, 16, 32}, 4 values) on val(N) = the first min(1000, N) validation rows, picks
the best (ties: smaller k) and freezes it in frozen_arms.json under
    arm_key(task, <hf model id>, <technique>, N, sampling).
Finals (replicate >= 1) read the frozen arm and score the full 1,000-row validation carve-out
of that replicate (the carve-out is the same for every replicate; the pool is not). Neither
step reads the test split. zero_shot has no k and no pool: it needs no tuning and is only run
as a final (n_budget 0).

The prompt text is the API arm's (src/api/prompts.py via src/icl/render.py); decoding is
greedy free generation parsed by task_spec.parse_label. `--constrained` is a separate,
labelled robustness check: it reuses the arm frozen under free generation (never tuned
separately), writes its own run_id (suffix __constrained) and is flagged in every record.

Record `arm` is "baseline" (see ARM below); extra.arm_detail is "slm_icl".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.api.prompts import Example, Prompt
from src.icl.generate import Generation, Generator
from src.icl.render import TECHNIQUES, IclTask, ShotSelector, get_task, render

# src/eval/schema.py allows ARMS = finetuned | api | baseline and nothing in the repo branches
# on the arm (checked by grep). "finetuned" and "api" would be wrong labels; "baseline" is the
# closest: a base model, no weight update, no paid API. Technique (icl_*) and system (hf id)
# already separate these runs from the kNN baseline. A dedicated "slm_icl" arm would be cleaner
# but needs a one-line change to src/eval/schema.py ARMS (listed in the report); switch ARM then.
ARM = "baseline"
ARM_DETAIL = "slm_icl"


# ---------------------------------------------------------------- pure logic

def accuracy(pred: Sequence[Any], gold: Sequence[Any]) -> float:
    if len(pred) != len(gold) or not gold:
        raise ValueError("pred and gold must be non-empty and of equal length")
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def select_k(per_k: Dict[int, float]) -> Dict[str, Any]:
    """Best k by validation score; ties go to the smaller k (fewer prompt tokens).
    at_grid_edge flags either end of the grid (protocol section 4 edge rule)."""
    ks = sorted(per_k)
    best = max(ks, key=lambda k: (per_k[k], -k))
    edge = None
    if len(ks) > 1 and best == ks[-1]:
        edge = "high"
    elif len(ks) > 1 and best == ks[0]:
        edge = "low"
    return {"k": best, "val_accuracy": per_k[best], "per_k": {str(k): per_k[k] for k in ks},
            "at_grid_edge": edge}


def run_id_for(model: str, technique: str, n: int, seed: int, pool_seed: int, sampling: str,
               constrained: bool) -> str:
    from src.train.config import run_id

    rid = run_id(model, technique, n, seed, pool_seed, sampling)
    return rid + ("__constrained" if constrained else "")


def generate_all(gen: Generator, task: IclTask, prompts: Sequence[Prompt],
                 choices: Optional[Sequence[str]] = None) -> List[Generation]:
    out = gen.generate(list(prompts), task.max_new_tokens, choices)
    if len(out) != len(prompts):
        raise RuntimeError(f"generator returned {len(out)} outputs for {len(prompts)} prompts")
    return out


def make_record(*, run_id: str, system: str, task: IclTask, technique: str, n_budget: int, seed: Optional[int],
                item: Example, gen: Generation, k: int, shot_ids: Sequence[str], max_cosine: Optional[float],
                constrained: bool, extra: Dict[str, Any]) -> Dict[str, Any]:
    """One prediction record in the field order of src/eval/schema.py. Unparseable: parsed None, parse_ok False."""
    parsed, ok = task.parse(gen.text)
    return {
        "run_id": run_id, "system": system, "arm": ARM, "technique": technique, "task": task.name,
        "n_budget": n_budget, "seed": seed, "item_id": item.id,
        "prediction": gen.text, "parsed": parsed, "gold": item.output, "parse_ok": ok,
        "confidence": None, "latency_ms": None,  # batched generation: no per-item latency
        "input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens,
        "extra": {"arm_detail": ARM_DETAIL, "split": "validation", "k": k, "shot_ids": list(shot_ids),
                  "max_cosine_to_pool": max_cosine, "finish_reason": gen.finish_reason,
                  "decoding": "constrained" if constrained else "free", **extra},
    }


def score_k(gen: Generator, task: IclTask, selector: ShotSelector, items: Sequence[Example], k: int,
            *, constrained: bool = False):
    """Render, generate and parse every item at one k. Returns (parsed list, generations, shots, seconds)."""
    shots = selector.shots(k)
    prompts = [render(task, s, it.input) for s, it in zip(shots, items)]
    choices = list(task.context["labels"]) if constrained else None
    t0 = time.perf_counter()
    gens = generate_all(gen, task, prompts, choices)
    seconds = time.perf_counter() - t0
    parsed = [task.parse(g.text)[0] for g in gens]
    return parsed, gens, shots, seconds


# ---------------------------------------------------------------- runner

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="registry key (qwen3-4b, phi4-mini) or Hugging Face id")
    p.add_argument("--task", default="banking77", choices=("banking77", "billsum"))
    p.add_argument("--technique", required=True, choices=TECHNIQUES)
    p.add_argument("--n", type=int, default=0, help="pool size N (required for icl_*; zero_shot ignores it)")
    p.add_argument("--sampling", choices=("random", "stratified"), default="random")
    p.add_argument("--arm-sampling", choices=("random", "stratified"), default=None)
    p.add_argument("--replicate", type=int, required=True)
    p.add_argument("--tune", action="store_true", help="replicate 0: choose k on val(N) and freeze it")
    p.add_argument("--ks", default=None, help="comma-separated k grid (default: the task's grid)")
    p.add_argument("--random-seed", type=int, default=None,
                   help="icl_random per-item seed (default: the replicate's train seed, 100 + r)")
    p.add_argument("--constrained", action="store_true",
                   help="ROBUSTNESS CHECK ONLY: constrained decoding to the label set; reuses the free-generation arm")
    p.add_argument("--frozen-arms", required=True)
    p.add_argument("--results", default="runs/icl.jsonl")
    p.add_argument("--predictions-dir", default="runs/predictions")
    p.add_argument("--cache-dir", default="runs/emb_cache")
    p.add_argument("--device", default=None, help="device of the embedder")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--eval-limit", type=int, default=None, help="smoke mode: first k validation rows")
    return p


def run(argv: Optional[List[str]] = None, embedder=None, loader=None, generator: Optional[Generator] = None):
    """embedder / loader / generator are injectable for offline tests
    (defaults: pinned bge, Hub train split, vLLM)."""
    from src.api.retrieval import SentenceTransformerEmbedder
    from src.train import data, env, registry, task_spec, tuning

    a = build_parser().parse_args(argv)
    task = get_task(a.task)
    zero = a.technique == "zero_shot"
    if a.task != "banking77" and loader is None:
        raise NotImplementedError("BillSum loader and ROUGE-L selection are not wired in yet (src/train/data.py is BANKING77 only)")
    if task.metric != "accuracy":
        raise NotImplementedError(f"selection metric {task.metric!r} is not implemented")
    if a.tune and zero:
        raise ValueError("zero_shot has no k to tune; run it as a final")
    if a.tune and a.constrained:
        raise ValueError("--constrained is a robustness check on a frozen arm; it is never tuned")
    if a.tune and a.replicate != tuning.TUNING_REPLICATE:
        raise ValueError("--tune runs on replicate 0 only")
    if not a.tune and a.replicate == tuning.TUNING_REPLICATE:
        raise ValueError("replicate 0 is the tuning replicate; finals use --replicate >= 1")
    if not zero and a.n < 1:
        raise ValueError("--n is required for icl_retrieved and icl_random")
    ks = tuple(int(x) for x in a.ks.split(",")) if a.ks else task.ks

    hf = registry.resolve(a.model).hf_id
    pool_seed, seed = tuning.replicate_seeds(a.replicate)
    n_budget = 0 if zero else a.n
    arm_sampling = a.arm_sampling or a.sampling
    key = None if zero else tuning.arm_key(task.name, hf, a.technique, a.n, arm_sampling)

    if a.tune:  # do not spend GPU time on an arm that is already frozen
        existing = tuning.load_arms(a.frozen_arms)
        if key in existing:
            rec = {"status": "already_frozen", "arm_key": key, "selection": existing[key],
                   "frozen_arms_sha256": tuning.file_hash(a.frozen_arms)}
            print(f"[icl] already_frozen {key} k={existing[key]['k']}", flush=True)
            return rec

    texts, labels = (loader or data.load_train_split)()  # train split only; the test split is never read
    splits = data.make_splits(len(texts), pool_seed, sampling=a.sampling, labels=labels)
    pool_idx = [] if zero else splits.pool(a.n)
    val_idx = list(splits.val)[: tuning.val_size(a.n)] if a.tune else list(splits.val)
    if a.eval_limit:
        val_idx = val_idx[: a.eval_limit]
    pool = [Example(data.item_id(i), texts[i], labels[i]) for i in pool_idx]
    val = [Example(data.item_id(i), texts[i], labels[i]) for i in val_idx]
    gold = [e.output for e in val]

    emb = None
    if a.technique == "icl_retrieved":
        emb = embedder or SentenceTransformerEmbedder(device=a.device)
    rand_seed = a.random_seed if a.random_seed is not None else seed
    t0 = time.perf_counter()
    selector = ShotSelector(a.technique, pool, val, embedder=emb, cache_dir=a.cache_dir, seed=rand_seed)
    select_seconds = time.perf_counter() - t0
    gen = generator or _default_generator(hf, a)

    rid = run_id_for(hf, a.technique, n_budget, seed, pool_seed, a.sampling, a.constrained)
    base = {"model": hf, "technique": a.technique, "task": task.name, "n": n_budget, "seed": seed,
            "pool_seed": pool_seed, "sampling": a.sampling, "replicate": a.replicate,
            "embedder": emb.name if emb else None, "select_seconds": select_seconds,
            "pool_hash": None if zero else data.indices_hash(pool_idx),
            "val_hash": data.indices_hash(val_idx), "decoding": "constrained" if a.constrained else "free",
            "git_commit": env.git_commit(), "timestamp_utc": env.utc_now()}

    if a.tune:
        per_k: Dict[int, float] = {}
        tokens_in = tokens_out = 0
        seconds = 0.0
        for k in ks:
            parsed, gens, _, sec = score_k(gen, task, selector, val, k)
            per_k[k] = accuracy(parsed, gold)
            tokens_in += sum(g.input_tokens or 0 for g in gens)
            tokens_out += sum(g.output_tokens or 0 for g in gens)
            seconds += sec
        sel = select_k(per_k)
        entry = {**sel, "ks": list(ks), "val_size": len(val), "tuning_replicate": 0,
                 "technique": a.technique, "model": hf, "embedder": base["embedder"],
                 "decoding": "free", "max_new_tokens": task.max_new_tokens,
                 "zero_shot_system_sha256": _system_hash(task),
                 "pool_hash": base["pool_hash"], "val_hash": base["val_hash"]}
        sha = tuning.freeze_arm(a.frozen_arms, key, entry)
        record = {"status": "tuned", "run_id": rid + "__tune", "arm_key": key, "frozen_arms_sha256": sha,
                  "selection": entry, "prompt_tokens": tokens_in, "output_tokens": tokens_out,
                  "generate_seconds": seconds, **base}
        msg = f"k={sel['k']} val_acc={sel['val_accuracy']:.4f}"
    else:
        if zero:
            arm, k, frozen = None, 0, None
        else:
            arm = tuning.load_arm(a.frozen_arms, key)
            k = int(arm["k"])
            frozen = {"arm_key": key, "frozen_arms_sha256": tuning.file_hash(a.frozen_arms)}
        parsed, gens, shots, seconds = score_k(gen, task, selector, val, k, constrained=a.constrained)
        recs = [make_record(run_id=rid, system=hf, task=task, technique=a.technique, n_budget=n_budget,
                            seed=rand_seed if a.technique == "icl_random" else None, item=it, gen=g, k=k,
                            shot_ids=[s.id for s in sh], max_cosine=mc, constrained=a.constrained,
                            extra={"pool_seed": pool_seed, "pool_hash": base["pool_hash"],
                                   "replicate": a.replicate})
                for it, g, sh, mc in zip(val, gens, shots, selector.max_cosine)]
        pred_path = Path(a.predictions_dir) / f"{rid}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        record = {"status": "ok", "run_id": rid, "k": k, "frozen_arm": frozen,
                  "accuracy": sum(r["parsed"] == r["gold"] for r in recs) / len(recs),
                  "n_eval": len(recs), "n_unparseable": sum(not r["parse_ok"] for r in recs),
                  "prompt_tokens": sum(g.input_tokens or 0 for g in gens),
                  "output_tokens": sum(g.output_tokens or 0 for g in gens),
                  "generate_seconds": seconds, "eval_split": "validation", **base}
        msg = f"k={k} acc={record['accuracy']:.4f}"
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    with open(a.results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[icl] {record['status']} {key or 'zero_shot'} {msg}", flush=True)
    return record


def _system_hash(task: IclTask) -> str:
    import hashlib

    return hashlib.sha256(render(task, [], "").system.encode("utf-8")).hexdigest()


def _default_generator(hf_id: str, a: argparse.Namespace) -> Generator:
    from src.icl.generate import VLLMGenerator

    return VLLMGenerator(hf_id, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization)


def main(argv: Optional[List[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
