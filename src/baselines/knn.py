"""kNN baseline: similarity-weighted vote over the N-example pool (protocol section 2, L-01).

    python -m src.baselines.knn --n 200 --sampling stratified --replicate 0 --tune --frozen-arms A
    python -m src.baselines.knn --n 200 --sampling stratified --replicate 1 --frozen-arms A

Embeddings: BAAI/bge-small-en-v1.5 at the pinned revision (src/api/retrieval.py), normalised,
cosine similarity, no query prefix (symmetric short texts). k is chosen from {1, 5, 16} on
val(N) of replicate 0, like every other arm (equal effort: 3 values here, 4 lr trials for
fine-tuning), frozen in frozen_arms.json and never re-selected. Final runs score the full
1,000-row validation carve-out of replicate r >= 1. The test split is never read.

Vote: each of the k neighbours adds max(cosine, 0) to its label. Ties go to the label with more
neighbours, then to the label of the nearest neighbour; the result is deterministic.
kNN has no training randomness: a replicate only changes the pool (pool_seed = r).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

KS = (1, 5, 16)
SYSTEM_PREFIX = "knn:"


# ---------------------------------------------------------------- pure logic

def top_k(sims_row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest similarities, ties broken by pool order."""
    return np.argsort(-sims_row, kind="stable")[: min(k, len(sims_row))]


def vote(sims_row: np.ndarray, pool_labels: Sequence[str], k: int) -> str:
    idx = top_k(sims_row, k)
    weight: Dict[str, float] = {}
    count: Dict[str, int] = {}
    first: Dict[str, int] = {}
    for rank, i in enumerate(idx):
        lab = pool_labels[i]
        weight[lab] = weight.get(lab, 0.0) + max(float(sims_row[i]), 0.0)
        count[lab] = count.get(lab, 0) + 1
        first.setdefault(lab, rank)
    return max(weight, key=lambda lab: (weight[lab], count[lab], -first[lab]))


def predict(sims: np.ndarray, pool_labels: Sequence[str], k: int) -> List[str]:
    if sims.ndim != 2 or sims.shape[1] != len(pool_labels):
        raise ValueError(f"sims must be (queries, pool={len(pool_labels)}), got {sims.shape}")
    return [vote(row, pool_labels, k) for row in sims]


def accuracy(pred: Sequence[str], gold: Sequence[str]) -> float:
    if len(pred) != len(gold) or not gold:
        raise ValueError("pred and gold must be non-empty and of equal length")
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def select_k(sims: np.ndarray, pool_labels: Sequence[str], gold: Sequence[str],
             ks: Sequence[int] = KS) -> Dict[str, Any]:
    """Best k by validation accuracy; ties go to the smaller k."""
    per_k = {int(k): accuracy(predict(sims, pool_labels, k), gold) for k in sorted(ks)}
    best = max(per_k, key=lambda k: (per_k[k], -k))
    return {"k": best, "val_accuracy": per_k[best], "per_k": per_k,
            "at_grid_edge": "high" if best == max(ks) and len(ks) > 1 else None}


def cosine(queries: np.ndarray, pool: np.ndarray) -> np.ndarray:
    def norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n
    return norm(np.asarray(queries, dtype=np.float32)) @ norm(np.asarray(pool, dtype=np.float32)).T


# ---------------------------------------------------------------- runner

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--sampling", choices=("random", "stratified"), default="random")
    p.add_argument("--arm-sampling", choices=("random", "stratified"), default=None)
    p.add_argument("--replicate", type=int, required=True)
    p.add_argument("--tune", action="store_true", help="replicate 0: choose k on val(N) and freeze it")
    p.add_argument("--frozen-arms", required=True)
    p.add_argument("--results", default="runs/knn.jsonl")
    p.add_argument("--predictions-dir", default="runs/predictions")
    p.add_argument("--cache-dir", default="runs/emb_cache")
    p.add_argument("--device", default=None)
    p.add_argument("--eval-limit", type=int, default=None, help="smoke mode: first k validation rows")
    return p


def run(argv: Optional[List[str]] = None, embedder=None, loader=None) -> Dict[str, Any]:
    """embedder / loader are injectable for offline tests (default: pinned bge, Hub train split)."""
    from src.api.retrieval import SentenceTransformerEmbedder, embed_cached
    from src.train import data, env, task_spec, tuning
    from src.train.config import run_id
    from src.train.evaluate import make_record

    a = build_parser().parse_args(argv)
    if a.tune and a.replicate != tuning.TUNING_REPLICATE:
        raise ValueError("--tune runs on replicate 0 only")
    if not a.tune and a.replicate == tuning.TUNING_REPLICATE:
        raise ValueError("replicate 0 is the tuning replicate; finals use --replicate >= 1")
    pool_seed, seed = tuning.replicate_seeds(a.replicate)
    emb = embedder or SentenceTransformerEmbedder(device=a.device)
    system = SYSTEM_PREFIX + emb.name

    texts, labels = (loader or data.load_train_split)()
    splits = data.make_splits(len(texts), pool_seed, sampling=a.sampling, labels=labels)
    pool = splits.pool(a.n)
    val = list(splits.val)[: tuning.val_size(a.n)] if a.tune else list(splits.val)
    if a.eval_limit:
        val = val[: a.eval_limit]
    pool_labels = [labels[i] for i in pool]
    gold = [labels[i] for i in val]

    t0 = time.perf_counter()
    pv = embed_cached(emb, [texts[i] for i in pool], a.cache_dir)
    qv = embed_cached(emb, [texts[i] for i in val], a.cache_dir)
    embed_seconds = time.perf_counter() - t0
    sims = cosine(qv, pv)
    key = tuning.arm_key(task_spec.TASK, system, "knn", a.n, a.arm_sampling or a.sampling)
    rid = run_id(system.replace(":", "-"), "knn", a.n, seed, pool_seed, a.sampling)
    base = {"model": system, "technique": "knn", "n": a.n, "seed": seed, "pool_seed": pool_seed,
            "sampling": a.sampling, "task": task_spec.TASK, "replicate": a.replicate,
            "embed_seconds": embed_seconds, "pool_hash": data.indices_hash(pool),
            "val_hash": data.indices_hash(val), "git_commit": env.git_commit(),
            "timestamp_utc": env.utc_now()}

    if a.tune:
        sel = select_k(sims, pool_labels, gold)
        entry = {**sel, "ks": list(KS), "embedder": emb.name, "val_size": len(val),
                 "tuning_replicate": 0, "pool_hash": base["pool_hash"], "val_hash": base["val_hash"]}
        sha = tuning.freeze_arm(a.frozen_arms, key, entry)
        record = {"status": "tuned", "run_id": rid + "__tune", "arm_key": key,
                  "frozen_arms_sha256": sha, "selection": entry, **base}
    else:
        arm = tuning.load_arm(a.frozen_arms, key)
        t1 = time.perf_counter()
        pred = predict(sims, pool_labels, arm["k"])
        predict_seconds = time.perf_counter() - t1
        recs = []
        for i, p_, g in zip(val, pred, gold):
            r = make_record(run_id=rid, system=system, technique="knn", n=a.n, seed=seed, train_index=i,
                            prediction=p_, gold=g, input_tokens=None, output_tokens=None)
            r["arm"] = "baseline"
            r["extra"].update({"pool_seed": pool_seed, "pool_hash": base["pool_hash"], "k": arm["k"]})
            recs.append(r)
        pred_path = Path(a.predictions_dir) / f"{rid}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        record = {"status": "ok", "run_id": rid, "k": arm["k"],
                  "frozen_arm": {"arm_key": key, "frozen_arms_sha256": tuning.file_hash(a.frozen_arms)},
                  "accuracy": sum(r["parsed"] == r["gold"] for r in recs) / len(recs),
                  "n_eval": len(recs), "n_unparseable": sum(not r["parse_ok"] for r in recs),
                  "predict_seconds": predict_seconds, "eval_split": "validation", **base}
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    with open(a.results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[knn] {record['status']} {key} " + (f"k={record['selection']['k']} val_acc={record['selection']['val_accuracy']:.4f}"
          if a.tune else f"k={record['k']} acc={record['accuracy']:.4f}"), flush=True)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
