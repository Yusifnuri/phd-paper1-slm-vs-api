"""BillSum memory and throughput probe, stage 4 of the gated pilot (GPU, about 0.7 GPU-h at most).

    source ~/vault/Defaukt/paper1/env.sh
    python scripts/billsum_probe.py --model qwen3-4b --results runs/stage4/probe.jsonl

Trains 20 optimiser steps on the longest 1% of the US train bills with micro-batch 1, gradient
checkpointing and the completion-only loss, measures peak memory and tokens/s, and projects the
time of one N=200/1,000/3,000 trial with the whole schedule (upper bound). Prints the go/no-go
class of the protocol for the Qwen N=3,000 final. Train split only; the test split is never read.
An out-of-memory error is recorded as a failed probe, not hidden.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import billsum, env, probe, train  # noqa: E402
from src.train.config import TrainConfig  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=4, help="effective batch (micro-batch is 1)")
    p.add_argument("--frac", type=float, default=0.01)
    p.add_argument("--results", default="runs/stage4/probe.jsonl")
    a = p.parse_args(argv)

    cfg = TrainConfig(model=a.model, task="billsum", n=200, replicate=None, pool_seed=0, seed=100,
                      effective_batch=a.batch, max_steps=a.steps)
    spec, texts, targets, splits, tok, pad_id = train._prepare(cfg)
    full = {i: billsum.tokenize_example(tok, texts[i], targets[i], 10 ** 9) for i in range(len(texts))}
    lengths = {i: len(e["input_ids"]) for i, e in full.items()}
    idx = probe.longest_fraction(lengths, a.frac)
    examples = [full[i] for i in idx]
    mean_tokens = statistics.mean(lengths.values())
    print(f"[probe] {len(idx)} longest bills, {max(lengths.values())} max tokens, "
          f"mean over all train bills {mean_tokens:.0f}", flush=True)

    row = {"status": "ok", "task": "billsum", "model": spec.hf_id, "steps": a.steps, "effective_batch": a.batch,
           "micro_batch": 1, "grad_checkpointing": True, "n_longest": len(idx),
           "longest_tokens": max(lengths.values()), "mean_tokens_all_train": mean_tokens,
           **env.gpu_info(), "libs": env.lib_versions(), "git_commit": env.git_commit(), "timestamp_utc": env.utc_now()}
    try:
        model, load_s, n_train = train._build_model(spec, cfg)
        stats = train.train_loop(model, examples, cfg, pad_id)
        row.update(load_seconds=load_s, trainable_params=n_train, **{k: stats[k] for k in (
            "train_seconds", "tokens_per_second", "peak_mem_bytes", "first_train_loss", "final_train_loss")})
        row["peak_mem_gb"] = stats["peak_mem_bytes"] / 1e9
        proj = {n: probe.projected_seconds(n, mean_tokens, stats["tokens_per_second"]) for n in (200, 1000, 3000)}
        row["projected_one_run_seconds_upper_bound"] = proj
        row["go_no_go_qwen_n3000"] = probe.go_no_go(proj[3000]["seconds"])
        del model
        train._release_gpu()
    except Exception as e:  # OOM or anything else: recorded, never hidden
        row.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    with open(a.results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: row.get(k) for k in ("status", "peak_mem_gb", "tokens_per_second", "train_seconds",
                      "first_train_loss", "final_train_loss", "projected_one_run_seconds_upper_bound",
                      "go_no_go_qwen_n3000", "error")}, indent=1, default=str))
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
