#!/usr/bin/env python3
"""Run the pilot grid (models x techniques x N), one subprocess per combination.

    python scripts/run_pilot.py --hub-repo USER/paper1-pilot-adapters
    python scripts/run_pilot.py --smoke              # ~2 min sanity check, writes runs/smoke.jsonl

Each combination runs in its own process, so a CUDA OOM or crash cannot poison the next
run and GPU memory is fully released between runs. Combinations already recorded with
status "ok" in the results file are skipped; failures are recorded and retried next time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train.config import TECHNIQUES, run_id  # noqa: E402
from src.train.registry import MODELS, resolve  # noqa: E402

Key = Tuple[str, str, int, int, int, str]  # model, technique, N, seed, pool_seed, sampling


def plan(models: List[str], techniques: List[str], ns: List[int], seeds: List[int],
         pool_seeds: List[int] = (42,), sampling: str = "random") -> List[Key]:
    """Every replicate (pool_seed x seed) is run for every method, so methods share pools."""
    out: List[Key] = []
    for m in models:
        hf = resolve(m).hf_id
        out += [(hf, t, n, sd, ps, sampling)
                for ps in pool_seeds for sd in seeds for t in techniques for n in ns]
    return out


def read_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a torn last line after a crash
    return rows


def key_of(r: Dict[str, Any]) -> Key:
    return (r["model"], r["technique"], r["n"], r["seed"], r.get("pool_seed", 42), r.get("sampling", "random"))


def done_keys(rows: List[Dict[str, Any]]) -> set:
    return {key_of(r) for r in rows if r.get("status") == "ok"}


def build_command(key: Key, a: argparse.Namespace, results: Path) -> List[str]:
    model, tech, n, seed, pool_seed, sampling = key
    cmd = [sys.executable, "-m", "src.train.train", "--model", model, "--technique", tech,
           "--n", str(n), "--seed", str(seed), "--pool-seed", str(pool_seed),
           "--sampling", sampling, "--epochs", str(a.epochs), "--results", str(results)]
    if a.tune and not a.smoke:
        cmd.append("--tune")
    if a.hub_repo and not a.smoke:
        cmd += ["--hub-repo", a.hub_repo]
    if a.smoke:
        cmd += ["--max-steps", "10", "--eval-limit", "64",
                "--predictions-dir", "runs/smoke_predictions", "--checkpoint-dir", "checkpoints/smoke"]
    return cmd


def record_failure(results: Path, key: Key, returncode: int, tail: str) -> None:
    model, tech, n, seed, pool_seed, sampling = key
    row = {"status": "failed", "run_id": run_id(model, tech, n, seed, pool_seed, sampling), "model": model,
           "technique": tech, "n": n, "seed": seed, "pool_seed": pool_seed, "sampling": sampling, "returncode": returncode, "error_tail": tail}
    results.parent.mkdir(parents=True, exist_ok=True)
    with open(results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def latest_by_key(rows: List[Dict[str, Any]]) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for r in rows:
        out[key_of(r)] = r  # later lines win, so a successful retry replaces a failure
    return out


def summary_table(rows: List[Dict[str, Any]]) -> str:
    latest = latest_by_key(rows)
    lora_t = {(k[0], k[2], k[3], k[4]): r["train_seconds"] for k, r in latest.items()
              if k[1] == "lora" and r.get("status") == "ok"}
    head = ["model", "technique", "N", "status", "train s", "load s", "peak GB", "tok/s", "acc", "vs lora", "hub"]
    lines = [head]
    for k in sorted(latest):
        r = latest[k]
        if r.get("status") != "ok":
            lines.append([k[0].split("/")[-1], k[1], str(k[2]), "FAILED"] + ["-"] * 7)
            continue
        base = lora_t.get((k[0], k[2], k[3], k[4]))
        rel = f"{r['train_seconds'] / base:.2f}x" if base and k[1] != "lora" else "-"
        hub = "yes" if r.get("hub_pushed") else ("no" if r.get("hub_repo") else "n/a")
        lines.append([k[0].split("/")[-1], k[1], str(k[2]), "ok", f"{r['train_seconds']:.0f}",
                      f"{r['load_seconds']:.0f}", f"{r['peak_mem_gb']:.1f}",
                      f"{r['tokens_per_second']:.0f}", f"{r['accuracy']:.3f}", rel, hub])
    widths = [max(len(row[i]) for row in lines) for i in range(len(head))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), help="registry keys or HF ids")
    p.add_argument("--techniques", nargs="+", choices=TECHNIQUES, default=None)
    p.add_argument("--ns", nargs="+", type=int, default=None, help="default: 1000 (smoke: 50)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="training seeds")
    p.add_argument("--pool-seeds", nargs="+", type=int, default=[42], help="pool seeds (replicates)")
    p.add_argument("--sampling", choices=("random", "stratified"), default="random")
    p.add_argument("--tune", action="store_true", help="choose lr and epochs by validation loss per run")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--hub-repo", default=None, help="private Hub repo 'user/name' for adapters")
    p.add_argument("--results", default=None, help="default runs/pilot.jsonl (smoke: runs/smoke.jsonl)")
    p.add_argument("--smoke", action="store_true", help="N=50, 10 steps, 64 eval items, no Hub push")
    p.add_argument("--force", action="store_true", help="rerun combinations already recorded ok")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = p.parse_args(argv)

    techniques = a.techniques or (["lora"] if a.smoke else list(TECHNIQUES))
    ns = a.ns or ([50] if a.smoke else [1000])
    results = ROOT / (a.results or ("runs/smoke.jsonl" if a.smoke else "runs/pilot.jsonl"))
    keys = plan(a.models, techniques, ns, a.seeds, a.pool_seeds, a.sampling)
    done = set() if a.force else done_keys(read_results(results))

    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} combinations, {len(keys) - len(todo)} already done, {len(todo)} to run -> {results}")
    if not a.hub_repo and not a.smoke:
        print("WARNING: no --hub-repo; adapters stay on the ephemeral disk.")
    for i, key in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {key} ===", flush=True)
        if a.dry_run:
            print(" ".join(build_command(key, a, results)))
            continue
        proc = subprocess.Popen(build_command(key, a, results), cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        lines: List[str] = []
        for line in proc.stdout:  # stream live, keep a tail for the failure record
            sys.stdout.write(line)
            lines.append(line.rstrip("\n"))
            lines = lines[-40:]
        returncode = proc.wait()
        if returncode != 0:
            print(f"!!! failed (exit {returncode}); continuing", flush=True)
            record_failure(results, key, returncode, "\n".join(lines[-15:])[-2000:])
    print("\n" + summary_table(read_results(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
