"""The one path from a frozen arm to a held-out prediction file (review 06d, F1).

    python -m src.final.run_final configs --frozen-arms results/final/frozen_arms.json \
        --out results/final/final_configs.json          # list every confirmatory-side run, then commit
    python -m src.final.run_final run --config-hash <hex> [--adapter-dir DIR]
    python -m src.final.run_final dry-run --config-hash <hex> --out-dir runs/dry [--adapter-dir DIR]

Covers the local arms (fine-tuned, kNN, local ICL). The API arm has no path yet (it is built with the
API work, before tag B).

Order of one `run`: build the configuration; refuse if it already produced a prediction file; load
the train pool and the model (everything that can fail without held-out data); pass every gate of
`final_eval`; predict with the SAME functions the validation runners use; write the predictions
atomically. A repeat read is possible only through final_eval's infrastructure-failure reason AND only
while no prediction file exists: a run that produced a result is never repeated (L-42).

`dry-run` does the same inside a throw-away git repository, with validation rows served as if they
were held-out rows. It exercises the real gates; its records are marked "validation-dryrun".

final_eval needs git history and tags: on a server, deploy with `git clone` of a `git bundle`, not
with `git archive`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.train import billsum, data, env, tuning
from src.train.config import run_id

from . import ledger
from .final_eval import final_eval

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = "results/final"
REPLICATES = {"banking77": 8, "billsum": 5}
RANDOM_POOL_REPLICATES = 3  # BANKING77 N=200 random pools (L-04)
ICL_TECHNIQUES = ("icl_retrieved", "icl_random")
SPLIT_REAL, SPLIT_DRY = "heldout", "validation-dryrun"


class FinalRunError(RuntimeError):
    """The run cannot start; nothing was read."""


# ---------------------------------------------------------------- configurations

def parse_arm_key(key: str) -> Dict[str, Any]:
    task, model, technique, n, sampling = key.split("|")
    return {"task": task, "model": model, "technique": technique, "n": int(n[1:]), "sampling": sampling}


def make_config(arm_key: str, arm: Mapping[str, Any], replicate: int,
                pool_sampling: Optional[str] = None) -> Dict[str, Any]:
    """The hashed identity of one held-out evaluation: the frozen arm itself, the replicate and the
    pool sampling (different from the arm's only for the random-pool N=200 condition)."""
    p = parse_arm_key(arm_key)
    if replicate == tuning.TUNING_REPLICATE:
        raise FinalRunError("replicate 0 is the tuning replicate and is never analysed")
    pool_seed, seed = tuning.replicate_seeds(replicate)
    return {"arm_key": arm_key, "arm": dict(arm), "task": p["task"], "model": p["model"],
            "technique": p["technique"], "n": p["n"], "arm_sampling": p["sampling"],
            "sampling": pool_sampling or p["sampling"], "replicate": replicate,
            "pool_seed": pool_seed, "train_seed": seed}


def list_configs(arms: Mapping[str, Any], replicates: Optional[Mapping[str, int]] = None,
                 random_pool_n200: bool = True) -> List[Dict[str, Any]]:
    reps = {**REPLICATES, **(replicates or {})}
    out: List[Dict[str, Any]] = []
    for key in sorted(arms):
        p = parse_arm_key(key)
        for r in range(1, reps[p["task"]] + 1):
            out.append(make_config(key, arms[key], r))
        if random_pool_n200 and p["task"] == "banking77" and p["n"] == 200 and p["sampling"] == "stratified":
            out += [make_config(key, arms[key], r, "random") for r in range(1, RANDOM_POOL_REPLICATES + 1)]
    return out


def write_final_configs(frozen_arms: Path, out: Path, replicates: Optional[Mapping[str, int]] = None) -> int:
    arms = tuning.load_arms(str(frozen_arms))
    if not arms:
        raise FinalRunError(f"no frozen arms in {frozen_arms}")
    configs = {}
    for c in list_configs(arms, replicates):
        configs[ledger.config_hash(c)] = {"task": c["task"], "arm_family": "A", "config": c,
                                          "note": f"{c['arm_key']} r{c['replicate']} pool={c['sampling']}"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": 1, "frozen_arms_sha256": tuning.file_hash(str(frozen_arms)),
                               "configs": configs}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return len(configs)


# ---------------------------------------------------------------- held-out items

@dataclass(frozen=True)
class Item:
    index: int   # position in the held-out rows (BANKING77) or original row id (BillSum)
    item_id: str
    text: str
    gold: str


def items_from(task: str, loaded: Mapping[str, Any]) -> List[Item]:
    golds = loaded["summaries"] if task == "billsum" else loaded["labels"]
    idx = loaded.get("indices") or list(range(len(golds)))
    tag = SPLIT_REAL
    return [Item(int(i), f"{task}-{tag}-{i}", t, g) for i, t, g in zip(idx, loaded["texts"], golds)]


@dataclass(frozen=True)
class Pool:
    indices: List[int]
    texts: List[str]
    labels: List[str]  # gold labels or gold summaries


def load_pool(config: Mapping[str, Any], train_loader: Optional[Callable] = None) -> Pool:
    task, n = config["task"], config["n"]
    if task == "billsum":
        texts, labels = (train_loader or billsum.load_train_split)()
        splits = billsum.make_splits(config["pool_seed"])
    else:
        texts, labels = (train_loader or data.load_train_split)()
        splits = data.make_splits(len(texts), config["pool_seed"], sampling=config["sampling"], labels=labels)
    idx = splits.pool(n)
    return Pool(idx, [texts[i] for i in idx], [labels[i] for i in idx])


def _rid(config: Mapping[str, Any]) -> str:
    system = config["model"].replace(":", "-")
    return run_id(system, config["technique"], config["n"], config["train_seed"], config["pool_seed"],
                  config["sampling"], config["task"])


def _finish(records: List[Dict[str, Any]], config: Mapping[str, Any], items: Sequence[Item],
            pool: Pool, split: str) -> List[Dict[str, Any]]:
    """Give every record its held-out id and the fields the contrast builder needs."""
    if len(records) != len(items):
        raise RuntimeError(f"{len(records)} records for {len(items)} items")
    dup = None
    if config["task"] == "billsum":  # near-duplicate flag against the pool (F9.6)
        dup = billsum.max_jaccard_to_pool([i.text for i in items], pool.texts)
    for j, (r, it) in enumerate(zip(records, items)):
        r["item_id"] = it.item_id
        r["extra"].update({"split": split, "replicate": config["replicate"], "pool_seed": config["pool_seed"],
                           "pool_hash": data.indices_hash(pool.indices)})
        if dup is not None:
            r["extra"]["max_jaccard_to_pool"] = dup[j][0]
    return records


# ---------------------------------------------------------------- predictors
# Each factory does all heavy loading first and returns predict(items) -> records.

Predictor = Callable[[Sequence[Item]], List[Dict[str, Any]]]


def knn_predictor(config: Mapping[str, Any], pool: Pool, *, embedder=None, cache_dir: str = "runs/emb_cache",
                  device: Optional[str] = None) -> Predictor:
    from src.api.retrieval import SentenceTransformerEmbedder, embed_cached
    from src.baselines import knn
    from src.train.evaluate import make_record

    emb = embedder or SentenceTransformerEmbedder(device=device)
    if config["model"] != knn.SYSTEM_PREFIX + emb.name:
        raise FinalRunError(f"arm was frozen with {config['model']}, embedder is {emb.name}")
    pool_vecs = embed_cached(emb, pool.texts, cache_dir)

    def predict(items: Sequence[Item]) -> List[Dict[str, Any]]:
        sims = knn.cosine(embed_cached(emb, [i.text for i in items], cache_dir), pool_vecs)
        k = config["arm"]["k"]
        out = []
        for row, it, p in zip(sims, items, knn.predict(sims, pool.labels, k)):
            r = make_record(run_id=_rid(config), system=config["model"], technique="knn", n=config["n"],
                            seed=config["train_seed"], train_index=it.index, prediction=p, gold=it.gold,
                            input_tokens=None, output_tokens=None)
            r["arm"] = "baseline"
            r["extra"].update({"k": k, "max_cosine_to_pool": float(row.max())})
            out.append(r)
        return out

    return predict


def icl_predictor(config: Mapping[str, Any], pool: Pool, *, embedder=None, generator=None, tokenizer=None,
                  cache_dir: str = "runs/emb_cache", device: Optional[str] = None) -> Predictor:
    from src.api.prompts import Example
    from src.api.retrieval import SentenceTransformerEmbedder
    from src.icl import cli, retrievers
    from src.icl.render import ShotSelector, get_task

    task = get_task(config["task"])
    is_sum = task.name == "billsum"
    technique, arm = config["technique"], config["arm"]
    if arm.get("zero_shot_system_sha256") not in (None, cli._system_hash(task)):
        raise FinalRunError("the system prompt differs from the one the arm was frozen with")
    retriever = arm.get("retriever") or retrievers.DEFAULT_RETRIEVER
    iid = (lambda i: billsum.item_id("train", i)) if is_sum else data.item_id
    examples = [Example(iid(i), t, g) for i, t, g in zip(pool.indices, pool.texts, pool.labels)]
    emb = None
    if technique == "icl_retrieved" and retriever == retrievers.DEFAULT_RETRIEVER:
        emb = embedder or SentenceTransformerEmbedder(device=device)
    if generator is None:
        a = argparse.Namespace(max_model_len=cli.default_max_model_len(task), gpu_memory_utilization=0.85)
        generator = cli._default_generator(config["model"], a)

    def predict(items: Sequence[Item]) -> List[Dict[str, Any]]:
        queries = [Example(i.item_id, i.text, i.gold) for i in items]
        sims = None
        if technique == "icl_retrieved" and emb is None:
            sims = retrievers.similarity(retriever, pool.texts, [i.text for i in items])
        selector = ShotSelector(technique, examples, queries, embedder=emb, cache_dir=cache_dir,
                                seed=config["train_seed"], sims=sims)
        k = int(arm["k"])
        parsed, gens, shots, _ = cli.score_k(generator, task, selector, queries, k)
        gold = [q.output for q in queries]
        rouge = cli.score_values(task, parsed, gold) if is_sum else None
        toks = cli._gold_token_counts(config["model"], gold, tokenizer) if is_sum else None
        return [cli.make_record(
            run_id=_rid(config), system=config["model"], task=task, technique=technique, n_budget=config["n"],
            seed=config["train_seed"] if technique == "icl_random" else None, item=q, gen=g, k=k,
            shot_ids=[s.id for s in sh], max_cosine=mc, constrained=False,
            extra={**(cli.billsum_record_extra(rouge[j], g, q.output, toks[j]) if is_sum else {}),
                   **({"retriever": retriever} if is_sum else {})})
            for j, (q, g, sh, mc) in enumerate(zip(queries, gens, shots, selector.max_cosine))]

    return predict


def check_adapter(adapter_dir: Path, config: Mapping[str, Any], pool: Pool) -> Dict[str, str]:
    """The adapter must be the one train.run saved for this arm and replicate. Returns file hashes."""
    manifest = adapter_dir / "run_manifest.json"
    if not manifest.exists():
        raise FinalRunError(f"{adapter_dir} has no run_manifest.json (train with the current train.py)")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    want = {"arm_key": config["arm_key"], "arm": config["arm"], "replicate": config["replicate"],
            "pool_hash": data.indices_hash(pool.indices), "stop_step": config["arm"]["stop_step"]}
    bad = [k for k, v in want.items() if m.get(k) != v]
    if bad:
        raise FinalRunError(f"adapter manifest does not match the configuration: {bad}")
    files = sorted(p for p in adapter_dir.iterdir() if p.name.startswith("adapter_"))
    if not files:
        raise FinalRunError(f"no adapter_* files in {adapter_dir}")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def finetuned_predictor(config: Mapping[str, Any], pool: Pool, *, adapter_dir: Path,
                        eval_batch: int = 64, gen_batch: int = 8) -> Predictor:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.train import evaluate, summarise
    from src.train.config import TrainConfig

    cfg = TrainConfig(model=config["model"], technique=config["technique"], n=config["n"], task=config["task"])
    if config["technique"] == "qlora":
        from src.train import train
        from src.train.registry import resolve
        base = train.load_model(resolve(config["model"]), cfg)
    else:
        base = AutoModelForCausalLM.from_pretrained(config["model"], dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    tok = AutoTokenizer.from_pretrained(config["model"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def predict(items: Sequence[Item]) -> List[Dict[str, Any]]:
        triples = [(i.index, i.text, i.gold) for i in items]
        common = dict(run_id=_rid(config), system=config["model"], technique=config["technique"],
                      n=config["n"], seed=config["train_seed"], items=triples, pad_id=pad_id)
        if config["task"] == "billsum":
            stops = summarise.summary_stop_ids(tok, pool.labels)
            return summarise.evaluate_summaries(model, tok, stop_ids=stops, batch_size=gen_batch,
                                                max_new_tokens=billsum.MAX_NEW_TOKENS, **common)["records"]
        _, comp = data.render_pair(tok, "x", pool.labels[0])
        stops = evaluate.stop_token_ids(tok, tok(comp, add_special_tokens=False)["input_ids"])
        return evaluate.evaluate(model, tok, stop_ids=stops, batch_size=eval_batch,
                                 max_new_tokens=cfg.max_new_tokens, **common)["records"]

    return predict


def default_predictor(config: Mapping[str, Any], pool: Pool, adapter_dir: Optional[Path]) -> Predictor:
    if config["technique"] == "knn":
        return knn_predictor(config, pool)
    if config["technique"] in ICL_TECHNIQUES:
        return icl_predictor(config, pool)
    if adapter_dir is None:
        raise FinalRunError("a fine-tuned configuration needs --adapter-dir")
    return finetuned_predictor(config, pool, adapter_dir=adapter_dir)


# ---------------------------------------------------------------- running

@dataclass(frozen=True)
class Paths:
    repo: Path
    final_configs: Path
    frozen_arms: Path
    ledger: Path
    fingerprint: Path
    out_dir: Path

    @classmethod
    def default(cls, repo: Path = REPO_ROOT) -> "Paths":
        d = repo / FINAL_DIR
        return cls(repo, d / "final_configs.json", d / "frozen_arms.json", d / "test_reads.jsonl",
                   repo / "results" / "test_split_fingerprint.json", d)


def lookup_config(final_configs: Path, cfg_hash: str) -> Dict[str, Any]:
    entries = json.loads(final_configs.read_text(encoding="utf-8")).get("configs", {})
    found = [h for h in entries if h.startswith(cfg_hash)]
    if len(found) != 1:
        raise FinalRunError(f"{len(found)} configurations match hash {cfg_hash!r}")
    config = entries[found[0]]["config"]
    if ledger.config_hash(config) != found[0]:
        raise FinalRunError("the stored configuration does not hash to its key")
    return config


def run_config(config: Mapping[str, Any], paths: Paths, *, adapter_dir: Optional[Path] = None,
               predictor_factory: Optional[Callable] = None, loader: Optional[Callable] = None,
               train_loader: Optional[Callable] = None, split: str = SPLIT_REAL,
               rerun_reason: Optional[str] = None, rerun_detail: Optional[str] = None) -> Dict[str, Any]:
    cfg_hash = ledger.config_hash(dict(config))
    pred_path = paths.out_dir / "predictions" / f"{cfg_hash[:16]}__{_rid(config)}.jsonl"
    if pred_path.exists():
        raise FinalRunError(f"{pred_path} exists: this configuration already produced a result (L-42)")
    pool = load_pool(config, train_loader)
    adapter_hashes = None
    if config["technique"] not in ("knn",) + ICL_TECHNIQUES and adapter_dir is not None:
        adapter_hashes = check_adapter(adapter_dir, config, pool)
    predict = (predictor_factory or default_predictor)(config, pool, adapter_dir)

    read = final_eval(
        config, task=config["task"], repo_root=paths.repo, final_configs_path=paths.final_configs,
        frozen_arms_path=paths.frozen_arms, ledger_path=paths.ledger, fingerprint_path=paths.fingerprint,
        loader=loader, rerun_reason=rerun_reason, rerun_detail=rerun_detail)

    items = items_from(config["task"], read.data)
    records = _finish(predict(items), config, items, pool, split)
    from src.eval.schema import PredictionRecord, check_run_consistency
    check_run_consistency([PredictionRecord.from_dict(r) for r in records])

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pred_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, pred_path)

    if config["task"] == "billsum":
        score = {"rouge_l": sum(r["extra"]["rouge_l"] for r in records) / len(records)}
    else:
        score = {"accuracy": sum(r["parse_ok"] and r["parsed"] == r["gold"] for r in records) / len(records)}
    summary = {"status": "ok", "config_hash": cfg_hash, "run_id": _rid(config), "split": split,
               "ledger_run_id": read.record["run_id"], "n_eval": len(records),
               "n_unparseable": sum(not r["parse_ok"] for r in records), **score,
               "predictions": pred_path.name, "adapter_sha256": adapter_hashes,
               "git_commit": read.record["git_commit"], "timestamp_utc": env.utc_now()}
    with open(paths.out_dir / "final_runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
    return summary


def validation_loader(config: Mapping[str, Any], train_loader: Optional[Callable] = None,
                      limit: Optional[int] = None) -> Tuple[Dict[str, Any], Callable[[], Dict[str, Any]]]:
    """Validation rows in the shape of a held-out loader, for the dry run."""
    is_sum = config["task"] == "billsum"
    texts, labels = (train_loader or (billsum.load_train_split if is_sum else data.load_train_split))()
    splits = billsum.make_splits(config["pool_seed"]) if is_sum else data.make_splits(len(texts), config["pool_seed"])
    idx = list(splits.val)[:limit] if limit else list(splits.val)
    t, g = [texts[i] for i in idx], [labels[i] for i in idx]
    rows = {"texts": t, ("summaries" if is_sum else "labels"): g, "indices": idx, "n": len(idx),
            "hash": data.rows_fingerprint(t, g)}
    return rows, lambda: rows


def dry_run(config: Mapping[str, Any], frozen_arms: Path, out_dir: Path, *, adapter_dir: Optional[Path] = None,
            predictor_factory: Optional[Callable] = None, train_loader: Optional[Callable] = None,
            limit: Optional[int] = None) -> Dict[str, Any]:
    """run_config inside a temporary repository with real gates and validation rows."""
    rows, loader = validation_loader(config, train_loader, limit)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)

        def git(*a: str) -> None:
            subprocess.run(["git", "-C", tmp, "-c", "user.name=dry", "-c", "user.email=dry@localhost",
                            "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *a],
                           check=True, capture_output=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        p = Paths(repo, repo / "final_configs.json", repo / "frozen_arms.json", repo / "test_reads.jsonl",
                  repo / "fingerprint.json", out_dir.resolve())
        shutil.copy(frozen_arms, p.frozen_arms)
        p.final_configs.write_text(json.dumps({
            "schema": 1, "frozen_arms_sha256": tuning.file_hash(str(p.frozen_arms)),
            "configs": {ledger.config_hash(dict(config)): {"task": config["task"], "arm_family": "A",
                                                           "config": dict(config)}}}), encoding="utf-8")
        p.fingerprint.write_text(json.dumps({config["task"]: {"sha256": rows["hash"], "n": rows["n"]}}),
                                 encoding="utf-8")
        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "dry run")
        for tag in ("prereg-v1", "arms-A"):
            git("tag", "-a", tag, "-m", "dry run")
        return run_config(config, p, adapter_dir=adapter_dir, predictor_factory=predictor_factory,
                          loader=loader, train_loader=train_loader, split=SPLIT_DRY)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = Paths.default()
    c = sub.add_parser("configs", help="write final_configs.json from frozen_arms.json (no data read)")
    c.add_argument("--frozen-arms", type=Path, default=d.frozen_arms)
    c.add_argument("--out", type=Path, default=d.final_configs)
    c.add_argument("--banking77-replicates", type=int, default=REPLICATES["banking77"])
    for name in ("run", "dry-run"):
        s = sub.add_parser(name)
        s.add_argument("--config-hash", required=True, help="full hash or unique prefix")
        s.add_argument("--final-configs", type=Path, default=d.final_configs)
        s.add_argument("--adapter-dir", type=Path, default=None)
    r = sub.choices["run"]
    r.add_argument("--rerun-reason", choices=ledger.INFRA_REASONS, default=None)
    r.add_argument("--rerun-detail", default=None)
    y = sub.choices["dry-run"]
    y.add_argument("--frozen-arms", type=Path, default=d.frozen_arms)
    y.add_argument("--out-dir", type=Path, required=True)
    y.add_argument("--limit", type=int, default=None, help="first k validation rows only")
    a = ap.parse_args(argv)
    if a.cmd == "configs":
        n = write_final_configs(a.frozen_arms, a.out, {"banking77": a.banking77_replicates})
        print(f"wrote {a.out}: {n} configurations (commit it before the arms tag)")
        return 0
    config = lookup_config(a.final_configs, a.config_hash)
    if a.cmd == "dry-run":
        res = dry_run(config, a.frozen_arms, a.out_dir, adapter_dir=a.adapter_dir, limit=a.limit)
    else:
        res = run_config(config, d, adapter_dir=a.adapter_dir, rerun_reason=a.rerun_reason,
                         rerun_detail=a.rerun_detail)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
