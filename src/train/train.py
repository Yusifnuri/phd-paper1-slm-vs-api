"""Train and evaluate one (model, technique, N, seed, pool-seed) run; append one JSON line.

    python -m src.train.train --model qwen3-4b --technique lora --n 1000 --hub-repo user/adapters
    python -m src.train.train --model qwen3-4b --n 1000 --replicate 0 --tune --frozen-arms runs/frozen_arms.json
    python -m src.train.train --model qwen3-4b --n 1000 --replicate 1 --frozen-arms runs/frozen_arms.json

Plain transformers + peft with a small hand-written loop (no Trainer), so time, tokens and
peak memory are measured exactly around the optimiser steps. Loss is the mean over
completion (label) tokens; weight decay 0.

Protocol (docs/foundation/EXPERIMENT_PROTOCOL.md section 4; selection logic in tuning.py):
  B = min(16, ceil(N/20)); untuned runs train max(epochs * ceil(N/B), S_min) steps, S_min = 200.
  --tune (replicate 0 only): one trial per lr of the technique's grid, each a single cosine
  schedule to E_max epochs with validation at five checkpoints and early stopping (patience 2);
  the best checkpoint's (lr, schedule length, stop step) is frozen in frozen_arms.json.
  A final run with --frozen-arms trains that same schedule and stops at the frozen step.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import data, env, task_spec
from . import tuning
from .config import TrainConfig, config_from_args, lr_grid, run_id
from .evaluate import evaluate, stop_token_ids
from .hub import push_file, push_folder, safe_error
from .registry import ModelSpec, resolve
from .targets import check_lora_targets


def peft_kwargs(technique: str, targets, r: int, alpha: int, dropout: float) -> Dict[str, Any]:
    """Arguments for peft.LoraConfig. DoRA and rsLoRA are flags on LoRA; QLoRA is LoRA on a 4-bit base."""
    return {
        "r": r, "lora_alpha": alpha, "lora_dropout": dropout, "bias": "none",
        "target_modules": list(targets), "task_type": "CAUSAL_LM",
        "use_dora": technique == "dora", "use_rslora": technique == "rslora",
    }


def check_architecture(hf_id: str) -> None:
    """Fail early and clearly if the installed transformers does not know the architecture."""
    import transformers
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(hf_id)
    except (ValueError, KeyError) as e:
        if "model_type" in str(e) or "architecture" in str(e):
            raise RuntimeError(
                f"transformers {transformers.__version__} does not support the architecture of "
                f"{hf_id} ({str(e)[:150]}). Upgrade transformers (re-pin and re-test) or drop the model."
            ) from e
        raise


def load_model(spec: ModelSpec, cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kw: Dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if cfg.technique == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)


def _tensors(batch, device):
    import torch
    return (torch.tensor(batch["input_ids"], device=device),
            torch.tensor(batch["attention_mask"], device=device),
            torch.tensor(batch["labels"], device=device))


def val_loss(model, examples: List[Dict[str, Any]], pad_id: int, batch_size: int = 32) -> float:
    """Token-weighted mean loss over completion tokens of the validation examples."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(examples), batch_size):
            ids, mask, labels = _tensors(data.pad_batch(examples[s:s + batch_size], pad_id), device)
            n = int((labels[:, 1:] != -100).sum())
            total += float(model(input_ids=ids, attention_mask=mask, labels=labels).loss) * n
            count += n
    model.train()
    return total / count


def train_loop(model, examples: List[Dict[str, Any]], cfg: TrainConfig, pad_id: int,
               checkpoints: Sequence[int] = (), on_checkpoint=None) -> Dict[str, Any]:
    """Train cfg.steps_to_train() steps of a cosine schedule of cfg.schedule_length() steps.

    on_checkpoint(step) -> bool is called after each step in `checkpoints`; True stops training
    (in-run early stopping). Its wall-clock time is excluded from train_seconds and reported
    separately as checkpoint_seconds.
    """
    import torch
    from transformers import get_cosine_schedule_with_warmup

    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    total = cfg.schedule_length()
    stop_at = cfg.steps_to_train()
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(cfg.warmup_ratio * total), total)
    ck_set, ck_seconds, early_stopped = set(checkpoints), 0.0, False

    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # peak covers weights + training, not model loading
    t0 = time.perf_counter()
    step, tokens, loss_tokens, step_losses, epoch = 0, 0, 0, [], 0
    while step < stop_at and not early_stopped:  # cycle through epochs until the budget is spent
        order = list(range(len(examples)))
        random.Random(cfg.seed * 1000 + epoch).shuffle(order)
        epoch += 1
        for s in range(0, len(order), cfg.batch):
            if step >= stop_at:
                break
            batch_idx = order[s:s + cfg.batch]
            micro = [batch_idx[m:m + cfg.micro] for m in range(0, len(batch_idx), cfg.micro)]
            loss_sum = 0.0
            for mb in micro:
                ids, mask, labels = _tensors(data.pad_batch([examples[i] for i in mb], pad_id), device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                (loss / len(micro)).backward()
                loss_sum += loss.detach().float()
                tokens += int(mask.sum())
                loss_tokens += int((labels != -100).sum())
            step_loss = float(loss_sum) / len(micro)
            if not math.isfinite(step_loss):  # fail loudly: a NaN run must never look like a result
                raise FloatingPointError(
                    f"non-finite loss {step_loss} at optimiser step {step + 1} "
                    f"(technique={cfg.technique}); see scripts/diagnose_nan.py")
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step_losses.append(step_loss)
            step += 1
            if on_checkpoint is not None and step in ck_set:
                torch.cuda.synchronize()
                c0 = time.perf_counter()
                early_stopped = bool(on_checkpoint(step))
                ck_seconds += time.perf_counter() - c0
                model.train()
                if early_stopped:
                    break
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0 - ck_seconds
    tail = step_losses[-min(10, len(step_losses)):]
    return {
        "train_seconds": seconds, "steps": step, "epochs_effective": step * cfg.batch / len(examples),
        "trained_tokens": tokens, "loss_tokens": loss_tokens, "tokens_per_second": tokens / seconds,
        "final_train_loss": sum(tail) / len(tail),  # mean of the last <=10 optimiser steps
        "first_train_loss": step_losses[0],
        "peak_mem_bytes": torch.cuda.max_memory_allocated(),
        "schedule_steps": total, "checkpoint_seconds": ck_seconds, "early_stopped": early_stopped,
    }


def _prepare(cfg: TrainConfig):
    """Data and tokenizer shared by every trial of a run."""
    from transformers import AutoTokenizer

    spec = resolve(cfg.model)
    check_architecture(spec.hf_id)
    texts, labels_all = data.load_train_split()
    if len(texts) != data.N_TRAIN:
        raise RuntimeError(f"expected {data.N_TRAIN} train rows, got {len(texts)}")
    splits = data.make_splits(len(texts), cfg.pool_seed, sampling=cfg.sampling, labels=labels_all)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return spec, texts, labels_all, splits, tok, pad_id


def _build_model(spec: ModelSpec, cfg: TrainConfig):
    """Seeded base model plus a fresh adapter. Returns (model, load_seconds, n_trainable)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    env.set_all_seeds(cfg.seed)  # before the model is built
    t0 = time.perf_counter()
    model = load_model(spec, cfg)
    check_lora_targets(model, spec.target_modules)  # fail before training, not silently
    if cfg.technique == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(**peft_kwargs(
        cfg.technique, spec.target_modules, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)))
    torch.cuda.synchronize()
    return model, time.perf_counter() - t0, sum(p.numel() for p in model.parameters() if p.requires_grad)


def _release_gpu() -> None:
    """Call after the caller has deleted its last reference to the model."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _stop_ids(tok, labels_all: Sequence[str]) -> List[int]:
    _, comp = data.render_pair(tok, "x", labels_all[0])
    return stop_token_ids(tok, tok(comp, add_special_tokens=False)["input_ids"])


def _arm_key(cfg: TrainConfig, hf_id: str) -> str:
    return tuning.arm_key(task_spec.TASK, hf_id, cfg.technique, cfg.n, cfg.arm_sampling or cfg.sampling)


def apply_frozen_arm(cfg: TrainConfig, hf_id: str):
    """Final run: take lr, schedule length and stop step from frozen_arms.json.

    Returns (cfg, arm_info); arm_info is None when no frozen-arm file is given.
    """
    if not cfg.frozen_arms or cfg.tune:
        return cfg, None
    if cfg.pool_seed == tuning.TUNING_REPLICATE:
        raise ValueError("pool 0 is the tuning replicate and is never analysed; finals use --replicate >= 1")
    key = _arm_key(cfg, hf_id)
    arm = tuning.load_arm(cfg.frozen_arms, key)
    if arm["effective_batch"] != cfg.batch or arm["s_min"] != cfg.s_min:
        raise ValueError(f"frozen arm {key} was tuned with batch {arm['effective_batch']} / s_min "
                         f"{arm['s_min']}, this run has {cfg.batch} / {cfg.s_min}")
    new = dataclasses.replace(cfg, lr=arm["lr"], schedule_total=arm["schedule_steps"],
                              stop_step=arm["stop_step"])
    return new, {"arm_key": key, "frozen_arms_sha256": tuning.file_hash(cfg.frozen_arms), "arm": arm}


def run(cfg: TrainConfig, prep=None, selection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One final (or untuned) training run, evaluated on validation; appends one JSON line."""
    spec, texts, labels_all, splits, tok, pad_id = prep or _prepare(cfg)
    cfg, arm_info = apply_frozen_arm(cfg, spec.hf_id)
    rid = run_id(spec.hf_id, cfg.technique, cfg.n, cfg.seed, cfg.pool_seed, cfg.sampling)
    print(f"[run] {rid} lr={cfg.lr:g} schedule={cfg.schedule_length()} steps={cfg.steps_to_train()} "
          f"batch={cfg.batch} frozen={'yes' if arm_info else 'no'}", flush=True)

    pool = splits.pool(cfg.n)
    val = list(splits.val)[: cfg.eval_limit] if cfg.eval_limit else list(splits.val)
    examples = [data.tokenize_example(tok, texts[i], labels_all[i], cfg.max_length) for i in pool]
    val_examples = [data.tokenize_example(tok, texts[i], labels_all[i], cfg.max_length) for i in val]
    max_tokens = max(len(e["input_ids"]) for e in examples)

    model, load_seconds, n_trainable = _build_model(spec, cfg)
    stats = train_loop(model, examples, cfg, pad_id)
    vloss = val_loss(model, val_examples, pad_id)

    ckpt = Path(cfg.checkpoint_dir) / rid
    model.save_pretrained(ckpt)

    ev = evaluate(
        model, tok, run_id=rid, system=spec.hf_id, technique=cfg.technique, n=cfg.n,
        seed=cfg.seed, items=[(i, texts[i], labels_all[i]) for i in val],
        stop_ids=_stop_ids(tok, labels_all), pad_id=pad_id, batch_size=cfg.eval_batch,
        max_new_tokens=cfg.max_new_tokens,
    )
    pred_path = Path(cfg.predictions_dir) / f"{rid}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as fh:
        for r in ev["records"]:
            r["extra"].update({"pool_seed": cfg.pool_seed, "pool_hash": data.indices_hash(pool)})
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    hub_pushed, hub_error = False, None
    if cfg.hub_repo:
        try:
            shutil.copy(pred_path, ckpt / "predictions.jsonl")
            push_folder(cfg.hub_repo, str(ckpt), rid)
            hub_pushed = True
            shutil.rmtree(ckpt)  # only after a successful push
        except Exception as e:  # keep the adapter on disk if the push failed
            hub_error = safe_error(e)
            print(f"[hub] push failed, adapter kept at {ckpt}: {hub_error}", flush=True)
    else:
        print(f"[hub] no --hub-repo: adapter stays at {ckpt} (disk is ephemeral!)", flush=True)

    record = {
        "status": "ok", "run_id": rid, "model": spec.hf_id, "model_key": spec.key,
        "technique": cfg.technique, "n": cfg.n, "seed": cfg.seed, "task": task_spec.TASK,
        "replicate": cfg.replicate, "epochs": cfg.epochs, "lr": cfg.lr,
        "effective_batch": cfg.batch, "micro_batch": cfg.micro, "s_min": cfg.s_min,
        "schedule_total": cfg.schedule_length(), "stop_step": cfg.stop_step,
        "max_length": cfg.max_length, "frozen_arm": arm_info,
        "selected_by_validation": selection is not None or arm_info is not None, "selection": selection,
        "longest_example_tokens": max_tokens, "target_modules": list(spec.target_modules),
        "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
        "max_steps": cfg.max_steps, "trainable_params": n_trainable,
        "load_seconds": load_seconds, "val_loss": vloss,
        "peak_mem_gb": stats["peak_mem_bytes"] / 1e9, **stats,
        "accuracy": ev["accuracy"], "n_eval": ev["n_eval"], "n_unparseable": ev["n_unparseable"],
        "eval_seconds": ev["eval_seconds"], "eval_split": "validation",
        "prompt": "task_spec (system with 77 labels + 'Input: q\\nLabel:')",
        **data.split_summary(splits, cfg.n),
        **env.gpu_info(), "libs": env.lib_versions(), "git_commit": env.git_commit(),
        "timestamp_utc": env.utc_now(), "hub_repo": cfg.hub_repo, "hub_pushed": hub_pushed,
        "hub_error": hub_error,
    }
    _append(cfg.results, record)
    if cfg.hub_repo and hub_pushed:
        try:
            push_file(cfg.hub_repo, cfg.results, f"logs/{Path(cfg.results).name}")
        except Exception as e:
            print(f"[hub] log upload failed: {safe_error(e)}", flush=True)
    print(f"[done] {rid}: train {record['train_seconds']:.1f}s, peak {record['peak_mem_gb']:.1f} GB, "
          f"acc {record['accuracy']:.4f}", flush=True)
    del model
    _release_gpu()
    return record


def _append(path: str, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def tune(cfg: TrainConfig) -> Dict[str, Any]:
    """Replicate 0: one in-run early-stopped trial per lr; freeze the best checkpoint's procedure.

    Writes the arm to cfg.frozen_arms and one "tuned" line (all trials, GPU seconds) to cfg.results.
    No final run here: replicate 0 is never analysed.
    """
    if not cfg.frozen_arms:
        raise ValueError("--tune needs --frozen-arms (where the chosen arm is frozen)")
    spec, texts, labels_all, splits, tok, pad_id = _prepare(cfg)
    key = _arm_key(cfg, spec.hf_id)
    if key in tuning.load_arms(cfg.frozen_arms):
        raise tuning.FrozenArmConflict(f"{key} is already frozen in {cfg.frozen_arms}")

    pool = splits.pool(cfg.n)
    val = list(splits.val)[: tuning.val_size(cfg.n)]
    if cfg.eval_limit:
        val = val[: cfg.eval_limit]  # smoke mode only
    examples = [data.tokenize_example(tok, texts[i], labels_all[i], cfg.max_length) for i in pool]
    val_examples = [data.tokenize_example(tok, texts[i], labels_all[i], cfg.max_length) for i in val]
    items = [(i, texts[i], labels_all[i]) for i in val]
    stop_ids = _stop_ids(tok, labels_all)
    sched = cfg.max_steps or tuning.schedule_steps(cfg.n, cfg.batch, cfg.s_min)
    cks = tuning.checkpoint_steps(sched)
    grid = lr_grid(cfg.technique)
    rid = run_id(spec.hf_id, cfg.technique, cfg.n, cfg.seed, cfg.pool_seed, cfg.sampling) + "__tune"
    print(f"[tune] {rid} grid={list(grid)} schedule={sched} checkpoints={cks} val={len(val)}", flush=True)

    trials: List[Dict[str, Any]] = []
    for lr in grid:
        c = dataclasses.replace(cfg, lr=lr, schedule_total=sched, stop_step=None)
        model, load_s, _ = _build_model(spec, c)
        stopper, log = tuning.EarlyStopper("accuracy"), []

        def on_checkpoint(step: int) -> bool:
            ev = evaluate(model, tok, run_id=rid, system=spec.hf_id, technique=c.technique, n=c.n,
                          seed=c.seed, items=items, stop_ids=stop_ids, pad_id=pad_id,
                          batch_size=c.eval_batch, max_new_tokens=c.max_new_tokens)
            ck = {"step": step, "epoch": step / c.steps_per_epoch(), "accuracy": ev["accuracy"],
                  "val_loss": val_loss(model, val_examples, pad_id), "n_unparseable": ev["n_unparseable"]}
            log.append(ck)
            stop = stopper.update(ck)
            print(f"[tune] lr={lr:g} step={step} acc={ck['accuracy']:.4f} loss={ck['val_loss']:.4f}"
                  f"{' -> stop' if stop else ''}", flush=True)
            return stop

        try:
            stats, status = train_loop(model, examples, c, pad_id, cks, on_checkpoint), "ok"
        except FloatingPointError as e:  # a diverged trial is kept in the log and never wins
            stats, status = {"error": safe_error(e)}, "nonfinite"
        trials.append({
            "lr": lr, "schedule_steps": sched, "status": status, "checkpoints": log,
            "load_seconds": load_s,
            **{k: stats.get(k) for k in ("train_seconds", "checkpoint_seconds", "steps", "early_stopped",
                                         "first_train_loss", "final_train_loss", "peak_mem_bytes", "error")},
        })
        del model, on_checkpoint  # drop every reference before the next trial loads a model
        _release_gpu()

    sel = tuning.select(trials, grid, "accuracy", cfg.steps_per_epoch())
    entry = {
        **sel, "lr_grid": list(grid), "effective_batch": cfg.batch, "s_min": cfg.s_min,
        "tuning_replicate": tuning.TUNING_REPLICATE, "pool_seed": cfg.pool_seed, "train_seed": cfg.seed,
        "pool_hash": data.indices_hash(pool), "val_size": len(val), "val_hash": data.indices_hash(val),
        "git_commit": env.git_commit(), "timestamp_utc": env.utc_now(),
    }
    sha = tuning.freeze_arm(cfg.frozen_arms, key, entry)
    gpu_seconds = sum((t.get("train_seconds") or 0) + (t.get("checkpoint_seconds") or 0)
                      + (t.get("load_seconds") or 0) for t in trials)
    record = {
        "status": "tuned", "run_id": rid, "model": spec.hf_id, "model_key": spec.key,
        "technique": cfg.technique, "n": cfg.n, "seed": cfg.seed, "pool_seed": cfg.pool_seed,
        "sampling": cfg.sampling, "task": task_spec.TASK, "replicate": cfg.replicate,
        "arm_key": key, "frozen_arms_sha256": sha, "selection": entry, "trials": trials,
        "tuning_gpu_seconds": gpu_seconds, **env.gpu_info(), "libs": env.lib_versions(),
    }
    _append(cfg.results, record)
    print(f"[tune] frozen {key}: lr={sel['lr']:g} stop_step={sel['stop_step']} "
          f"(epochs* {sel['epochs_star']:.2f}) val_acc={sel['val_accuracy']:.4f} edge={sel['at_grid_edge']}",
          flush=True)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    cfg = config_from_args(argv)
    tune(cfg) if cfg.tune else run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
