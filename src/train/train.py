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
from . import billsum, summarise, tuning
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


def _mod(cfg: TrainConfig):
    """Task data module: BANKING77 (`data`) or BillSum (`billsum`); both expose tokenize_example."""
    return billsum if cfg.task == "billsum" else data


def _prepare(cfg: TrainConfig):
    """Data and tokenizer shared by every trial of a run. `labels_all` holds the gold label
    (classification) or the gold summary (BillSum) of each train row."""
    from transformers import AutoTokenizer

    spec = resolve(cfg.model)
    check_architecture(spec.hf_id)
    if cfg.task == "billsum":
        texts, labels_all = billsum.load_train_split()
        if len(texts) != billsum.N_TRAIN:
            raise RuntimeError(f"expected {billsum.N_TRAIN} train rows, got {len(texts)}")
        if cfg.sampling != "random":
            raise ValueError("BillSum has no classes: only random pools exist")
        splits = billsum.make_splits(cfg.pool_seed)
    else:
        texts, labels_all = data.load_train_split()
        if len(texts) != data.N_TRAIN:
            raise RuntimeError(f"expected {data.N_TRAIN} train rows, got {len(texts)}")
        splits = data.make_splits(len(texts), cfg.pool_seed, sampling=cfg.sampling, labels=labels_all)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return spec, texts, labels_all, splits, tok, pad_id


def _tokenize(cfg: TrainConfig, tok, texts, targets, indices, *, train: bool):
    """(examples, dropped_indices). Over-long items raise, except BillSum train items above the
    cap, which are dropped (and returned for logging) only if under 0.1% of the pool."""
    mod = _mod(cfg)
    if train and cfg.task == "billsum":
        full = {i: mod.tokenize_example(tok, texts[i], targets[i], 10 ** 9) for i in indices}
        kept, dropped = mod.select_trainable(list(indices), {i: len(e["input_ids"]) for i, e in full.items()},
                                             cfg.max_length)
        return [full[i] for i in kept], dropped
    return [mod.tokenize_example(tok, texts[i], targets[i], cfg.max_length) for i in indices], []


def _build_model(spec: ModelSpec, cfg: TrainConfig):
    """Seeded base model plus a fresh adapter. Returns (model, load_seconds, n_trainable)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    env.set_all_seeds(cfg.seed)  # before the model is built
    t0 = time.perf_counter()
    model = load_model(spec, cfg)
    check_lora_targets(model, spec.target_modules)  # fail before training, not silently
    if cfg.use_grad_checkpointing:  # long BillSum inputs: recompute activations (protocol stage 4)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if cfg.technique == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(**peft_kwargs(
        cfg.technique, spec.target_modules, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)))
    torch.cuda.synchronize()
    return model, time.perf_counter() - t0, sum(p.numel() for p in model.parameters() if p.requires_grad)


def _snapshot_adapter(model) -> Dict[str, Any]:
    """CPU copy of the adapter weights (summarisation tuning keeps the best-loss checkpoint)."""
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def _restore_adapter(model, state: Dict[str, Any]) -> None:
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state)


def _release_gpu() -> None:
    """Call after the caller has deleted its last reference to the model."""
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _stop_ids(tok, labels_all: Sequence[str]) -> List[int]:
    _, comp = data.render_pair(tok, "x", labels_all[0])
    return stop_token_ids(tok, tok(comp, add_special_tokens=False)["input_ids"])


def _arm_key(cfg: TrainConfig, hf_id: str) -> str:
    return tuning.arm_key(cfg.task, hf_id, cfg.technique, cfg.n, cfg.arm_sampling or cfg.sampling)


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
    """One final (or untuned) training run, evaluated on validation; appends one JSON line.

    BANKING77: generation on the validation rows, accuracy. BillSum: validation loss, plus (only with
    cfg.gen_eval) greedy summary generation on the first cfg.gen_limit validation bills and ROUGE-L
    with length statistics (src/train/summarise.py); without it generation_eval=False.
    """
    spec, texts, labels_all, splits, tok, pad_id = prep or _prepare(cfg)
    cfg, arm_info = apply_frozen_arm(cfg, spec.hf_id)
    is_sum = cfg.task == "billsum"
    rid = run_id(spec.hf_id, cfg.technique, cfg.n, cfg.seed, cfg.pool_seed, cfg.sampling, cfg.task)
    print(f"[run] {rid} lr={cfg.lr:g} schedule={cfg.schedule_length()} steps={cfg.steps_to_train()} "
          f"batch={cfg.batch} micro={cfg.micro} frozen={'yes' if arm_info else 'no'}", flush=True)

    pool = splits.pool(cfg.n)
    val = list(splits.val)[: cfg.eval_limit] if cfg.eval_limit else list(splits.val)
    examples, dropped = _tokenize(cfg, tok, texts, labels_all, pool, train=True)
    val_examples, _ = _tokenize(cfg, tok, texts, labels_all, val, train=False)
    max_tokens = max(len(e["input_ids"]) for e in examples)

    model, load_seconds, n_trainable = _build_model(spec, cfg)
    stats = train_loop(model, examples, cfg, pad_id)
    vloss = val_loss(model, val_examples, pad_id, batch_size=cfg.micro if is_sum else 32)

    ckpt = Path(cfg.checkpoint_dir) / rid
    model.save_pretrained(ckpt)
    # Binds the saved adapter to its frozen arm and pool; src/final/run_final.py checks it.
    (ckpt / "run_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (ckpt / "run_manifest.json").write_text(json.dumps({
        "run_id": rid, "model": spec.hf_id, "task": cfg.task, "technique": cfg.technique, "n": cfg.n,
        "replicate": cfg.replicate, "pool_hash": data.indices_hash(pool),
        "arm_key": arm_info["arm_key"] if arm_info else None, "arm": arm_info["arm"] if arm_info else None,
        "stop_step": cfg.stop_step, "git_commit": env.git_commit(),
    }, sort_keys=True), encoding="utf-8")

    ev = None
    if not is_sum:
        ev = evaluate(
            model, tok, run_id=rid, system=spec.hf_id, technique=cfg.technique, n=cfg.n,
            seed=cfg.seed, items=[(i, texts[i], labels_all[i]) for i in val],
            stop_ids=_stop_ids(tok, labels_all), pad_id=pad_id, batch_size=cfg.eval_batch,
            max_new_tokens=cfg.max_new_tokens,
        )
    sev = None
    if is_sum and cfg.gen_eval:
        gen_val = val[: cfg.gen_limit] if cfg.gen_limit else val
        sev = summarise.evaluate_summaries(
            model, tok, run_id=rid, system=spec.hf_id, technique=cfg.technique, n=cfg.n, seed=cfg.seed,
            items=[(i, texts[i], labels_all[i]) for i in gen_val],
            stop_ids=summarise.summary_stop_ids(tok, labels_all), pad_id=pad_id,
            batch_size=cfg.gen_batch, max_new_tokens=billsum.MAX_NEW_TOKENS,
        )
        ev = sev  # same prediction-file and hub path as BANKING77 below
    pred_path = Path(cfg.predictions_dir) / f"{rid}.jsonl"
    if ev is not None:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w", encoding="utf-8") as fh:
            for r in ev["records"]:
                r["extra"].update({"pool_seed": cfg.pool_seed, "pool_hash": data.indices_hash(pool)})
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    hub_pushed, hub_error = False, None
    if cfg.hub_repo:
        try:
            if ev is not None:
                shutil.copy(pred_path, ckpt / "predictions.jsonl")
            push_folder(cfg.hub_repo, str(ckpt), rid)
            hub_pushed = True
            shutil.rmtree(ckpt)  # only after a successful push
        except Exception as e:  # keep the adapter on disk if the push failed
            hub_error = safe_error(e)
            print(f"[hub] push failed, adapter kept at {ckpt}: {hub_error}", flush=True)
    else:
        print(f"[hub] no --hub-repo: adapter stays at {ckpt} (disk is ephemeral!)", flush=True)

    summ = billsum.split_summary(splits, cfg.n) if is_sum else data.split_summary(splits, cfg.n)
    record = {
        "status": "ok", "run_id": rid, "model": spec.hf_id, "model_key": spec.key,
        "technique": cfg.technique, "n": cfg.n, "seed": cfg.seed, "task": cfg.task,
        "replicate": cfg.replicate, "epochs": cfg.epochs, "lr": cfg.lr,
        "effective_batch": cfg.batch, "micro_batch": cfg.micro, "s_min": cfg.s_min,
        "grad_checkpointing": cfg.use_grad_checkpointing,
        "schedule_total": cfg.schedule_length(), "stop_step": cfg.stop_step,
        "max_length": cfg.max_length, "frozen_arm": arm_info,
        "selected_by_validation": selection is not None or arm_info is not None, "selection": selection,
        "longest_example_tokens": max_tokens, "dropped_over_cap": dropped, "target_modules": list(spec.target_modules),
        "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
        "max_steps": cfg.max_steps, "trainable_params": n_trainable,
        "load_seconds": load_seconds, "val_loss": vloss,
        "peak_mem_gb": stats["peak_mem_bytes"] / 1e9, **stats,
        "accuracy": ev["accuracy"] if ev and not sev else None, "n_eval": ev["n_eval"] if ev else len(val),
        "n_unparseable": ev["n_unparseable"] if ev else None,
        "eval_seconds": ev["eval_seconds"] if ev else None, "eval_split": "validation",
        "generation_eval": ev is not None,
        **({"summary_eval": {k: v for k, v in sev.items() if k != "records"}, "gen_batch": cfg.gen_batch,
            "gen_limit": cfg.gen_limit, "max_new_tokens": billsum.MAX_NEW_TOKENS} if sev else {}),
        "prompt": ("billsum task_spec" if is_sum else "task_spec (system with 77 labels + 'Input: q\\nLabel:')"),
        **summ,
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
    acc = (f"rougeL {sev['rouge_l']:.4f} (n={sev['n_eval']})" if sev
           else f"acc {record['accuracy']:.4f}" if ev else f"val_loss {vloss:.4f}")
    print(f"[done] {rid}: train {record['train_seconds']:.1f}s, peak {record['peak_mem_gb']:.1f} GB, {acc}", flush=True)
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

    is_sum = cfg.task == "billsum"
    # summarisation: loss stops a trial, ROUGE-L of generated summaries picks the winner (L-42)
    metric, stop_metric = ("rouge_l", "loss") if is_sum else ("accuracy", "accuracy")
    pool = splits.pool(cfg.n)
    val = list(splits.val)[: tuning.val_size(cfg.n, cfg.val_cap)]
    if cfg.eval_limit:
        val = val[: cfg.eval_limit]  # smoke mode only
    examples, dropped = _tokenize(cfg, tok, texts, labels_all, pool, train=True)
    val_examples, _ = _tokenize(cfg, tok, texts, labels_all, val, train=False)
    items = [] if is_sum else [(i, texts[i], labels_all[i]) for i in val]
    stop_ids = [] if is_sum else _stop_ids(tok, labels_all)
    vbatch = cfg.micro if is_sum else 32
    sched = cfg.max_steps or tuning.schedule_steps(cfg.n, cfg.batch, cfg.s_min)
    cks = tuning.checkpoint_steps(sched, early=tuning.EARLY_CHECKPOINT_FRACTION if is_sum else None)
    gen_val = val[: cfg.gen_limit] if is_sum and cfg.gen_limit else val  # gen_limit: smoke mode
    sum_stop_ids = summarise.summary_stop_ids(tok, labels_all) if is_sum else []
    grid = lr_grid(cfg.technique)
    rid = run_id(spec.hf_id, cfg.technique, cfg.n, cfg.seed, cfg.pool_seed, cfg.sampling, cfg.task) + "__tune"
    print(f"[tune] {rid} metric={metric} grid={list(grid)} schedule={sched} checkpoints={cks} val={len(val)}", flush=True)

    trials: List[Dict[str, Any]] = []
    for lr in grid:
        c = dataclasses.replace(cfg, lr=lr, schedule_total=sched, stop_step=None)
        model, load_s, _ = _build_model(spec, c)
        stopper, log = tuning.EarlyStopper(stop_metric), []
        best: Dict[str, Any] = {}  # summarisation: best-loss checkpoint of this trial and its weights

        def on_checkpoint(step: int) -> bool:
            ck = {"step": step, "epoch": step / c.steps_per_epoch(), "accuracy": None, "n_unparseable": None,
                  "val_loss": val_loss(model, val_examples, pad_id, batch_size=vbatch)}
            if not is_sum:
                ev = evaluate(model, tok, run_id=rid, system=spec.hf_id, technique=c.technique, n=c.n,
                              seed=c.seed, items=items, stop_ids=stop_ids, pad_id=pad_id,
                              batch_size=c.eval_batch, max_new_tokens=c.max_new_tokens)
                ck.update(accuracy=ev["accuracy"], n_unparseable=ev["n_unparseable"])
            log.append(ck)
            if is_sum and (not best or tuning.score(ck, "loss") > tuning.score(best["ck"], "loss")):
                best.update(ck=ck, state=_snapshot_adapter(model))
            stop = stopper.update(ck)
            shown = f"acc={ck['accuracy']:.4f} " if ck["accuracy"] is not None else ""
            print(f"[tune] lr={lr:g} step={step} {shown}loss={ck['val_loss']:.4f}"
                  f"{' -> stop' if stop else ''}", flush=True)
            return stop

        try:
            stats, status = train_loop(model, examples, c, pad_id, cks, on_checkpoint), "ok"
        except FloatingPointError as e:  # a diverged trial is kept in the log and never wins
            stats, status = {"error": safe_error(e)}, "nonfinite"
        gen_seconds = 0.0
        if is_sum and status == "ok" and best:
            _restore_adapter(model, best["state"])
            sev = summarise.evaluate_summaries(
                model, tok, run_id=rid, system=spec.hf_id, technique=c.technique, n=c.n, seed=c.seed,
                items=[(i, texts[i], labels_all[i]) for i in gen_val], stop_ids=sum_stop_ids,
                pad_id=pad_id, batch_size=cfg.gen_batch, max_new_tokens=billsum.MAX_NEW_TOKENS)
            best["ck"].update(rouge_l=sev["rouge_l"], n_unparseable=sev["n_unparseable"])
            gen_seconds = sev["eval_seconds"]
            print(f"[tune] lr={lr:g} step={best['ck']['step']} rougeL={sev['rouge_l']:.4f}", flush=True)
        trials.append({
            "lr": lr, "generation_seconds": gen_seconds, "schedule_steps": sched, "status": status, "checkpoints": log,
            "load_seconds": load_s,
            **{k: stats.get(k) for k in ("train_seconds", "checkpoint_seconds", "steps", "early_stopped",
                                         "first_train_loss", "final_train_loss", "peak_mem_bytes", "error")},
        })
        del model, on_checkpoint  # drop every reference before the next trial loads a model
        _release_gpu()

    sel = tuning.select(trials, grid, metric, cfg.steps_per_epoch())
    entry = {
        **sel, "lr_grid": list(grid), "effective_batch": cfg.batch, "s_min": cfg.s_min,
        "tuning_replicate": tuning.TUNING_REPLICATE, "pool_seed": cfg.pool_seed, "train_seed": cfg.seed,
        "pool_hash": data.indices_hash(pool), "val_size": len(val), "val_hash": data.indices_hash(val),
        "task": cfg.task, "gen_size": len(gen_val) if is_sum else None, "dropped_over_cap": dropped, "max_length": cfg.max_length,
        "git_commit": env.git_commit(), "timestamp_utc": env.utc_now(),
    }
    sha = tuning.freeze_arm(cfg.frozen_arms, key, entry)
    gpu_seconds = sum((t.get("train_seconds") or 0) + (t.get("checkpoint_seconds") or 0)
                      + (t.get("load_seconds") or 0) + (t.get("generation_seconds") or 0) for t in trials)
    record = {
        "status": "tuned", "run_id": rid, "model": spec.hf_id, "model_key": spec.key,
        "technique": cfg.technique, "n": cfg.n, "seed": cfg.seed, "pool_seed": cfg.pool_seed,
        "sampling": cfg.sampling, "task": cfg.task, "replicate": cfg.replicate,
        "arm_key": key, "frozen_arms_sha256": sha, "selection": entry, "trials": trials,
        "tuning_gpu_seconds": gpu_seconds, **env.gpu_info(), "libs": env.lib_versions(),
    }
    _append(cfg.results, record)
    score = (f"val_rouge_l={sel['val_rouge_l']:.4f}" if is_sum
             else f"val_acc={sel['val_accuracy']:.4f}")
    print(f"[tune] frozen {key}: lr={sel['lr']:g} stop_step={sel['stop_step']} "
          f"(epochs* {sel['epochs_star']:.2f}) {score} edge={sel['at_grid_edge']}", flush=True)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    cfg = config_from_args(argv)
    tune(cfg) if cfg.tune else run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
