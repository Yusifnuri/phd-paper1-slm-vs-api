"""Train and evaluate one (model, technique, N, seed, pool-seed) run; append one JSON line.

    python -m src.train.train --model qwen3-4b --technique lora --n 1000 --hub-repo user/adapters
    python -m src.train.train --model qwen3-4b --technique lora --n 200 --tune   # choose lr, epochs

Plain transformers + peft with a small hand-written loop (no Trainer), so time, tokens and
peak memory are measured exactly around the optimiser steps. Loss is the mean over
completion (label) tokens; weight decay 0.

Protocol (docs/research/verification/05-design-redteam.md):
  steps = max(epochs * ceil(N/B), S_min), S_min = 200; B = min(16, ceil(N/20)) so that S_min
  covers >= 10 epochs at small N. --tune picks lr from the technique's grid and epochs from
  {3,5,10,20,40} by validation loss (patience 2 over epochs); the chosen values are logged.
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
from typing import Any, Dict, List, Optional

from . import data, env, task_spec
from .config import EPOCH_GRID, PATIENCE, TrainConfig, config_from_args, lr_grid, run_id
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


def train_loop(model, examples: List[Dict[str, Any]], cfg: TrainConfig, pad_id: int) -> Dict[str, Any]:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    total = cfg.total_steps()
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(cfg.warmup_ratio * total), total)

    model.train()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # peak covers weights + training, not model loading
    t0 = time.perf_counter()
    step, tokens, loss_tokens, step_losses, epoch = 0, 0, 0, [], 0
    while step < total:  # cycle through epochs until the step budget is spent
        order = list(range(len(examples)))
        random.Random(cfg.seed * 1000 + epoch).shuffle(order)
        epoch += 1
        for s in range(0, len(order), cfg.batch):
            if step >= total:
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
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    tail = step_losses[-min(10, len(step_losses)):]
    return {
        "train_seconds": seconds, "steps": step, "epochs_effective": step * cfg.batch / len(examples),
        "trained_tokens": tokens, "loss_tokens": loss_tokens, "tokens_per_second": tokens / seconds,
        "final_train_loss": sum(tail) / len(tail),  # mean of the last <=10 optimiser steps
        "first_train_loss": step_losses[0],
        "peak_mem_bytes": torch.cuda.max_memory_allocated(),
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


def run(cfg: TrainConfig, prep=None, trial: bool = False,
        selection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One training run. trial=True: train and return validation loss only (used by tune)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    spec, texts, labels_all, splits, tok, pad_id = prep or _prepare(cfg)
    rid = run_id(spec.hf_id, cfg.technique, cfg.n, cfg.seed, cfg.pool_seed, cfg.sampling)
    print(f"[{'trial' if trial else 'run'}] {rid} lr={cfg.lr:g} epochs={cfg.epochs} "
          f"steps={cfg.total_steps()} batch={cfg.batch}", flush=True)

    pool = splits.pool(cfg.n)
    val = list(splits.val)[: cfg.eval_limit] if cfg.eval_limit else list(splits.val)
    examples = [data.tokenize_example(tok, texts[i], labels_all[i], cfg.max_length) for i in pool]
    val_examples = [data.tokenize_example(tok, texts[i], labels_all[i], cfg.max_length) for i in val]
    max_tokens = max(len(e["input_ids"]) for e in examples)

    env.set_all_seeds(cfg.seed)  # before the model is built
    t0 = time.perf_counter()
    model = load_model(spec, cfg)
    check_lora_targets(model, spec.target_modules)  # fail before training, not silently
    if cfg.technique == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(**peft_kwargs(
        cfg.technique, spec.target_modules, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = train_loop(model, examples, cfg, pad_id)
    vloss = val_loss(model, val_examples, pad_id)
    if trial:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {"val_loss": vloss, **stats}

    ckpt = Path(cfg.checkpoint_dir) / rid
    model.save_pretrained(ckpt)

    _, comp = data.render_pair(tok, "x", labels_all[0])
    stop_ids = stop_token_ids(tok, tok(comp, add_special_tokens=False)["input_ids"])
    ev = evaluate(
        model, tok, run_id=rid, system=spec.hf_id, technique=cfg.technique, n=cfg.n,
        seed=cfg.seed, items=[(i, texts[i], labels_all[i]) for i in val],
        stop_ids=stop_ids, pad_id=pad_id, batch_size=cfg.eval_batch,
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
        "epochs": cfg.epochs, "lr": cfg.lr, "effective_batch": cfg.batch,
        "micro_batch": cfg.micro, "s_min": cfg.s_min, "max_length": cfg.max_length,
        "selected_by_validation": selection is not None, "selection": selection,
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
    Path(cfg.results).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.results, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    if cfg.hub_repo and hub_pushed:
        try:
            push_file(cfg.hub_repo, cfg.results, f"logs/{Path(cfg.results).name}")
        except Exception as e:
            print(f"[hub] log upload failed: {safe_error(e)}", flush=True)
    print(f"[done] {rid}: train {record['train_seconds']:.1f}s, peak {record['peak_mem_gb']:.1f} GB, "
          f"acc {record['accuracy']:.4f}", flush=True)
    return record


def select_trials(trial_fn, technique: str, n: int, cfg: TrainConfig) -> Dict[str, Any]:
    """Pure selection logic (testable without a GPU).

    For each lr in the technique's grid, try epochs in EPOCH_GRID ascending, skip epoch values
    that give the same step count as one already tried (the S_min floor makes small epoch counts
    identical), and stop after PATIENCE consecutive non-improvements of validation loss.
    trial_fn(lr, epochs) -> validation loss. Returns the best (lr, epochs) and all trials.
    """
    trials, best = [], None
    for lr in lr_grid(technique):
        seen, bad, best_lr = set(), 0, math.inf
        for ep in EPOCH_GRID:
            steps = dataclasses.replace(cfg, epochs=ep).total_steps()
            if steps in seen:
                continue
            seen.add(steps)
            vl = trial_fn(lr, ep)
            if not math.isfinite(vl):
                vl = math.inf  # a diverged trial never wins
            trials.append({"lr": lr, "epochs": ep, "steps": steps, "val_loss": vl})
            if best is None or vl < best["val_loss"]:
                best = trials[-1]
            if vl < best_lr:
                best_lr, bad = vl, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best is None or not math.isfinite(best["val_loss"]):
        raise RuntimeError("no finite validation loss in any tuning trial")
    return {"lr": best["lr"], "epochs": best["epochs"], "val_loss": best["val_loss"], "trials": trials}


def tune(cfg: TrainConfig) -> Dict[str, Any]:
    prep = _prepare(cfg)

    def trial_fn(lr: float, ep: int) -> float:
        c = dataclasses.replace(cfg, lr=lr, epochs=ep, tune=False)
        return run(c, prep=prep, trial=True)["val_loss"]

    sel = select_trials(trial_fn, cfg.technique, cfg.n, cfg)
    print(f"[tune] chosen lr={sel['lr']:g} epochs={sel['epochs']} val_loss={sel['val_loss']:.4f}", flush=True)
    final = dataclasses.replace(cfg, lr=sel["lr"], epochs=sel["epochs"], tune=False)
    return run(final, prep=prep, selection=sel)


def main(argv: Optional[List[str]] = None) -> int:
    cfg = config_from_args(argv)
    tune(cfg) if cfg.tune else run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
