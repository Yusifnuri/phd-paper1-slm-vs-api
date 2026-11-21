"""Train and evaluate one (model, technique, N, seed) run; append one JSON line of measurements.

    python -m src.train.train --model qwen3-4b --technique lora --n 1000 --hub-repo user/adapters

Plain transformers + peft with a small hand-written loop: no Trainer, so timing, token
counts and peak memory are measured exactly around the optimiser steps and nothing else.
Loss is the mean over completion (label) tokens; weight decay is 0 (the Trainer default).
Written without GPU access to test against: see the README section in the pilot report.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import data, env
from .config import TASK, TrainConfig, config_from_args, run_id
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
    step, tokens, loss_tokens, step_losses = 0, 0, 0, []
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        random.Random(cfg.seed * 1000 + epoch).shuffle(order)
        for s in range(0, len(order), cfg.effective_batch):
            if step >= total:
                break
            batch_idx = order[s:s + cfg.effective_batch]
            loss_sum = 0.0
            micro = [batch_idx[m:m + cfg.micro_batch] for m in range(0, len(batch_idx), cfg.micro_batch)]
            for mb in micro:
                b = data.pad_batch([examples[i] for i in mb], pad_id)
                ids = torch.tensor(b["input_ids"], device=device)
                mask = torch.tensor(b["attention_mask"], device=device)
                labels = torch.tensor(b["labels"], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                (loss / len(micro)).backward()
                loss_sum += loss.detach().float()
                tokens += int(mask.sum())
                loss_tokens += int((labels != -100).sum())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step_losses.append(float(loss_sum) / len(micro))
            step += 1
        if step >= total:
            break
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    tail = step_losses[-min(10, len(step_losses)):]
    return {
        "train_seconds": seconds, "steps": step, "trained_tokens": tokens,
        "loss_tokens": loss_tokens, "tokens_per_second": tokens / seconds,
        "final_train_loss": sum(tail) / len(tail),  # mean of the last <=10 optimiser steps
        "first_train_loss": step_losses[0],
        "peak_mem_bytes": torch.cuda.max_memory_allocated(),
    }


def run(cfg: TrainConfig) -> Dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    spec = resolve(cfg.model)
    rid = run_id(spec.hf_id, cfg.technique, cfg.n, cfg.seed)
    print(f"[run] {rid}", flush=True)

    # Data first (cheap, fails early); never touches the test split.
    texts, labels_all = data.load_train_split()
    splits = data.make_splits(len(texts), cfg.data_seed)
    pool = splits.pool(cfg.n)
    label_set = set(labels_all)
    val = list(splits.val)[: cfg.eval_limit] if cfg.eval_limit else list(splits.val)

    env.set_all_seeds(cfg.seed)  # before the model is built
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    examples = [data.tokenize_example(tok, texts[i], labels_all[i], cfg.max_length) for i in pool]
    max_tokens = max(len(e["input_ids"]) for e in examples)

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

    ckpt = Path(cfg.checkpoint_dir) / rid
    model.save_pretrained(ckpt)

    # Completion ids of one example give the end-of-turn token(s) for generation stopping.
    _, comp = data.render_pair(tok, "x", labels_all[0])
    stop_ids = stop_token_ids(tok, tok(comp, add_special_tokens=False)["input_ids"])
    ev = evaluate(
        model, tok, run_id=rid, system=spec.hf_id, technique=cfg.technique, n=cfg.n,
        seed=cfg.seed, items=[(i, texts[i], labels_all[i]) for i in val], labels=label_set,
        stop_ids=stop_ids, pad_id=pad_id, batch_size=cfg.eval_batch,
        max_new_tokens=cfg.max_new_tokens,
    )
    pred_path = Path(cfg.predictions_dir) / f"{rid}.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as fh:
        for r in ev["records"]:
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
        "technique": cfg.technique, "n": cfg.n, "seed": cfg.seed, "task": TASK,
        "epochs": cfg.epochs, "lr": cfg.lr, "effective_batch": cfg.effective_batch,
        "micro_batch": cfg.micro_batch, "max_length": cfg.max_length,
        "longest_example_tokens": max_tokens, "target_modules": list(spec.target_modules),
        "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
        "max_steps": cfg.max_steps, "steps": stats["steps"], "trainable_params": n_trainable,
        "train_seconds": stats["train_seconds"], "load_seconds": load_seconds,
        "peak_mem_bytes": stats["peak_mem_bytes"],
        "peak_mem_gb": stats["peak_mem_bytes"] / 1e9,
        "trained_tokens": stats["trained_tokens"], "loss_tokens": stats["loss_tokens"],
        "tokens_per_second": stats["tokens_per_second"],
        "final_train_loss": stats["final_train_loss"], "first_train_loss": stats["first_train_loss"],
        "accuracy": ev["accuracy"], "n_eval": ev["n_eval"], "n_unparseable": ev["n_unparseable"],
        "eval_seconds": ev["eval_seconds"], "eval_split": "validation",
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


def main(argv: Optional[List[str]] = None) -> int:
    run(config_from_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
