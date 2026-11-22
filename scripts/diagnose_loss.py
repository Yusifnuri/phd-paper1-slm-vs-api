#!/usr/bin/env python3
"""GPU diagnostic for two pilot anomalies (run on the server; needs the model weights):

  1. Phi-4-mini first_train_loss 18.88 (LoRA/DoRA/rsLoRA) vs 8.50 (QLoRA), above ln(vocab)=12.2.
  2. OLMo-3 DoRA first_train_loss NaN at step 1.

At step 1 the adapter is a no-op (LoRA B=0), so the loss is the BASE model's loss on the label
tokens. This prints, for 16 training examples: per-position token and loss (to see whether a few
tokens such as <|end|>/<|endoftext|> dominate the mean), the mean in bf16 autocast / pure fp32 /
4-bit, and, on NaN, the first module whose output is non-finite plus any zero-norm weight rows
(DoRA divides by the per-row weight norm).

    python scripts/diagnose_loss.py --model phi4-mini --technique lora
    python scripts/diagnose_loss.py --model phi4-mini --technique qlora
    python scripts/diagnose_loss.py --model olmo3-7b --technique dora
    python scripts/diagnose_loss.py --model phi4-mini --technique lora --fp32   # bf16 vs fp32
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from src.train import data
    from src.train.config import TECHNIQUES
    from src.train.registry import resolve
    from src.train.train import peft_kwargs

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--technique", choices=TECHNIQUES, default="lora")
    ap.add_argument("--fp32", action="store_true", help="load the base model in float32 (no autocast)")
    ap.add_argument("--examples", type=int, default=16)
    a = ap.parse_args()

    spec = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    texts, labels = data.load_train_split()
    pool = data.make_splits(len(texts)).pool(a.examples)
    exs = [data.tokenize_example(tok, texts[i], labels[i], 4096) for i in pool]

    kw = {"dtype": torch.float32 if a.fp32 else torch.bfloat16, "device_map": {"": 0}}
    if a.technique == "qlora":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)
    if a.technique == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(**peft_kwargs(a.technique, spec.target_modules, 16, 32, 0.05)))
    model.eval()

    bad = []
    def hook(name):
        def f(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o) and not bad and not torch.isfinite(o).all():
                bad.append(name)
        return f
    for n, m in model.named_modules():
        m.register_forward_hook(hook(n))

    b = data.pad_batch(exs, pad)
    dev = next(model.parameters()).device
    ids, mask, lab = (torch.tensor(b[k], device=dev) for k in ("input_ids", "attention_mask", "labels"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not a.fp32):
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
    print(f"ln(vocab) = {torch.log(torch.tensor(float(logits.shape[-1]))):.2f}")
    print("first non-finite module:", bad[0] if bad else None)
    tl = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), lab[:, 1:],
                                           ignore_index=-100, reduction="none")
    m = lab[:, 1:] != -100
    print(f"mean loss over label tokens: {tl[m].mean():.3f}  (max token loss {tl[m].max():.1f})")
    for i in range(min(3, len(exs))):
        pos = m[i].nonzero().flatten().tolist()
        print(" ", [(tok.decode([int(lab[i, p + 1])]), round(float(tl[i, p]), 1)) for p in pos])
    last = [float(tl[i][m[i]][-1]) for i in range(len(exs))]
    print(f"mean loss on the LAST label token (end-of-turn): {sum(last) / len(last):.2f}")
    for n, mod in model.named_modules():  # zero-norm rows break DoRA (magnitude / ||W_row||)
        if hasattr(mod, "base_layer") and hasattr(mod.base_layer, "weight") and a.technique == "dora":
            w = mod.base_layer.weight
            if w.dtype in (torch.float16, torch.bfloat16, torch.float32):
                z = int((w.float().norm(dim=1) == 0).sum()) + int((~torch.isfinite(w.float())).any(dim=1).sum())
                if z:
                    print(f"  {n}: {z} zero-norm or non-finite weight rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
