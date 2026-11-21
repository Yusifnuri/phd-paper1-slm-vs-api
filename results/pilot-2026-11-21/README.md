# Pilot and label-budget curve, 2026-11-21

Exploratory runs on BANKING77 intent classification. **Not confirmatory**: no pre-registration existed when they ran, and every accuracy is on a 1,000-item validation split carved from the training set (the test split was never read).

| File | Content |
|---|---|
| `pilot.jsonl` | 12 runs: Phi-4-mini-instruct, Qwen3-4B-Instruct-2507, OLMo-3-7B-Instruct × LoRA / QLoRA / DoRA / rsLoRA, N = 1000, seed 42 |
| `curve.jsonl` | 36 runs: the same three models, LoRA only, N ∈ {50, 200, 1000, 5000}, seeds 42, 43, 44 |
| `smoke.jsonl` | 3 smoke-test runs (N = 50, 10 steps), not results |
| `predictions/` | one JSONL file per run with the per-item predictions (`parse_ok` false = unparseable, scored as wrong) |

Produced by the code on branch `claude/pilot-train-226845` (commit `3a5c3a3`) on one NVIDIA H200 NVL (driver 580, torch 2.9.0+cu128, transformers 4.57.3, peft 0.18.0). Adapters are in the private Hugging Face repo `yusifnuri/paper1-pilot-adapters`.

Known problems, do not cite these numbers without reading them:

- **Undertraining at small N.** Three fixed epochs give about 9 optimizer steps at N = 50, and 74–95% of outputs are unparseable at N ≤ 200, so small-N accuracy measures undertraining, not label budget.
- **OLMo-3-7B DoRA is a pipeline failure** (`!!!!…` for every output, accuracy 0.000), not a result.
- Mistakes of the model itself dominate the unparseable outputs at N ≥ 1000: it invents plausible intent names outside the 77 labels. The parser is not at fault.
- The N = 1000, seed 42, LoRA run appears in both `pilot.jsonl` and `curve.jsonl`; the two share an adapter name on the Hub.
