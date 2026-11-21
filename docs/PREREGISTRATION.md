# Pre-registration (DRAFT, not frozen)

Status: **draft written 2026-11-21, before any confirmatory run.** It must be frozen (committed with a tag, optionally also registered on OSF) before any test split is read. The BANKING77 pilot and label-budget curve in `results/pilot-2026-11-21/` are exploratory and are never pooled with confirmatory results. Sources for every choice: `docs/research/verification/00-summary.md` and reports 01–05.

## Question

Under what data, quality and cost conditions does a small open model adapted with LoRA beat commercial cheap-tier LLM APIs that receive in-context examples from the same labelled pool, and does the answer depend on the task family?

## Tasks and data (test splits frozen and hashed; each read once per final configuration)

| Task | Dataset | Notes | Status |
|---|---|---|---|
| Intent classification | BANKING77 (CC BY 4.0) | bridge to Bertsch et al. 2025 | decided |
| Summarisation | BillSum, **US bills only** (train / test 3,269; California split excluded) | CC0 per card; long inputs | decided; original licence still to be checked by a human |
| Legal classification | LEDGAR (LexGLUE version: 100 labels, single-label, 60k/10k/10k) | one fixed label list for all methods; pools must include rare labels | **conditional** on a human licence check; fallback CLINC150 plus |
| Excluded | SAMSum (reported NC-ND, unverified), Spider/BIRD, MBPP+/HumanEval+, CoNLL-2003, Broad Twitter, NER and finance sets | licences, leakage, power | decided |

## Systems

- **Small models:** Qwen3-4B-Instruct-2507, Phi-4-mini-instruct (OLMo-3-7B only after its DoRA failure is explained). Arms: zero-shot, k-shot random, k-shot retrieved, LoRA fine-tuning.
- **APIs:** `claude-haiku-5-5`, `gpt-6-luna` (reasoning off), optionally `gemini-3.1-flash-lite`. Arms: zero-shot, k-shot random, k-shot retrieved. No API fine-tuning (stated limitation). Temperature 0 where the API supports it; model ids, price date and run date logged; a 10% repeat measures self-agreement.
- **Baseline without an LLM:** kNN label vote over the retrieved shots.
- **Label budget:** pool of N labelled examples available to every arm: N ∈ {200, 1000, 5000} (N = 50 for in-context arms and BANKING77 only). k_max = 32 (classification), 5 (summarisation). Same task text, label list, output format, decoding and parser for every arm.

## Replicates and protocol

- 5 replicates = pool seed × training seed; each replicate draws a fresh nested pool shared by all methods. **Pending:** random vs stratified pool sampling.
- Training: steps = max(epochs·ceil(N/B), S_min ≈ 200); epochs chosen per (model, N) from {3, 5, 10, 20, 40} by validation loss with early stopping (patience 2); learning rate from one small grid per technique (rsLoRA grid shifted for α/√r); equal tuning effort for in-context arms (k ∈ {4, 8, 16, 32}). Validation split carved from train; never the test split.
- Everything else locked before the first run: optimiser, schedule, r, dropout, max length, batch size.

## Metrics

- Classification: accuracy (primary), macro-F1. Unparseable outputs are counted wrong; the unparseable rate is reported separately.
- Summarisation: ROUGE-L and BERTScore (`microsoft/deberta-xlarge-mnli`, rescaled); LLM judge only as a secondary check on a sample, dropped if Krippendorff α with the human labels is below 0.6.
- Cost per correct answer: provider-billed tokens (never character counts), price date recorded; self-hosted cost = measured wall-clock × an imputed H200 hourly rate shown as a band; all-in (training, evaluation, retries). Hardware: one H200 NVL; latency and energy from that card only.

## Confirmatory family (small on purpose)

Per task: best validated fine-tuned configuration versus best in-context configuration of the best-performing API, at two label budgets (N = 1,000 and N = 5,000), item as the unit, paired across arms on the same test items and replicates. McNemar (classification) or paired bootstrap over items (ROUGE-L); Holm over tasks × 2 contrasts; α = 0.05. Test sets of at least 1,500 items where available; BillSum test may be subsampled by a pre-fixed seed only if cost forces it.

## Exploratory (estimates and CIs, no p-values)

Technique ablation (QLoRA, DoRA, rsLoRA, each with its own learning rate) at N = 1,000 on BANKING77; the label-budget curve; break-even query volume against the hourly-rate band; per-item overlap with OLMo's open pretraining data, with a placebo stratification (random split); contamination remarks. Failed runs (such as the OLMo-3 DoRA NaN) are reported, not dropped.

## Pending decisions and human checks

Pool sampling (random vs stratified); whether to include Gemini; LEDGAR and BillSum licences against primary sources; X/Twitter terms are no longer relevant (no tweet data); whether CC BY-SA/NC terms of any dataset reach the published adapters; one live logprobs and temperature call per API; annotators for any judge check.
