# Step plan

Target: a journal submission about 8 weeks from 2026-11-21, on one university H200 and an API budget of $50–200. Facts behind each choice: [verified findings](research/2026-11-21-verified-findings.md).

## Design in one paragraph

Four small open models (Phi-4-mini-instruct, Qwen3-4B-Instruct-2507, OLMo-3-7B-Instruct, Gemma-4-E4B-it) are fine-tuned on N ∈ {50, 200, 1000, 5000} labelled examples. Three cheap APIs (Claude Haiku 5.5, GPT-6 Luna, Gemini Flash-Lite) answer the same test items with the k most similar examples retrieved from the same N-example pool, plus zero-shot and random-k controls.

There are five tasks. Every system is scored on identical items (at least 1,000 per task where the data allows), and every per-instance prediction is saved.

Hypotheses:

- **H1, crossover:** fine-tuning overtakes ICL as N grows.
- **H2, cost:** where it does, the self-hosted model wins on cost per correct answer under vLLM batching.
- **H3, capability redirection:** the fine-tuning gain is concentrated on items where the base model already shows latent knowledge.
- **H5, technique:** QLoRA, DoRA and rsLoRA are equivalent to LoRA within ±1 point.
- **H6, format:** fine-tuned models comply with the output format more often.
- **H4, contamination:** exploratory only.

## Step 0: decisions and pre-registration (days 1–2, blocks everything)

**Decisions:**

| Decision | Options |
|---|---|
| NER dataset | Broad Twitter Corpus (Twitter terms caveat) or Few-NERD (share-alike licence) |
| Financial dataset | SEntFiN (news headlines, needs reformulating) or Twitter Financial News (tweets) |
| Code task | Replace with text-to-SQL (Spider, CC BY-SA, licence not yet verified), replace with contract classification (LEDGAR), or keep code with Self-OSS-Instruct training and our own decontamination |
| API arm | Confirm the three models above, or swap Gemini 3.1 Flash-Lite for the cheaper 2.5 Flash-Lite |
| Annotators | Two annotators for the 200-item human check of the summarisation judge |

**Then:**

- Write the hypotheses, primary metrics, tests and exclusion rules into `docs/PREREGISTRATION.md`.
- Commit it before any result exists. Optionally also register it on OSF.

**Done when:** the pre-registration is committed and pushed.

## Step 1: data pipeline (week 1, parallel with Steps 2 and 4)

**Build:**

- Loaders for the five datasets with fixed, seeded splits.
- Nested training budgets: 50 ⊂ 200 ⊂ 1000 ⊂ 5000, so each smaller pool is part of the next.
- One frozen test set per task.
- A data card per dataset recording licence, source, sizes and caveats.
- A decontamination check of each test set against its training pool (n-gram plus embedding similarity).

**Done when:** loaders are unit-tested, split hashes are recorded, and the data cards are written.

## Step 2: evaluation harness (week 1, parallel)

**Metrics, ported from the thesis repo where possible:**

- Accuracy and macro-F1.
- Typed NER micro-F1: `ner_typed.py` plus seqeval.
- ROUGE-L and BERTScore (deberta-xlarge-mnli).
- Execution accuracy, if text-to-SQL is chosen.
- Format-compliance rate.
- ECE for the fine-tuned arm.

**Also:**

- Per-instance prediction files for every system.
- Paired statistics:
  - McNemar for accuracy-type tasks.
  - Hierarchical paired bootstrap over items and seeds.
  - Holm correction within each task.

**Done when:** all of it is unit-tested on synthetic data.

## Step 3: training harness and pilot (week 2)

**Training harness:**

- PEFT variants: LoRA, QLoRA, DoRA, rsLoRA, and full fine-tuning for 4B models.
- One config per model, with the LoRA-target check from the thesis repo.
- Seeds, and MLflow logging of the GPU model, driver and hourly rate per run.

**Pilot on the H200:**

- One task × four models × all techniques × N = 1000 × one seed.
- Measures real hours, which settles DoRA's overhead, and catches failures.

**Done when:** the pilot completes, and the GPU-hour estimate for the full grid is recomputed from the measured times.

## Step 4: API harness (weeks 1–2, parallel)

**Build:**

- An embedding retriever for k-shot (k = 32 for short tasks, 5 for long ones).
- Zero-shot and random-k controls.
- Pinned model ids and run dates.
- Logging of provider-billed usage.
- Use of the batch APIs.
- A cost estimator that must run before any spend.

**Then:** a 50-item pilot per API.

**Done when:** the pilot is logged, and the cost estimate for the full run is under budget.

## Step 5: main runs (weeks 3–5)

**GPU, in this order:**

1. Core label-budget curve with LoRA: 4 models × 4 N × 5 tasks × 3 seeds.
2. Technique ablation at N = 1000.
3. Full-fine-tuning reference for the 4B models.

**API, in parallel:** the full API arm through the batch APIs.

**Done when:** every cell has per-instance predictions for 3 seeds, and the aggregation reproduces from MLflow.

## Step 6: serving economics (week 5)

**Measure with vLLM on the same H200 type:**

- Concurrency levels 1, 8, 32 and 128.
- TTFT, p50/p95 latency and throughput.
- Energy (CodeCarbon plus NVML).

**Serving rules:** LoRA and rsLoRA adapters are served directly; DoRA is merged first.

**Compute:**

- Cost per correct answer.
- Break-even volume against each API, across a band of H200 hourly rates.

**Done when:** H2 is tested.

## Step 7: capability redirection and contamination (week 6)

- **H3:** stratify test items by the base model's latent knowledge (the gold answer is in its top-k outputs, or its log-probability is above the median). Then compare fine-tuning gains with a permutation test.
- **Contamination (exploratory):** OlmoTrace for OLMo-3, and post-cutoff LiveCodeBench items if code is kept.

## Step 8: judge validation (week 6, parallel with Step 7)

- **Human check:** two annotators rate 200 stratified summaries.
- **Agreement:** Krippendorff's α between the annotators, and Spearman ρ between the judge and the humans.
- **Judge scope:** apply the judge to a sampled subset only, to stay in budget.

## Step 9: analysis and writing (weeks 7–8)

- Figures and tables.
- Threats to validity.
- Choice of venue.
- Draft, internal review, submission.

## What can run at the same time

| Weeks | Parallel work |
|---|---|
| 1–2 | Steps 1, 2 and 4 |
| 3–5 | Step 5's GPU and API work |
| 5 | Step 6 starts as soon as the core LoRA adapters exist |
| 6 | Steps 7 and 8 |
