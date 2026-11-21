# Verified findings, 2026-11-21

Four research agents (datasets, methodology, compute, replacement task) produced claims with sources. An independent verifier, with no access to the researchers' reasoning, opened each source. Every fact below is marked with its verdict:

- **[V]** confirmed by the verifier.
- **[V~]** confirmed with a correction, given in the text.
- **[U]** not yet verified. This covers everything from the replacement-task search, which finished after verification.

Prices and model ids change; re-read them on the day of each run.

## Datasets

Requirements: commercial-use licence that also covers derived adapters, at least 5,000 training examples (for the N = 5,000 budget level), and at least 1,000 test items for paired tests.

| Task | Dataset | Train / test | Licence | Status |
|---|---|---|---|---|
| Intent classification | PolyAI/banking77 | 10,003 / 3,080 | CC BY 4.0 [V] | Recommended primary |
| Intent classification | clinc/clinc_oos (plus) | 15,250 / 5,500 | CC BY 3.0 [V] | Fallback |
| NER | GateNLP/broad_twitter_corpus | 5,342 / 2,002 | CC BY 4.0 on the HF card; the paper says the tweet text follows Twitter's 2015 guidelines [V~] | Usable as a secondary set, with a terms-of-service caveat |
| NER | DFKI-SLT/few-nerd (supervised) | 131,767 / 37,648 | CC BY-SA 4.0, share-alike [V] | Legal question: does share-alike reach derived adapters? |
| NER | leondz/wnut_17 | 3,394 train [V] | CC BY 4.0 | Excluded: below 5,000 |
| Summarisation | FiscalNote/billsum | 18,949 / 3,269 | CC0-1.0 [V] | Recommended. Long inputs: 10,268 characters on average, about 2.6k tokens |
| Summarisation | EdinburghNLP/xsum | — | "unknown" [V] | Excluded as a primary set |
| Summarisation | DialogSum, MeetingBank | — | CC BY-NC-SA 4.0 [V] | Excluded: non-commercial |
| Financial sentiment | SEntFiN (pyRis/SEntFiN) | 10,753 headlines, no official split | MIT on the repository [V]; the headlines come from news publishers, and MIT may not cover them | Recommended primary. Entity-level labels, so the task needs reformulating |
| Financial sentiment | zeroshot/twitter-financial-news-sentiment | 9,543 / 2,388 (validation used as test) | MIT on the card; tweets collected via the Twitter API [V] | Secondary, terms-of-service caveat |
| Financial sentiment | Financial PhraseBank | — | CC BY-NC-SA 3.0 [V] | Excluded: non-commercial (the thesis adapters derive from it) |
| Financial sentiment | FiQA-2018 task 1 | 961 train [V] | no licence stated | Excluded |
| Code | evalplus MBPP+ / HumanEval+ | test only: 378 / 164 [V] | Apache-2.0 | Underpowered: detects only about 7 points (HumanEval+) or 4.5 points (MBPP+) |
| Code | bigcode/self-oss-instruct-sc2-exec-filter-50k | 50,661 train | ODC-BY [V] | Possible training set; it needs our own decontamination against the test sets |
| Text-to-SQL (replacement for code) | xlangai/spider | 7,000 / 1,034 dev (test held back) | CC BY-SA 4.0, share-alike [U] | Candidate |
| Contract classification (fallback) | LexGLUE LEDGAR | 60k / 10k / 10k | CC BY 4.0 on the card [U] | Fallback; a second classification task |

Code leakage and contamination [V]:

- 108 of the 378 MBPP+ task ids are in MBPP's train split.
- OLMo-3's post-training data contains semantic duplicates of all MBPP items (arXiv 2602.12413).

So code results involving OLMo-3, or any training data derived from MBPP, are invalid as designed.

Twitter terms [V~]: X's developer agreement (since June 2025) forbids using X content to train foundation or frontier models. A small task adapter trained on third-party-hosted data is arguably outside that clause, so the legal risk is moderate to low. The risk with reviewers and enterprise readers is real. Do not redistribute tweet text, and state the caveat.

## Models and training cutoffs

| Model | Licence | Training cutoff [V] | Note |
|---|---|---|---|
| microsoft/Phi-4-mini-instruct | MIT | June 2024 | Fused `qkv_proj`; LoRA must target it |
| Qwen/Qwen3-4B-Instruct-2507 | Apache-2.0 | not disclosed | Contamination tests are not possible for it |
| allenai/Olmo-3-7B-Instruct | Apache-2.0 | December 2024 | Open data (Dolma 3); OlmoTrace exact-match tracing [V] |
| google/gemma-4-E4B-it | Apache-2.0 | January 2025 | vLLM serves LoRA for the text-only class only |

## Techniques and serving

- **PEFT variants available:** LoRA, QLoRA, DoRA (`use_dora`) and rsLoRA (`use_rslora`; scales by α/√r) [V].
- **DoRA cannot be served as an adapter by vLLM** (it is rejected). PiSSA, OLoRA and LoftQ adapters were *silently mis-served*. rsLoRA is served correctly. [V~]
  - Consequence: serving experiments use LoRA or rsLoRA, and DoRA is merged into the base model before serving.
- **DoRA's training overhead is unknown.** The published +139% time figure is an inference benchmark, not a training one [V~]. Measure it in the pilot.
- **Full fine-tuning memory:** AdamW mixed precision needs about 16 bytes per parameter [V].
  - A 4B model needs about 64 GB plus activations and fits on one H200.
  - An 8B model needs about 128 GB before activations and does not fit without 8-bit optimisers.

## Compute

- **H200:** 141 GB HBM3e, 4.8 TB/s [V]. Confirm whether the university card is SXM or NVL.
- **Demand:** about 125 GPU-hours for the pruned grid and about 280 for the full grid. The verifier independently estimated about 200. All three numbers are estimates. BillSum's long inputs make summarisation about 6 times slower than the thesis's 512-token runs. Either way the work fits easily on one H200 in 2 months.
- **H200 on-demand prices** [V]:
  - JarvisLabs $4.59 (spot $2.29).
  - Hyperbolic $3.99.
  - RunPod $5.29 (Secure) / $3.59 (Community).
  - Nebius $5.40.
  - Vast.ai median $4.44 (live API).
- **Hourly rate for the paper:** the thesis imputed $3.99 citing JarvisLabs, which now charges $4.59. Report a band, not a single rate.
- **One hardware type for all measurements:** latency, throughput and energy must all come from one H200 type. The thesis's NER re-run on an RTX A6000 broke this.

## API arm

| Model | Price per 1M tokens, input / output [V] | Notes |
|---|---|---|
| Claude Haiku 5.5 | $0.10 / $0.50 (batch 50% off) | New tokenizer gives about 30% more tokens; no logprobs |
| OpenAI GPT-6 Luna | $0.10 / $0.50 (batch 50% off) | Reasoning model: set effort to none, or reasoning tokens are billed as output |
| Gemini 3.1 Flash-Lite | $0.25 / $1.50 | No logprobs for 3.x models; 2.5 Flash-Lite ($0.10 / $0.40) is an alternative |

- **Budget:** the core grid costs about $51 at standard prices, or about $26 through the batch APIs [V~]. Summarisation input is about 17k tokens for 5 shots, not 3.8k.
- **Judge cost:** an LLM judge over every summary would cost about $250 on a mid-tier model. Judge a sampled subset only.
- **Calibration:** logprob-based ECE is impossible for at least two of the three APIs [V]. Report ECE for the fine-tuned arm, plus verbalised confidence for the APIs.

## Methodology

**Prior work** [V]:

- Few-shot fine-tuning versus ICL on the same model has been studied (Mosbach et al., Findings ACL 2023).
- Fine-tuning needs more data than long-context ICL but can overtake it (Bertsch et al., NAACL 2025).
- Many-shot ICL keeps improving (NeurIPS 2024).
- LoRA Land reports fine-tuned models beating GPT-4.
- On knowledge:
  - The Superficial Alignment Hypothesis (LIMA).
  - New-knowledge examples are learned slowly and raise hallucination (EMNLP 2024).
  - LoRA learns less and forgets less (TMLR).

**What appears new** (based on a search finding nothing, which is not proof of absence): small open models with several PEFT variants against *current* cheap APIs, compared across label budgets, together with cost per correct answer and break-even volume. And a test of capability redirection per test item in task fine-tuning.

**Power** [V]:

- McNemar with a discordant rate of 0.10, a 3-point difference and power 0.8 needs n ≈ 870. With Holm correction over about 10 comparisons, n ≈ 1,480.
- These test sets reach that: BANKING77, BillSum, BTC, the Twitter financial set and CLINC.

**Contamination hypothesis:**

- There is no LiveCodeBench release after April 2025 [V]. The post-cutoff items are about 443 for Phi-4-mini and about 175–200 for OLMo-3 and Gemma.
- It cannot be tested for Qwen3 or for any API.
- So contamination stays exploratory, not confirmatory.
