# A5 draft: FEASIBILITY_AND_BUDGET (compute, API, cost)

Date: 2026-11-21. Author: agent A5. Nothing here was run on a GPU or paid API. Labels: **[V]** verified from an official page today, **[M]** measured by me today, **[I]** interpretation, **[R]** recommendation, **[U]** unverified assumption. Provider pages were read through fetch sub-agents (page text is model-relayed; long pages were truncated, so each row carries a confidence). No page shows a publication date unless stated; "date" = 2026-11-21 access.

## 1. API cost

### 1.1 Verified prices and limits (USD per 1M tokens, standard, text)

| Item | Haiku 5.5 | GPT-6 Luna | Gemini 3.1 Flash-Lite |
|---|---|---|---|
| id [V] | `claude-haiku-5-5`; no dated id, but "Every Claude model ID is a pinned snapshot, including the dateless IDs" (overview) | `gpt-6-luna`; "Default snapshot: `gpt-6-luna`", no dated snapshot | `gemini-3.1-flash-lite`, "Stable"; shutdown "earliest possible" 2027-05-07 |
| input / output [V] | 0.10 / 0.50 (prompt ≤100k); 0.50 / 2.50 above | 0.10 / 0.50 (≤272k); 0.20 / 0.75 above | 0.25 / 1.50 (**summary-derived**, no tiers) |
| cached read / write [V] | 0.01 / 0.125 (5 min), 0.20 (1 h); min 512 tokens | 0.01 / 0.125 (30 min); min 1,024 stated for "GPT-5.6 and later", Luna not named [I] | 0.025, storage 1.00/1M/h; implicit min for this model NOT FOUND |
| batch [V] | 50%: 0.05 / 0.25; stacks with caching | 50%: 0.05 / 0.25 | 50% ("at 50% of the standard cost"), 0.125 / 0.75 summary-derived; target 24 h |
| reasoning billed as output? [V] | yes ("billed as output tokens"). Off: `thinking: {type: "disabled"}` (effort ≤ `high`); default adaptive **on**, effort `medium`; `output_config.effort` | yes ("billed as output tokens"). Off: `reasoning.effort = "none"`; default `medium` | yes ("sum of output tokens and thinking tokens"). **Cannot be fully off**: lowest is `thinking_level: minimal`, default on |
| temperature [V] | **not accepted**: "Omit `temperature`, `top_p`, and `top_k`"; migration page: sending them "returns a 400 error" | not stated for Luna (NOT FOUND) | not stated (NOT FOUND) |
| tokenizer [V] | "same newer tokenizer as Claude 4.7"; "approximately 30 percent more tokens than on Claude Haiku 4.5"; free `/v1/messages/count_tokens` ("an estimate") | not stated; no public GPT-6 encoding found | "about 4 characters" per token; `countTokens` API |
| context / max out [V] | 1M / 128k (batch up to 300k with beta header) | 1.05M (922k input) / 128k | 1,048,576 / 65,536 |
| lowest-tier limits [V] | Start: 4,000 RPM, 2M ITPM (uncached only), 400k OTPM; Batch: 100k requests or 256 MB per batch, 200,000 requests in queue; monthly cap $500 | Build ($5 credit): 5,000 RPM, 2M TPM, batch queue **20M enqueued tokens** | Tier 1 batch enqueued **10M tokens**, 100 concurrent batches; per-model RPM/TPM only inside AI Studio (NOT FOUND) |

Sources (all accessed 2026-11-21): platform.claude.com/docs/en/about-claude/pricing, …/api/rate-limits, …/build-with-claude/{batch-processing,prompt-caching,token-counting,thinking}, …/models/haiku-5-5/overview (dated "Released October 7, 2026"); developers.openai.com/api/docs/{models/gpt-6-luna,pricing,guides/rate-limits,guides/batch,guides/reasoning,guides/prompt-caching}; aistudio.google.com/docs/{pricing,models/gemini-3.1-flash-lite,rate-limits,batch-api,thinking,caching,tokens,deprecations} (ai.google.dev redirects here). Confidence: high for Anthropic and OpenAI price rows (read verbatim); **medium** for Gemini price rows (taken from a model summary of a truncated pricing page, must be re-read in a browser); medium for OpenAI reasoning/caching (long pages truncated).

Changes against earlier reports: (i) the Haiku temperature question is now **settled [V]**: no temperature, so `TEMPERATURE = 0.0` in `task_spec.py` cannot apply to Haiku (the call would return 400); determinism is impossible by design. (ii) Haiku thinking can be switched off, Gemini's cannot. (iii) Gemini 2.5 Flash-Lite is dropped from the plan (earlier report: access limited to prior users).

### 1.2 Measured token counts [M]

Samples: 1,000 rows per dataset from the HF datasets-server `rows` API (train splits, blocks at random offsets, not independent draws, so the sample is clustered; test splits may differ). Tokenizers: Qwen3-4B-Instruct-2507, Phi-4-mini-instruct, OLMo-3-7B-Instruct `tokenizer.json` (public) and tiktoken `o200k_base`. Phi-4-mini and o200k gave identical counts. **No public Claude or Gemini tokenizer exists**, so those need a band [I].

| Item (1,000 rows) | chars mean (p95) | Qwen tokens mean / p50 / p95 / max | Phi = o200k mean | chars/token |
|---|---|---|---|---|
| BANKING77 query | 54 (130) | 12.8 / 10 / 30 / 67 | 12.5 | 4.3 |
| 77-label system prompt (1 string) | 1,941 | 476 | 490 | 4.0 |
| BANKING77 zero-shot prompt (system + user) | 2,010 | **494** (p95 511) | 508 | 4.0 |
| BillSum bill text (US, train) | 10,298 (18,123) | **2,111** / 1,942 / 3,673 / 5,578 | 2,058 | 4.9 |
| BillSum reference summary | 1,193 (2,583) | **229** / 204 / 512 / 916 | 224 | 5.2 |
| LEDGAR provision (LexGLUE, 100 labels, 96 seen) | 684 (1,854) | **136** / 98 / 381 / 741 | 134 | 5.0 |
| LEDGAR label list (100 labels) | 1,441 | 413 | n/a | 3.5 |

Derived prompt sizes (Qwen-token basis, +≈15 chat-template tokens): BANKING77 zero-shot ≈ 510; each shot "Input/Label" = 22 tokens, so k=32 adds ≈ 710 → **≈ 1,220**. BillSum zero-shot ≈ 2,250 (instruction + text); each shot (text + summary) = 2,344, so k=3 → **≈ 9,300**, k=5 → **≈ 13,900**. LEDGAR zero-shot ≈ 620; shot = 144, so k=8 → ≈ 1,770. Billing multiplier relative to the Qwen count: Haiku 1.0–1.35 [U] (the 30% rise is vs Haiku 4.5, absolute unknown; run the count endpoint on 100 prompts before spending); Luna 0.95–1.1 [U]; Gemini 0.95–1.1 [U].

Correction [M]: BillSum inputs average 2.1k tokens, not 17k. The 17k figure only arises with ≈5 retrieved shots; zero-shot BillSum is cheap, retrieval BillSum is the cost driver. Because the 490–660-token Claude-counted label list sits near Haiku's 512-token cache minimum, even the shared prefix may or may not cache [I].

### 1.3 Cost per request and per scenario

Per request (standard): BANKING77 k=32, Haiku/Luna $0.000126–0.000169, Gemini $0.00032–0.0005 (the high end assumes 100 hidden thinking tokens [U]); BillSum k=3 Haiku/Luna ≈ $0.0011–0.0014, Gemini ≈ $0.0025–0.0035. Request counts per scenario are in §3. Caching: per-item retrieved shots do not cache; only the label list prefix could (B77 best case ≈ −35% on Haiku; nothing on Luna because 490 < 1,024; Gemini unknown). Batch ×0.5. Batch+cache stacking is documented for Anthropic only. [R] Use sync for BillSum on Luna and Gemini: batch queue limits are 20M / 10M tokens, i.e. ≈ 1,400–2,000 BillSum-k=3 requests per wave, and 24 h turnaround makes waves slow.

## 2. Training and inference time

### 2.1 Basis from the pilot [M]

Pilot rows (80-token prompt, batch 16): Phi 7.6k tok/s, Qwen 5.4k, OLMo 6.2k at N=5000. Converting at 4·params FLOPs per real token (frozen-weight LoRA, forward + activation backward) these are **116, 87 and 182 TFLOP/s** effective, so already above the "latency-bound" label for OLMo [I]. Peak memory 14–35 GB at sequences ≤152 tokens.

Assumption [U], all ranges: effective **100–250 TFLOP/s** short sequences, 150–250 with gradient checkpointing (GC, 6·params, +10% attention, ×1.2 padding) for BillSum. H200 NVL bf16 dense peak ≈ 835 TFLOP/s [U, from spec memory], so this is 12–30% MFU. Resulting real-token throughput (slow–fast): B77/LEDGAR LoRA: Phi 6.6–13k, Qwen 6.2–12k, OLMo 3.4–6.8k tok/s; BillSum with GC: Phi 5.0–8.3k, Qwen 4.7–7.9k, OLMo 2.6–4.3k.

Tokens per training example: B77 ≈ 515 (490 system + 12 query + ≈ 10 label/template); BillSum ≈ 2,400 mean, p95 ≈ 4,300, max ≈ 6,700; LEDGAR ≈ 600.

Optimiser samples per run (from `config.py`: S_MIN = 200 steps, batch 3 at N=50, 10 at N=200, 16 at N≥320): N=50 600–1,000; N=200 2,000–3,000; N=1000 3,200–5,000; N=5000 15,000–25,000 (3–5 epochs, early stopping) [I].

### 2.2 Time and memory per run (LoRA, one replicate, fixed config)

| Cell | Phi | Qwen3-4B | OLMo-3-7B |
|---|---|---|---|
| B77, N=1000 | 2–6.5 min | 2.3–7 min | 4–12.6 min |
| B77, N=5000 | 10–33 min | 11–35 min | 19–63 min |
| BillSum, N=1000 | 15–40 min | 16–43 min | 30–77 min |
| BillSum, N=5000 | **1.2–3.3 h** | **1.3–3.5 h** | **2.3–6.4 h** |

Multipliers: DoRA +31% (Phi), +42% (Qwen) time and +19% memory (pilot, measured); QLoRA +35–45%; grid tuning multiplies N≤1000 cells by (R + g − 1)/R. Per-run overhead 10 s load plus eval below.

Peak memory [I, ±40%, activations 34·S·B·h per layer, fp32 logits ×3.5 for logits, bf16 copy and grad]: B77 micro-batch 16 × ≈ 560 padded tokens: Phi ≈ 45–70 GB, Qwen ≈ 40–65 GB, OLMo ≈ 45–85 GB (card has 141 GB), so B77 and LEDGAR fit without GC. **BillSum does not**: Phi's 200k vocabulary gives ≈ 3.4 GB of fp32 logits per 4.3k-token sequence ×3.5 ≈ 12 GB, plus ≈ 13 GB of activations without GC. Micro-batch 16 OOMs for every model; 4 OOMs for Phi at the 6.7k maximum; [R] micro-batch 1–2 with GC, accumulation to 16, bucket by length, compute the LM head only on completion positions.

### 2.3 Inference engine

vLLM 0.13.0 lists LoRA ✅ for `Phi3ForCausalLM`, `Qwen3ForCausalLM`, `OLMo3ForCausalLM` (docs table only, from verification 04; not re-tested), rejects DoRA (source-confirmed), and mis-serves PiSSA/OLoRA (issue 60223, fix after 0.13.0). So LoRA/rsLoRA adapters serve directly; DoRA must be merged [R]. Estimates [U]: vLLM evaluation of 1,000 items takes 20–60 s (B77, prefix cache on), 30–90 s (LEDGAR), 90–150 s (BillSum, 4B; OLMo ×1.8). HF `generate` at batch 64 needs roughly 3–5× that (BillSum decode of 250 tokens is the slow part) [U]. [R] Use one engine (vLLM) for every arm so numerics are comparable, and load all adapters of one base model in a single server session (startup ≈ 2–4 min [U]). Zero-shot/few-shot/retrieval cells for open models cost 18–60 s per 1,000 items on B77, 3–6 min on BillSum (k=3, 9.3M prefill tokens) and 5–10 min at k=5. Context: retrieval k=5 reaches ≈ 38k tokens at the p95+ tail; Phi (128k) and Qwen3-2507 are safe, OLMo-3's limit must be checked [U].

## 3. Three scenarios

Allowance: 200 credits ÷ 15 = **13.3 GPU-h/week**; 8 Sunday resets between 2026-11-22 and 2027-01-10 ≈ 107 h gross. Carry-over is not documented [U]. The first 1–2 weeks go to the re-pilot, loss anomalies (I-14) and BillSum memory; the server also bills idle time. [I] Plan on **60–75 productive GPU-h** in two months. Ranges below come from the slow/fast throughput band in §2 (not statistical intervals); they exclude a ~15% rerun margin. R = fresh pool × seed; "tuning g" = learning rates tried at N≤1000 on one replicate, none at N=5000 (reuse N=1000's).

| | (a) Minimum viable | (b) Recommended | (c) Extended |
|---|---|---|---|
| Tasks | BANKING77, BillSum | BANKING77, BillSum | + LEDGAR (conditional on A2/A3) |
| Open models | Qwen3-4B, Phi-4-mini | Qwen, Phi on both; OLMo-3-7B on B77 only | all three on all tasks |
| Arms (open) | LoRA, zero-shot, retrieval | LoRA, zero-shot, random-k, retrieval | same + QLoRA/rsLoRA ablation at N=1000 (not costed) |
| N (B77 / BillSum) | 200, 1000, 5000 / same | 50, 200, 1000, 5000 / 200, 1000, 5000 | 4 levels all tasks, BillSum k=5 |
| Fine-tune replicates | R=5 (N≤1000), 3 (N=5000) | R=5, BillSum N=5000 R=3 | R=5 everywhere |
| Tuning g | 2 | 3 | 4 |
| Test items | 1,000 B77 / 500 BillSum | same | 1,000 each |
| Fine-tune runs (incl. tuning) | 60 | 112 | 237 |
| API arms | zero-shot + retrieval; 3 pools | zero-shot ×2 reps, random-k + retrieval; B77 5 pools, BillSum 3 | same, 5 pools, 1,000 items |
| API requests (per provider) | 15,000 | 52,000 | 116,000 |
| API cost, standard (all three) | **$27–34** | **$66–86** | **$257–322** |
| API cost, batch (50%) | $13–17 | $33–43 | $128–161 |
| API best-case caching | ≈ −10% [U] | ≈ −10% [U] | ≈ −10% [U] |
| GPU-hours | **16–42 h** | **17–47 h** | **65–172 h** |
| Weeks at 13.3 h | 1.2–3.2 | 1.3–3.5 | 4.9–13 |
| Rental, $3.6–6.0/h (SXM/HGX) | $56–252 | $60–280 | $234–1,030 |
| Engineering effort [U] | ≈ 1.5 weeks | ≈ 2.5 weeks | ≈ 4.5 weeks |

GPU-hours here were computed for (b) as Qwen+Phi on both tasks (13–34.5 h) plus OLMo on B77 (3.8–12.2 h); full OLMo on BillSum alone would add 18.6–48.9 h in (b)'s grid.

**Plain verdict.** (a) and (b) fit two months and the $50–200 budget: (b) with API $33–86 and GPU 17–47 h of ≈ 65 available (upper end leaves only a thin margin; stop BillSum N=5000 to R=2 first). (c) fits neither: GPU 65–172 h exceeds 8 weeks even at the fast end after setup, and its API standard bill ($257–322) exceeds $200 unless run in batch ($128–161), which BillSum queue limits make slow. [R] Choose (b); treat LEDGAR as a post-hoc extension only if A2/A3 clear it and (b) finishes by week 5.

Risks and contingencies: (1) throughput below the slow band → drop BillSum N=5000 or R to 3, then drop OLMo; (2) BillSum OOM → micro-batch 1 + GC (+33% time, already in the band); (3) `train.tune` absent, so tuning is untested; fall back to a fixed lr plus early stopping, and state it; (4) Gemini overspend through hidden thinking → log `thought_tokens`, cap by pilot of 50 items; (5) vLLM LoRA load failure on OLMo-3 → HF generate for that model only, flagged as an engine difference; (6) weekly credits lost to idle → script the queue and stop the server automatically; (7) Haiku/Luna model drift mid-campaign → run all API cells for one task within one 48-hour window and log `model_id_returned`.

## 4. Cost per correct prediction and per successful summary

Let r = imputed GPU rate ($/h, band **3.6–6.0**: RunPod Community $3.59 and Secure $5.29, DigitalOcean $4.47 HGX, Modal ≈ $4.54 SXM, Jarvislabs $4.59 SXM, Hyperbolic "from" $3.99, Nebius $5.40, Together $5.99 HGX; all SXM/HGX; **no NVL price exists on any official page read**, CoreWeave/Lambda/Vast gave no single-GPU H200 price; the university's credits are 15/h with an unknown credit-to-dollar value, so r is imputed [V/I]).

Self-hosted, for a deployment of V items:
`C_correct = [ r·(H_tune + H_final + H_eval)  +  V·r·h_inf·(1+ρ) ] / ( V·a )`
where H_tune = all tuning runs, H_final = final training for the reported config, H_eval = evaluation passes (all arms), h_inf = GPU-hours per served item, ρ = retry/unparseable-regeneration rate, a = accuracy (or success rate s for summaries). API:
`C_correct = [ (T_in·p_in + T_cached·p_cr + T_write·p_cw + (T_out + T_think)·p_out)·(1+ρ) ] / a`, per request, T = billed tokens (provider-reported, not estimated), batch factor 0.5 where used.

Illustration [I, not a result]: BANKING77, Qwen3-4B N=5000, accuracy assumed 0.90 for both arms purely for arithmetic. Training 0.18–0.58 h plus two extra learning-rate runs at N=1000 (0.08 h–0.23 h) = 0.26–0.81 h → **$0.92–4.85**. Inference 20–60 s per 1,000 items at saturation → $0.00002–0.0001 per item. Haiku retrieval request $0.000126–0.000169 per item. Break-even deployment volume V\* = training $ ÷ (API $ − self-hosted $ per item) = **≈ 6k items (best case) to ≈ 186k (worst case)**, and never if self-hosted per-item cost ≥ API's. At a = 0.90 and ρ = 0, API cost per correct prediction = $0.00014–0.00019. The self-hosted figure assumes the GPU is busy 100% of the time; at utilisation u, h_inf scales by 1/u [U]. BillSum, Qwen N=1000: training 3 runs = 0.81–2.13 h ($2.9–12.8), inference $0.00009–0.00025 per summary, vs Haiku zero-shot $0.00035–0.00043 and k=3 retrieval $0.0011–0.0014. With an assumed success rate s=0.5, a successful summary costs $0.0007–0.0009 (Haiku zero-shot) vs $0.0021–0.0028 (k=3). Success definition (ROUGE threshold or judge pass) is A4's decision.

## 5. Reproducibility limits [R unless noted]

- Haiku 5.5 has no sampling control, so repeated calls differ [V]; Luna and Gemini may too. Pin the id and `anthropic-version` header, store `model_id_returned`, repeat a 10% random subset once, report flip rate. "Every Claude model ID is a pinned snapshot" is a provider claim; it cannot be independently verified [I]. Luna's only snapshot is the undated default.
- Luna: set `reasoning.effort=none` and log reasoning tokens (should be 0). Gemini: minimal thinking cannot be zero; log `thought_tokens`.
- Open models: transformers 4.57.3, peft 0.18.0, trl 0.26.2, torch 2.9.0+cu128, vLLM 0.13.0 (server's actual versions). The pinned `requirements/train.txt` on the pilot branch had newer versions that were never installed [V, HANDOFF]. Record `pip freeze`, adapter hashes, seed, GPU and driver per run; GPU kernels are nondeterministic, so replicate variance must absorb it. A single engine (vLLM) for evaluation.

## 6. New issues (proposed)

| ID | Issue | Sev | Proposed action |
|---|---|---|---|
| A5-01 | `max_length` defaults to 1024; BillSum examples average ≈ 2.4k tokens (p95 4.3k) so training would silently truncate | HIGH | set ≥ 6,800 or truncate documents by documented rule; test |
| A5-02 | Tuning protocol (4 lr × epochs 3–40, patience 2) is referenced as `train.tune` but no such module exists on the branch; grid at N=5000 BillSum is infeasible | HIGH | implement, limit grid to N≤1000, report |
| A5-03 | Haiku rejects `temperature`; `task_spec.TEMPERATURE = 0.0` "for every arm" would 400 | HIGH | omit parameter for Haiku, document, replicate |
| A5-04 | BillSum micro-batch 16 will OOM; GC and micro-batch 1–2 needed | HIGH | measure on first GPU session |
| A5-05 | BillSum retrieval ICL dominates API cost (13.9k-token prompts at k=5; $225 standard at 5 pools × 1,000 items) | HIGH | k=3, 500 items, 3 pools in (b) |
| A5-06 | Batch queue limits (Luna 20M, Gemini 10M enqueued tokens) are ≈ 1.4–2k BillSum requests per wave | MEDIUM | sync calls for BillSum |
| A5-07 | Gemini thinking cannot be disabled (hidden output cost, unknown size); price rows came from a summary | MEDIUM | re-read pricing in a browser; log thought tokens |
| A5-08 | Haiku 512-token cache minimum vs 490–660-token label list: caching may silently not apply | LOW | test with usage fields |
| A5-09 | GPU allowance: carry-over unknown, idle billed, no NVL rental price, credit-to-dollar rate unknown | MEDIUM | band r; human: confirm carry-over, credit value |
| A5-10 | My token statistics come from clustered 1,000-row blocks of the train splits | LOW | re-run on test splits with count endpoints |
| A5-11 | Throughput band rests on FLOP arithmetic, not a measurement at 500 or 2,400 tokens | HIGH | 1-hour GPU benchmark session before committing the grid |

**I-09 (API protocol):** most of the document-checkable parts are now closed: Haiku temperature is unavailable, reasoning is controllable on Haiku and Luna but not fully on Gemini, ids are known. Remaining: live logprobs/usage fields, determinism flip rate, and whether Luna accepts temperature. Keep OPEN, severity HIGH→MEDIUM after one 50-item live test per provider (user's keys).

**I-15 (compute/cost estimates):** superseded by §2–3: token counts are measured, GPU-hours are given as ≈ 2.6–3× ranges, but throughput at 500 and 2.4k tokens is still extrapolated. Keep OPEN, HIGH until A5-11's benchmark; then FIXED-UNVERIFIED.

**Human-dependent:** confirm weekly-credit carry-over and the credit-to-dollar value; provide API keys for the 50-item live test.


---

## Round 2: re-pricing of the decided design (2026-11-21)

Labels as above. Basis: coordinator's decided design (B77 + BillSum confirmatory, Qwen3-4B + Phi-4-mini, N ∈ {200, 1000, 3000}, R = 8 / 5, 4 trials per system, LEDGAR exploratory). Same throughput band and token counts as §1.2/§2 (measured tokens, FLOP-based throughput [U]); all GPU numbers are ranges (fast–slow), not intervals. Scripts: scratchpad `s/r2.py`, `r2api.py`.

### R2.0 Correction to A5-02, and a cost trap in `tune()`
`tune()` and `select_trials()` exist (commit `b12f80c`); I looked for a module, not a function. But as coded, `select_trials` retrains **from scratch for every (lr, epochs)** pair (EPOCH_GRID 3, 5, 10, 20, 40 ascending, patience 2, duplicates removed by the S_MIN step floor) [V, code]. That is ≥ 3 epoch-settings per lr, i.e. ≥ 18 epochs of training per lr, up to 78. The decided protocol ("epochs by early stopping inside the run", 4 trials) is a **code change** (A5-12) and I priced the decided one: each trial = 1.5–2× the final run's sample-passes (best epoch + 2 patience epochs), plus a validation-loss/accuracy pass. Priced as coded, BillSum N=1000 tuning alone would be 6–43 GPU-h per model; it is infeasible.

### R2.1 GPU-hours (vLLM evaluation, LoRA, both models)

Per-run times (fast–slow): B77 N=3000 Phi 6–20 min, Qwen 6–21 min; BillSum N=3000 Phi **0.72–2.0 h**, Qwen **0.75–2.1 h**. BillSum tuning arithmetic fixed (red-team item): trials = 4 lr × (R2.0 length), **per model × N, once**, not 2 lr × 3 epoch settings.

| Component | Formula | GPU-h (fast–slow) |
|---|---|---|
| B77 fine-tune finals | 2 models × 8 reps × (N=200, 1000, 3000), 3,080 test items per run | 3.5–10.8 |
| B77 tuning | 4 trials × 2 models × 3 N | 2.1–8.5 |
| B77 SLM ICL + kNN | per model: 24 retrieved + 3 (N=77) + 3 random-k + 1 zero-shot test passes + validation (4 k × 3 N); ×2 models | 1.0–3.5 |
| BillSum finals | 2 × 5 reps × 3 N, 1,000 test items | 12.5–33.0 |
| BillSum tuning (decided: all N) | 4 trials × 2 models × 3 N | **14.4–51.2** |
| BillSum SLM ICL | per model 15 retrieved + 3 random-k + zero-shot + validation, k ≤ 3; ×2 | 1.9–5.3 |
| **Decided design total** | | **35.5–112** |
| Gated extras (F3 pilot ≈ 3, throughput benchmark 1, OOM probe and vLLM LoRA load test ≈ 1–2) | | 5–6 |
| LEDGAR exploratory (k_max 8, 3,000 items, N = 1000, 3000, R=3, both models) [I] | not in the total | +3–9 |

Memory (BillSum N=3000, micro-batch 1, gradient checkpointing, LM head only on completion positions): ≈ 25–45 GB for Phi/Qwen at the 6.7k-token maximum [I, ±40%]; no OOM expected if A3-04's rule (probe the longest 1% first) passes. Without `logits_to_keep` the fp32 logits at 6.7k × 200k are ≈ 5.4 GB ×3.5, which still fits at micro-batch 1.

### R2.2 API cost per provider (request counts by formula)

n_test = 3,080 (B77), 1,000 (BillSum). Per provider: `requests = n_test × (Σ_N R_N,retrieved + R_rand + 1 zero) + n_val × 4 k × (N levels) + 0.2 × n_test × 3` (10% repeat, 2 reruns). Confirmatory provider ("full"): B77 R = 8 at 3 N, N=77 at R=3 (exploratory), random-k N=1000 R=3; BillSum R = 5. Other two ("R=3"): R = 3 at 3 N, no N=77. Validation items per cell: min(1,000, N). Tokens: B77 retrieved/random 900–1,220, zero-shot 510; BillSum k ≤ 3: 6,950–9,300, zero-shot 2,250; outputs 8 / 250; billing multipliers Haiku 1.0–1.35, Luna 0.95–1.1, Gemini 0.95–1.1 [U].

| Provider role | B77 requests | BillSum requests | Standard $ | Batch (all) $ |
|---|---|---|---|---|
| Haiku, full | 106,436 | 28,400 | 33–56 | 16–28 |
| Haiku, R=3 | 50,688 | 22,400 | 23–38 | 11–19 |
| Luna, full | 106,436 | 28,400 | 31–46 | 16–23 |
| Luna, R=3 | 50,688 | 22,400 | 22–32 | 11–16 |
| Gemini, full | 106,436 | 28,400 | 80–118 | 40–59 |
| Gemini, R=3 | 50,688 | 22,400 | 55–81 | 28–40 |

Totals by confirmatory provider (standard / **mixed** = batch for B77, sync for BillSum on Luna and Gemini because their batch queues hold only ≈ 1.4–2k BillSum requests per wave [V limits above] / batch for everything): Haiku $110–168 / **$85–128** / $55–84; Luna $109–165 / **$88–130** / $55–82; Gemini $124–187 / **$99–147** / $62–94. B77 at 1.2k tokens fits Luna's 20M-token queue in ≈ 16k-request waves and Gemini's 10M in ≈ 8k-request waves (7 and 13 waves for 106k requests), so Gemini B77 batch is slow [I].
Extras: 5-replicate validation-split discordance runs (B77 N=1000, 1,000 items; BillSum 300 items; 3 providers) **$7.5–11.5**; 50-item live tests per provider < $1.
**Thinking-token margin (priced):** hidden output tokens per request, [U]: Gemini minimal thinking 0–300 × $1.50/M, Haiku 0–500 × $0.50/M only if thinking cannot be disabled (my page read says `thinking: {type: "disabled"}` is accepted at effort ≤ high; red team could not find it, so the live test decides), Luna 0. Margin: Gemini +$33–61 (more if confirmatory), Haiku +$0–34. Worst-case margin across the whole design is $51–79.
Total API budget: mixed + extras + worst margin = **$93–140 best case; up to ≈ $220 if Gemini is confirmatory with maximal thinking.** Standard-price mode exceeds $200 only in that worst case.

### R2.3 Rate band reconciliation (A4 vs A5)
Use **$3.6–6.0/h** (on-demand, own pages: RunPod Community $3.59 to Together HGX $5.99; DigitalOcean $4.47, Modal ≈ $4.54, Jarvislabs $4.59, Nebius $5.40; all SXM/HGX, **no NVL price exists on any page read**, so the NVL value is not like-for-like). A4's $2.3 corresponds to Jarvislabs's *spot* price ($2.29, interruptible, from verification 04) and is not an on-demand rate; keep it only as a sensitivity row. Rental equivalent of the design after the first two cuts (R2.4): 23–71 h → $83–426.

### R2.4 Does it fit, and cut order
Available: ≈ 60–75 productive GPU-h in total (§3), of which ≈ 5–6 are the gated items, leaving **≈ 54–70 for the grid**. The decided design needs **35.5–112 h**, so it fits only if throughput is in the fast-to-middle part of the band (geometric mid ≈ 63 h); at the slow end it does not. It fits the $50–200 API budget in mixed mode.

Cut ladder (cumulative GPU-h, grid only; first cut first):

| # | Cut | Effect | Cumulative GPU-h |
|---|---|---|---|
| 0 | Decided design | | 35.5–112 |
| 1 | BillSum tuning at N=3000: reuse N=1000's lr and epochs (A5 addition to the coordinator's order: it is the largest single item, −9 to −33 h; weakens F5 symmetry, so report it) | −9 to −33 | 26.5–79 |
| 2 | LEDGAR (not in total) and OLMo (already out) | 0 | 26.5–79 |
| 3 | Random-k (GPU −0.4 to −0.9 h; API ≈ −$10–25) | small | 26–78 |
| 4 | BillSum N=3000 R: 5 → 3 (status becomes exploratory; A6 F7: R=3 invalid for confirmatory) | −3 to −8 | 23–71 |
| 5 | BillSum N=3000 dropped; confirmatory BillSum = N 200, 1000 | −5 to −13 | 19–58 |
| 6 | BillSum with Qwen only | −6 to −18 | 18–54 |
API-only levers: BillSum API validation 300 items instead of 1,000 (−$22–39), random-k out (−$10–25), N=77 out (≈ −$2).

Go/no-go rule, written now: at the end of week 2, if the benchmark gives Qwen BillSum N=3000 final run ≤ 1.2 h and B77 N=3000 ≤ 12 min, keep the design with cut 1; if 1.2–2.0 h apply cuts 1 and 4; if > 2.0 h apply cuts 1, 5 and 6 and run BillSum as Qwen-only.

### R2.5 Build-and-run schedule (weeks start on the Sunday credit reset)

[USAGE-HEAVY] = long coding/debugging loops that burn the user's Claude allowance, more so if run in parallel sessions; parallel sessions shorten calendar time, not usage.

| Week | CPU/coding (dependency order) | GPU (≈13.3 h) / API |
|---|---|---|
| W1 Nov 22–28 | 1. Merge branches to one tagged commit, pin requirements (A5 §5). 2. Shared few-shot rendering in `task_spec`, kNN, embedder revision pin, replicate id in API records, test-read ledger [A3-04, USAGE-HEAVY]. 3. BillSum/LEDGAR loaders; tokenise all 18,949 train bills, set `max_length` [A3-04]. 4. `tune()` change to in-run early stopping + restricted grids (A5-12) [USAGE-HEAVY]. | GPU ≈ 3 h: 1-h throughput benchmark at 500 and 2.4k tokens, BillSum 20-step OOM probe. API: 50-item live tests per provider (keys), discordance runs ($8–12) |
| W2 Nov 29–Dec 5 | 5. SLM ICL harness (vLLM, LoRA serving test) [USAGE-HEAVY]. 6. BillSum gradient checkpointing + `logits_to_keep` [USAGE-HEAVY]. 7. A4-04 crossed-model/ratio function. 8. `frozen_arms.json` writer, cost estimator. | GPU ≈ 6 h: **F3 pilot** (Qwen B77 N=200, 1000, new prompt, tuned), vLLM adapter load test. **Gate Dec 5** (F3 pass + benchmark rule above) |
| W3 Dec 6–12 | API repeats, flip-rate logging; pre-registration tagged | Replicate 0 tuning, B77 both models (2–8 h); API validation (provider choice, k) |
| W4 Dec 13–19 | BillSum tuning code check | Replicate 0 BillSum tuning N ≤ 1000; freeze arms (hash committed) |
| W5–W6 Dec 20–Jan 2 | analysis scripts; A4's human faithfulness study (60 bills, author time, no dollars) | Main runs, replicate-major: B77 R=8, BillSum R=5 (≈ 25–45 h over two weeks at the middle of the band) |
| W7 Jan 3–9 | blinded variance re-estimation; extension to R=10 only if time | finish runs; API confirmatory cells inside 72 h per provider |
| W8 Jan 10–16 | writing; reruns of failed cells only | reserve ≈ 8–10 h |
LEDGAR only if the gate passes and W7 ends early.

### R2.6 New or updated issues
| ID | Issue | Sev | Action |
|---|---|---|---|
| A5-02 (withdrawn) | `train.tune` exists | - | closed |
| A5-12 | `select_trials` retrains from scratch per (lr, epochs), ≥ 18 epochs per lr; decided protocol needs in-run early stopping, else BillSum tuning is 6–43 GPU-h per model | HIGH | recode, test on CPU, keep restricted grid |
| A5-13 | Design needs 35.5–112 GPU-h vs ≈ 54–70 available; BillSum (finals 12.5–33 + tuning 14–51) is ≈ 80% of it | HIGH | cut ladder, gate rule |
| A5-14 | Thinking-token cost unmeasured (Gemini cannot be off, Haiku may); margin up to $51–79 | MEDIUM | live test, `usage` fields |
| A5-15 | Gemini batch waves of ≈ 8k (B77) and ≈ 700 (BillSum) requests; confirmatory-provider choice moves cost by ≈ $15–40 | LOW | sync fallback |
Register view: I-15 stays HIGH (ranges now re-priced, throughput still unmeasured). I-09 stays OPEN (live test pending).
