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
