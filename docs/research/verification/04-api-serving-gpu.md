# Verification 04: API arm, serving, compute, pilot sanity

Read 2026-11-21 from official pages (fetched through web-fetch subagents). **No provider page shows a publication date**, so "date" below means the date read. Items the page tail did not show are marked UNVERIFIABLE, not guessed. Gemini pages redirect from ai.google.dev to aistudio.google.com/docs/…; the 2.5 Flash-Lite and 2.5 Flash prices were read from a model-extracted summary, not verbatim.

## A. API arm

| | Haiku 5.5 | GPT-6 Luna | Gemini 3.1 Flash-Lite | Gemini 2.5 Flash-Lite |
|---|---|---|---|---|
| id | `claude-haiku-5-5` (alias = id, no dated snapshot) | `gpt-6-luna` (no dated snapshot) | `gemini-3.1-flash-lite` (stable) | `gemini-2.5-flash-lite` (stable) |
| in / out per 1M | $0.10 / $0.50 (≤100k prompt); $0.50 / $2.50 above | $0.10 / $0.50 (≤272k); $0.20 / $0.75 above | $0.25 / $1.50 (thinking billed as output) | $0.10 / $0.40 (summary) |
| batch | 50% | 50% ($0.05 / $0.25) | 50% ($0.125 / $0.75) | $0.05 / $0.20 (summary) |
| caching | explicit; read $0.01, write 5-min $0.125 / 1-h $0.20; min 512 tok | automatic; cached $0.01, 30-min default; min 1,024 inferred (guide names GPT-5.6+, not Luna) | implicit default + explicit; read $0.025, storage $1/1M/h; min tokens not found | read $0.01 (summary); page also says "Not available" in one column: conflict |
| context / max out | 1M / 128k | 1.05M / 128k | 1,048,576 / 65,536 | same |
| logprobs | not documented on 6 pages read | Chat `logprobs` param exists; Luna support not stated | `responseLogprobs` exists (top-N 0–20); no 3.x exclusion found | same |
| temperature 0 | see below | allowed 0–2; Luna restrictions not stated | allowed 0–2 | same |
| reasoning | adaptive, default effort medium; how to disable not found | `none` supported; default `medium`; billed as output | default "minimal" (near-off, not fully off); billed as output | default off |
| lowest-tier limits | Start tier 4,000 RPM / 2M ITPM / 400k OTPM | Build 5,000 RPM / 2M TPM | not published per model | n/a |
| deprecation | none; retirement not before 2027-10-07 | not listed | earliest shutdown 2027-05-07 | none announced, but access "limited to users who have actively used" 2.5 |

Key findings:
- **Haiku 5.5 and Luna are both $0.10/$0.50.** The earlier numbers are right, but Haiku is priced by prompt length (≥100k tokens costs 5×). The 17k summarisation prompt is under it.
- **Anthropic temperature trap.** The deprecations page lists `temperature`, `top_p`, `top_k` "Removed in API version 2026-10-21" (400 error) for Claude 4.7+ models. It does not name Haiku 5.5, so I could not confirm the rule applies. Temperature-0 determinism on Haiku is therefore UNVERIFIED and probably unavailable; test one call and pin the API version.
- **Haiku logprobs** are absent from the docs, so "none" cannot be confirmed from the docs.
- **Gemini "logprobs no longer returned for 3.X" is not supported by the docs** I read; the only "unsupported" list is for Live API tokens. Run one real call.
- **Gemini 2.5 Flash-Lite may not be available to a new account.** Gemini 3.1 Flash-Lite shuts down no earlier than 2027-05-07, outside the paper window.
- **Tokenizer.** Claude 4.7+ uses about 30% more tokens. Haiku 5.5 is not named; count with the token-count endpoint, because 1,400 tokens may already be a different tokenizer's count.
- **Batch queue limits.** Gemini Tier 1 allows 10M enqueued tokens; OpenAI Build allows 20M. One summarisation level is 17M input tokens, so split the batches.

### Grid cost recomputed
Per-request tokens do not depend on N, so each API gets 4 N-levels × 1,000 = 4,000 requests per task. The Gemini row excludes thinking tokens, and Haiku/Luna must have thinking or reasoning off or they add output cost.

- Classification, Haiku/Luna: (1,400×0.10 + 5×0.50)/1e6 = $0.0001425 per request, so ×4,000 = **$0.57** ($0.285 batch).
- Summarisation, Haiku/Luna: (17,000×0.10 + 300×0.50)/1e6 = $0.00185, so ×4,000 = **$7.40** ($3.70 batch).
- Classification, Gemini 3.1: (1,400×0.25 + 5×1.5)/1e6 = $0.0003575, so ×4,000 = **$1.43** ($0.72 batch).
- Summarisation, Gemini 3.1: (17,000×0.25 + 300×1.5)/1e6 = $0.0047, so ×4,000 = **$18.80** ($9.40 batch).

| | Haiku | Luna | Gemini 3.1 | all three |
|---|---|---|---|---|
| standard | $7.97 | $7.97 | $20.23 | **$36.17** |
| batch | $3.99 | $3.99 | $10.12 | **$18.09** |
| fully cached prefix, standard (upper bound on savings) | $1.74 | $1.74 | $4.66 | $8.14 |

The fully cached figure assumes only 100 (classification) or 1,000 (summarisation) tokens per request are uncached. **It does not apply to the planned design.** Retrieved shots differ per item, so only the shared instruction block caches (maybe a few hundred tokens). Caching helps only the fixed-shot or random-k controls. Whether batch and cache discounts stack is not verified. The earlier "$51 / $26" does not reproduce: my total is $36 / $18, and the gap is probably reasoning or thinking tokens or the 2.5 variants. Budget a margin for thinking tokens and the tokenizer: +30% input on Haiku adds about $2.

## B. Serving
- **vLLM 0.13.0 LoRA support** is CONFIRMED from the supported-models table (docs.vllm.ai/en/v0.13.0/models/supported_models/): `Phi3ForCausalLM`, `Qwen3ForCausalLM` and `OLMo3ForCausalLM` (the table's spelling) all have LoRA ✅. This is the docs table only, not a test; run a real adapter load before relying on it.
- **`max_lora_rank`:** the allowed values are UNVERIFIABLE from the docs (the examples use 64 and 256). Pilot rank 16 is fine. Set `--max-lora-rank` to the largest adapter rank. `--max-loras` and `--max-cpu-loras` have no documented defaults.
- **DoRA rejection is CONFIRMED in v0.13.0 source.** `peft_helper.py` has `if self.use_dora: error_msg.append("vLLM does not yet support DoRA.")`.
- **Issue #60223 is REAL.** "LoRA adapters using PEFT features vLLM does not implement load without error and give wrong outputs", opened 2026-10-06, closed 2026-10-09 by PR #60225.
  - The issue measured PiSSA and OLoRA mis-served; LoftQ and CorDA were *not* measured. It lists rsLoRA as unaffected.
  - **The fix is not in v0.13.0** (the merge is after the tag), so on 0.13.0 PiSSA/OLoRA/LoftQ still load silently wrong. The DoRA-serving PR #47395 is open and labelled stale. The earlier claim is PARTLY correct.
- **Merged DoRA serves normally:** PARTLY. PEFT says "recommended to merge weights for inference", but nothing documents serving a merged DoRA model. Test once.
- **PEFT 0.18.0.** `use_rslora` scales by `lora_alpha/math.sqrt(r)` versus the default `lora_alpha/r`: CONFIRMED. `use_dora` "decomposes the updates … magnitude and direction", linear and Conv2D only. Overhead: only "a bigger overhead than pure LoRA", with no figure. The only quantitative evidence is the pilot: DoRA is +31% train time on Phi (23.4/17.9 s) and +42% on Qwen, +19% peak memory on both. The "+139%" is not an official figure.

## C. Compute
- **H200 SXM vs NVL** (nvidia.com/en-us/data-center/h200, "Preliminary specifications"): both 141 GB and 4.8 TB/s. TDP is "Up to 700W" for SXM and "Up to 600W" for NVL (PCIe dual-slot, NVLink bridge). 143,771 MiB equals 141 GB.
- **Rental prices are SXM/HGX, not NVL.** I found no NVL listing. A band quoted for "H200" overstates the NVL hourly value, or at least is not like-for-like. Say so in the paper.
- **Prices, own pages, on-demand:**
  - RunPod Pods $5.29 Secure / $3.59 Community ("Updated November 12, 2026").
  - Jarvislabs $4.59 for "H200 SXM" (spot $2.29).
  - Nebius $5.40 "HGX H200".
  - Hyperbolic $3.99, labelled "Starting From" on /marketplace, not a fixed rate.
  - CoreWeave $50.44 per 8-GPU instance, so $6.31 per GPU (my division).
  - Lambda and Vast.ai: no H200 price on the pages. The earlier "Vast median $4.44" is UNVERIFIABLE.
- **CodeCarbon.**
  - It measures GPU + CPU + RAM only. GPU energy comes from NVML through `nvmlDeviceGetTotalEnergyConsumption` (power-estimation page; the methodology page words it differently). So its GPU number equals a direct NVML read.
  - If NVML is unavailable, GPU energy is not counted at all. CPU uses RAPL if readable, otherwise a TDP×load model. In Docker, RAPL needs `-v /sys/class/powercap:/sys/class/powercap:ro` plus host read permission, and is not exposed by default.
  - RAM is a per-DIMM guess (5 W per DIMM), described as unsourced.
  - Carbon factor order: forced value, cloud region, Electricity Maps, regional data, country (OWID), world average of 475 g. Online mode geolocates by IP, which the docs say is wrong for some cloud regions.
  - The server is "h200-madrid" and codecarbon is not installed. **Pin `country_iso_code="ESP"` (or the actual host country).** Use direct NVML GPU energy as primary, CodeCarbon for the factor and CPU/RAM, and state that CPU/RAM are modelled.

## D. Pilot sanity (`results/pilot-2026-11-21/pilot.jsonl`, `curve.jsonl`)
- **Throughput is plausible but not transferable.** The pilot measures 3.5k–7.6k real tokens/s; Phi LoRA is about 90 ms per 16-sequence step. Sequences are very short (longest example 62–152 tokens, about 42 on average), so the run is latency-bound and well below the H200's compute limit. Throughput already rises with N (Phi 3.6k at N=50, 7.6k at N=5000) as sequences lengthen.
  - The handoff's "20+ h for BillSum LoRA from 5,000 tok/s" therefore likely **over**-estimates; at 2.6k tokens the run is compute-bound. Measure it.
- **Peak memory (14–35 GB) is not transferable either.** Micro-batch 16 at 2.6k tokens multiplies fp32 logits by about 5. Phi's vocabulary is about 200k, giving roughly 30 GB of logits alone. Expect OOM risk for 7B without gradient checkpointing or a smaller micro-batch. At 512 tokens it is fine: weights are 8–15 GB.
- **Anomalies:**
  1. `first_train_loss` is 18.88 for Phi LoRA, DoRA and rsLoRA (identical), but 8.5 for QLoRA. An initial loss above ln(200k) ≈ 12.2 means the base model is confidently wrong on the label tokens, and 18.9 versus 8.5 for the *same* base weights is too large a gap for 4-bit noise alone. Check masking, padding or chat-template handling before trusting Phi.
  2. OLMo DoRA's `first_train_loss` is NaN, so it failed at step 1 rather than diverged later. The handoff's "likely diverged" guess is wrong.
  3. rsLoRA beating LoRA by 4–6 points at identical lr is consistent with undertraining: the scale is α/√r = 8 versus 2, effectively a 4× larger step. Do not read it as a technique effect.
  4. DoRA time overhead by the unrounded numbers is +31% (Phi) and +42% (Qwen); the handoff says +28% and +41%.

## Master table

| Item | Value | Source | Date read | Verdict |
|---|---|---|---|---|
| Haiku id | claude-haiku-5-5 | platform.claude.com/docs/en/about-claude/models/overview | 2026-11-21 | CONFIRMED |
| Haiku price | $0.10 / $0.50 | …/about-claude/pricing | 2026-11-21 | CONFIRMED |
| Haiku logprobs none | not documented | pricing, overview, caching, rate-limits, deprecations, batch | 2026-11-21 | UNVERIFIABLE |
| Haiku temperature 0 | removed in API ver 2026-10-21 for 4.7+; Haiku not named | …/model-deprecations | 2026-11-21 | UNVERIFIABLE |
| Luna id / price | gpt-6-luna, $0.10 / $0.50 | developers.openai.com/api/docs/models/gpt-6-luna | 2026-11-21 | CONFIRMED |
| Luna reasoning off | `none` supported, default medium | …/guides/reasoning | 2026-11-21 | CONFIRMED |
| Luna logprobs | param exists; Luna support not stated | …/api/reference chat completions | 2026-11-21 | UNVERIFIABLE |
| Gemini 3.1 FL id / price | gemini-3.1-flash-lite, $0.25 / $1.50 | aistudio.google.com/docs/pricing | 2026-11-21 | CONFIRMED |
| Gemini 2.5 FL price | $0.10 / $0.40 | same, summary-only | 2026-11-21 | PARTLY |
| Gemini 3.x no logprobs | no such statement | aistudio.google.com/api-reference/generate-content | 2026-11-21 | REFUTED (docs) / needs live test |
| gemini-2.5-flash deprecated 2026-12-01 | auto-routes to gemini-3.5-flash-lite | …/docs/deprecations | 2026-11-21 | CONFIRMED |
| Batch discount | 50% on all three | each batch page | 2026-11-21 | CONFIRMED |
| Caching reads | $0.01 / $0.01 / $0.025 | pricing pages | 2026-11-21 | CONFIRMED |
| vLLM LoRA for Phi3/Qwen3/OLMo3 | ✅ | docs.vllm.ai/en/v0.13.0/models/supported_models/ | 2026-11-21 | CONFIRMED (docs only) |
| vLLM rejects DoRA | in peft_helper.py v0.13.0 | raw.githubusercontent.com/vllm-project/vllm/v0.13.0/vllm/lora/peft_helper.py | 2026-11-21 | CONFIRMED |
| PiSSA/OLoRA/LoftQ mis-served | issue #60223, LoftQ unmeasured, fix not in 0.13.0 | github.com/vllm-project/vllm/issues/60223 | 2026-11-21 | PARTLY |
| rsLoRA α/√r | confirmed | huggingface.co/docs/peft/v0.18.0/en/package_reference/lora | 2026-11-21 | CONFIRMED |
| DoRA overhead | qualitative only | same | 2026-11-21 | PARTLY |
| H200 NVL 600 W vs SXM 700 W | both 141 GB, 4.8 TB/s | nvidia.com/en-us/data-center/h200 | 2026-11-21 | CONFIRMED |
| RunPod / Jarvis / Nebius / Hyperbolic | $5.29 / $4.59 / $5.40 / $3.99 (from) | each provider's pricing page | 2026-11-21 | CONFIRMED (SXM; Hyperbolic "from") |
| Lambda, Vast.ai | no H200 value | own pages | 2026-11-21 | UNVERIFIABLE |
| Start-tier Haiku limits | 4,000 RPM / 2M ITPM / 400k OTPM | …/api/rate-limits | 2026-11-21 | CONFIRMED |
| Build-tier OpenAI limits | 5,000 RPM / 2M TPM | …/guides/rate-limits | 2026-11-21 | CONFIRMED |
| Gemini per-model limits | in AI Studio only | …/docs/rate-limits | 2026-11-21 | UNVERIFIABLE |

## Corrections

**`docs/research/2026-11-21-verified-findings.md`**
1. Line 55: state that DoRA rejection is source-confirmed for v0.13.0, that LoftQ mis-serving was not measured, and that the fix (PR #60225) postdates 0.13.0.
2. Line 57: replace "unknown" overhead with the pilot's +31% / +42% train time and +19% memory.
3. Line 64: the university card is NVL (600 W). All quoted prices are SXM/HGX.
4. Lines 66–72: drop "Vast median $4.44" (unverified). Mark Hyperbolic as "from". Add CoreWeave $6.31 (derived).
5. Lines 79–81: add the >100k Haiku/Luna tiers. Replace "no logprobs" with "not documented". Delete the "no logprobs for 3.x" claim or label it untested.
6. Add the Anthropic temperature-removal risk and that Gemini 2.5 Flash-Lite access is limited to prior users.
7. Line 83: replace "$51 / $26" with $36 / $18 (excluding thinking), and note that caching does not apply to per-item retrieved shots.

**`docs/PLAN.md`**
1. Line 29: swapping to Gemini 2.5 Flash-Lite may not be possible for a new account; do not plan on it.
2. Lines 95–110: split batches for the 10M/20M enqueued-token limits. Disable thinking and reasoning explicitly for all three.
3. Lines 116–122: pin the CodeCarbon country, use NVML direct for the GPU, and run a real adapter-load test of all three architectures.
4. Add to the pilot section: investigate the Phi 18.9 initial loss and the OLMo DoRA step-1 NaN. Re-measure tok/s and memory at BillSum length before computing GPU-hours.
