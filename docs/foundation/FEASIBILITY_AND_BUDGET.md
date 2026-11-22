# Feasibility and budget

Assembled from `agents/05-feasibility.md` (A5; **Round 2 supersedes** the three Round 1 scenarios), with request counts from A5 Round 2. **Updated 2026-11-22 (second pass):** the request counts are now settled by `analysis/count_requests.py` (L-22, I-29: BANKING77 207,504, BillSum 67,200, total 274,704; section 2 below shows both readings), the API arm is **unfunded** (L-32: no keys, no money), the GPU credits carry no dollar value (L-32), and the fine-tuning design changed (Qwen only is confirmatory, D-05; five learning rates, D-06). The dollar and GPU-hour figures below are **not re-derived** for these changes; each affected block says so. Open contradictions: `PREREG_CONSISTENCY.md` section F. Labels: [V] verified from an official page on 2026-11-21 (pages read through fetch sub-agents, so page text is model-relayed), [M] measured, [I] interpretation, [U] unverified. All GPU numbers are fast-slow ranges from FLOP arithmetic (assumed 12-30% MFU), **not measurements**; throughput at 500 and 2,400 tokens is unmeasured (A5-11, HIGH).

## 1. API prices and protocol facts (USD per 1M tokens, standard, text)
| | Haiku 5.5 | GPT-6 Luna | Gemini 3.1 Flash-Lite |
|---|---|---|---|
| input / output | 0.10 / 0.50 (prompt <=100k) | 0.10 / 0.50 (<=272k) | 0.25 / 1.50 (medium confidence, from a truncated page) |
| batch | 50% | 50% | 50% |
| reasoning billed as output | yes; off via `thinking: {type: disabled}` per A5, not confirmed by the red team | yes; off via `reasoning.effort=none` | yes; **cannot be fully off** (minimal) |
| temperature | omit (do not send) | not stated | not stated |
| tokenizer | newer tokenizer, about 30% more tokens than Haiku 4.5; free `count_tokens` | not public | about 4 chars/token; `countTokens` |
| batch queue | 200,000 requests | 20M enqueued tokens | 10M enqueued tokens |
Sources: platform.claude.com, developers.openai.com, aistudio.google.com pages, accessed 2026-11-21 (list in A5 section 1.1). Haiku 5.5 id is a "pinned snapshot" by provider claim; Luna has only an undated default id. Billing multiplier vs the Qwen token count: Haiku 1.0-1.35, Luna 0.95-1.1, Gemini 0.95-1.1 [U]. Use **paid tiers only** (Google's unpaid tier trains on inputs; DATASET_AUDIT).

Measured tokens [M, 1,000-row clustered train blocks]: BANKING77 zero-shot prompt about 494-510 tokens, k=32 about 1,220; BillSum bill mean 2,111 (p95 3,673), summary mean 229, zero-shot about 2,250, k=3 about 9,300 (k=5 13,900); LEDGAR k=8 about 1,770.

## 2. API budget (decided design; request counts by formula) [UNFUNDED, L-32: a plan, no request has been or can now be made]
**Count check (2026-11-22).** `python analysis/count_requests.py` gives, per provider role: BANKING77 selected 106,128, other 50,688; BillSum selected 28,400, other 19,400; totals 207,504 + 67,200 = 274,704 (L-22). The table below keeps A5's counts (106,436; 28,400; 22,400): the BANKING77 figure differs by 308 (0.3%) from the script, and A5's BillSum 'R=3' rows include random-k on every provider (3,000 more requests per other provider; the protocol default runs random-k on the selected provider only, `PC-08`). **The dollar columns were priced on A5's counts and are not re-priced** (L-22: no dollar figure until the API harness cost gate and the price pages of the run day); they therefore overstate BillSum by 6,000 requests in total and ignore the decided k grid. The script uses 4 k values per cell, the decisions use 5 for BANKING77 and 3 for BillSum: by arithmetic on the script's formula the validation term changes by +6,600 BANKING77 requests and -6,600 BillSum requests across three providers (a coincidence of cancelling, valid only if BillSum has no retriever trials; k=64 prompts are also longer than the measured k=32 ones) (`PC-33`). Sync vs batch: the researcher's decision is batch-based (D-14, L-27); the 'mixed' totals below follow it, `PC-02` recommended sync and `PC-35` is open.
| Provider role | B77 requests | BillSum requests | Standard $ | Batch $ |
|---|---|---|---|---|
| Haiku, full (R=8/5) | 106,436 | 28,400 | 33-56 | 16-28 |
| Haiku, R=3 | 50,688 | 22,400 | 23-38 | 11-19 |
| Luna, full | 106,436 | 28,400 | 31-46 | 16-23 |
| Luna, R=3 | 50,688 | 22,400 | 22-32 | 11-16 |
| Gemini, full | 106,436 | 28,400 | 80-118 | 40-59 |
| Gemini, R=3 | 50,688 | 22,400 | 55-81 | 28-40 |
Totals by confirmatory provider (standard / mixed = batch for B77, sync for BillSum / all batch): Haiku $110-168 / $85-128 / $55-84; Luna $109-165 / $88-130 / $55-82; Gemini $124-187 / $99-147 / $62-94. Mixed mode is used because Luna and Gemini batch queues hold only about 1.4-2k BillSum k=3 requests per wave. Extras: validation-split discordance runs $7.5-11.5; 50-item live tests under $1 per provider. Thinking-token margin (hidden output): Gemini +$33-61, Haiku +$0-34 only if thinking cannot be disabled. **Total API budget $93-140 best case, up to about $220 if Gemini is the confirmatory provider with maximal thinking.** [A5 estimates on A5's request counts and four k values, price pages of 2026-11-21; not re-priced after L-22, not a funded budget (L-32)] Flip-rate branch (if a selected provider exceeds 2%): +$7-20 (B77) and +$25-40 (BillSum). API-only levers: BillSum validation on 300 items (-$22-39), random-k out (-$10-25), N=77 out (about -$2). Caching: per-item retrieved shots do not cache; the label-list prefix may or may not (Haiku 512-token minimum).

## 3. GPU budget (H200 NVL, university allowance)
Allowance: 200 credits/week, reset Sunday, 15 credits/h, about 13.3 h/week, about 60-75 productive hours in the eight weeks; carry-over unknown; the credits carry no dollar value (HD-3 answered, L-32). Per-run (fast-slow): B77 N=3,000 about 6-21 min (either model); BillSum N=3,000 **0.72-2.1 h**.

| Component | GPU-h (fast-slow) |
|---|---|
| B77 fine-tune finals (2 models x 8 reps x 3 N) | 3.5-10.8 |
| B77 tuning (4 trials x 2 models x 3 N; now 5 trials, Qwen only confirmatory) | 2.1-8.5 |
| B77 SLM ICL + kNN | 1.0-3.5 |
| BillSum finals (2 x 5 x 3 N) | 12.5-33.0 |
| BillSum tuning (all N) | **14.4-51.2** |
| BillSum SLM ICL | 1.9-5.3 |
| **Decided design total (A5, two models, four trials; not re-derived for D-05 and D-06)** | **35.5-112** |
| Gated extras (pilot about 3, throughput 1, OOM and vLLM LoRA tests 1-2) | 5-6 |
| LEDGAR exploratory (not in total) | +3-9 |
Available for the grid about 54-70 h, so the design fits only if throughput is fast-to-middle (geometric mid about 63 h). BillSum is about 80% of the load. Memory: B77/LEDGAR fit without gradient checkpointing; BillSum needs micro-batch 1, gradient checkpointing and an LM head on completion positions only (about 25-45 GB [I, +-40%]).

**Cut ladder (cumulative GPU-h for the grid):** 1) BillSum tuning at N=3,000 reuses N=1,000's lr and epochs (-9 to -33; weakens tuning symmetry, to be reported) -> 26.5-79; 2) LEDGAR and OLMo out; 3) random-k out; 4) BillSum N=3,000 at R=3 (exploratory, since R=3 is invalid for confirmatory) -> 23-71; 5) BillSum N=3,000 dropped -> 19-58; 6) BillSum Qwen-only -> 18-54.
**Go/no-go at end of week 2:** Qwen BillSum N=3,000 final run <=1.2 h and B77 N=3,000 <=12 min: keep the design with cut 1; 1.2-2.0 h: cuts 1 and 4; >2.0 h: cuts 1, 5 and 6 (BillSum Qwen-only).
GPU rate band **$3.6-6.0/h** (RunPod Community $3.59 to Together HGX $5.99; DigitalOcean $4.47, Modal about $4.54, Jarvislabs $4.59, Nebius $5.40; all SXM/HGX, **no NVL price found**, so a scenario not an interval). Rental equivalent of the design after cuts 1-2: 23-71 h, $83-426. The university allowance is not billed in dollars; the imputed rate is for the cost model only [`HUMAN_DECISIONS.md` HD-3].

## 3a. What has been measured on the GPU (2026-11-22; exploratory pilot runs, Qwen3-4B-Instruct-2507 on one H200 NVL; [M] unless marked)
Sources: `results/pilot-stage2-2026-11-22/`, `pilot-stage2b-2026-11-22/`, `pilot-session3-2026-11-22/`, `PROGRESS.md`.
| Item | Measured | Note |
|---|---|---|
| B77 fine-tuning final, N=200 / N=1,000 (train time, peak memory) | 71 s, 53 GB / 432 s, 81 GB (630 steps) | N=1,000 peak is 81 GB without gradient checkpointing, inside the 141 GB card |
| B77 tuning, Qwen (all trials of one cell) | N=200: 457-509 s (4 lr, 200 val rows); N=1,000: 2,534-2,537 s (4 lr, 1,000 val rows); N=3,000: 4,241 s = 70.7 min (5 lr, 1,000 val rows) | sum as run 7,287 s = 2.0 GPU-h [arithmetic]; the N=3,000 extension trial (1.6e-3, `PC-32`) is about +12 GPU-min [estimate, `PROGRESS.md`] |
| B77 N=3,000 final | **not measured**; linear extrapolation from 432 s for 630 steps to 940 steps: about 646 s = 10.8 min [I] | the go/no-go threshold is 12 min |
| BillSum probe (20 steps, longest 1% of bills) | 24.4 GB peak, 8,248 tokens/s; projected one-run upper bounds 15.4 / 48.6 / 72.5 min for N = 200 / 1,000 / 3,000 | class "cuts 1 and 4" (72.5 min vs the 72 min threshold) |
| BillSum runs | tuning N=200 (5 lr, 200 val bills): 2,568.5 s = 42.8 min; frozen final N=200: 120.8 s, 19.2 GB; untuned N=1,000 (315 steps, batch 16): 1,512 s, 19.2 GB, about 4.8 s per step | extrapolating the measured step time to 630 / 940 steps gives 50.4 / 75.2 min [arithmetic], slightly above the probe's 48.6 / 72.5; the projections are upper bounds only if loss-based stopping ends runs early [I] |
| Local ICL k tuning (5 k) | about 50-100 s per N on 200 / 1,000 / 1,000 validation rows | from the log timestamps |
| GPU used on 2026-11-22 | about 6 h of the 13.3 h weekly allowance (assistant's accounting; billed time may be higher) | next reset on the Sunday after (2026-11-29) per the stated weekly rule |
Consequences. (1) A5's fast-slow GPU table above is FLOP arithmetic for two models and four trials; it has **not** been recomputed for Qwen-only confirmatory cells (D-05) and five trials (D-06). (2) `HANDOFF.md` gives a campaign estimate of 20-27 GPU-h for Qwen BANKING77 plus BillSum without BillSum N=3,000 (25-33 with it), about two weeks of the weekly allowance; no foundation file derives it, so it is an estimate, not a budget (`PC-38`). (3) BillSum generation and ROUGE-L evaluation have no timing at all yet. (4) The credits have no dollar value (L-32); the imputed rental band $3.6-6.0/h remains the only dollar basis, a scenario.

## 4. Schedule [SUPERSEDED as a calendar, kept for the order of work: it predates the two-stage freeze (tag A, tag B; D-08, L-30), the 'API last' decision (`PROGRESS.md` 05:50 entry) and the funding answer (L-32); no new dates are set, `PC-14`, `PC-29`] (weeks start on the Sunday credit reset; [USAGE-HEAVY] = long coding loops that consume the researcher's Claude allowance)
| Week | Coding (dependency order) | GPU / API |
|---|---|---|
| W1 Nov 22-28 | merge branches, tag, pin requirements; shared few-shot rendering, kNN, embedder pin, replicate id, test-read ledger [USAGE-HEAVY]; BillSum/LEDGAR loaders, tokenise all bills; `tune()` to in-run early stopping [USAGE-HEAVY] | about 3 GPU-h: throughput benchmark, BillSum OOM probe; API live tests and discordance runs (needs keys, HD-1: none exist, L-32) |
| W2 Nov 29-Dec 5 | SLM ICL harness (vLLM) [USAGE-HEAVY]; BillSum gradient checkpointing + `logits_to_keep` [USAGE-HEAVY]; crossed-model/ratio functions; `frozen_arms.json` writer | about 6 GPU-h: **F3 pilot**, vLLM adapter load test. **Gate Dec 5** |
| W3 Dec 6-12 | API repeats, flip-rate logging; pre-registration tagged | replicate-0 tuning B77 (2-8 h); API validation (provider, k) |
| W4 Dec 13-19 | BillSum tuning code check | replicate-0 BillSum tuning N<=1,000; freeze arms (hash committed) |
| W5-W6 Dec 20-Jan 2 | analysis scripts; human faithfulness study (author time) | main runs, replicate-major (about 25-45 h at mid band) |
| W7 Jan 3-9 | blinded variance re-estimation | finish runs; API confirmatory cells within 72 h per provider |
| W8 Jan 10-16 | writing; reruns of failed cells | reserve 8-10 h |
**Cap-aware reading (06c C5):** 200 credits / 15 = 13.3 h per week, 106.7 h gross over eight weeks; W5-W6 hold at most 26.7 h, so the 25-45 h listed there does not fit unless unused credit carries over (HD-3: carry-over unknown). Base case: no carry-over; with it unknown, treat 54-70 productive hours as possibly 20-30% too high. The BillSum OOM probe needs the gradient-checkpointing code listed in W2: move that code to W1 or the probe to W2. Add a budget line of 0.7-2.0 GPU-h and under $1 API for the tuning-bar sensitivity (EXPERIMENT_PROTOCOL section 4, L-19). Red-team caveat (F9): engineering is at least four weeks [I], so writing has little slack; LEDGAR only if the gate passes and W7 ends early. Calendar dates assume the pilot gate is passed on schedule: the schedule is **not** a promise.

## 5. Reproducibility limits
Haiku has no sampling control and Luna/Gemini may differ: pin ids, store `model_id_returned`, repeat 10%, report flip rate; run each provider's confirmatory cells inside 72 h. Open models: record `pip freeze`, adapter hashes, seeds, GPU and driver; GPU kernels are nondeterministic so replicate variance absorbs it. Three branches are unmerged and the pinned requirements never installed on the server: merge to one tagged commit first.

## 6. Open items
A5-02 withdrawn; A5-12 (HIGH, recode `select_trials`), A5-13 (HIGH, 35.5-112 vs 54-70 GPU-h, handled by the ladder), A5-14 (MEDIUM, thinking-token cost unmeasured), A5-15 (LOW, Gemini batch waves), A5-11 (HIGH, throughput benchmark), A5-09 (MEDIUM, credit value and carry-over). Human: API keys and a spending cap for the live tests (HD-1: answered 2026-11-22, none exist, L-32); credit value (HD-3: answered, no dollar value; carry-over unknown).
