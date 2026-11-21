# A3: Experimental protocol (draft of EXPERIMENT_PROTOCOL.md)

Date 2026-11-21. No GPU and no paid API used. Labels: **[V]** verified by me (code read, offline test, or measurement), **[R]** own recommendation, **[U]** unverified assumption. Evidence base: the three harness branches extracted to a scratch directory and merged there (145 offline tests pass, 3 skipped for missing `seqeval`/`rouge_score`; on the training branch alone the prompt-equivalence tests are *skipped* because `src/api` is absent, so they only mean something after merging); pilot files; HF datasets-server statistics and 1,000 BillSum rows plus 400 LEDGAR rows (small requests); the BANKING77 train parquet (0.3 MB, CC BY 4.0), for a CPU-only kNN analysis; real Qwen3-4B and Phi-4-mini tokenizers.

## 1. Information equivalence (decision D1)

**Decision [R].** One shared `task_spec` per task defines system text (task sentence + full label list, `- label` lines), user turn (`Input: …\nLabel:`), output cap, decoding and parser. Arms differ only in *what they do with the pool of N labelled items*:

| Arm | Receives |
|---|---|
| zero-shot (SLM, API) | spec only |
| random-k / retrieved-k (SLM, API) | spec + k pool items in the user turn, nearest neighbour last (as in `plan_items`), same `Input/Label` rendering |
| kNN baseline | no LLM; label of the most similar pool item(s) |
| LoRA | trained on the pool with the *zero-shot* spec as prompt (the shots-free prompt), tested on the same |

[V] The zero-shot system+user text built by `src/train/task_spec.py` is byte-identical to the API `CLASSIFICATION` template output (checked in the merged copy). Parsers agree on all 77 gold labels and on 10 probe outputs.

Equal information means: same task text, same label list, same pool, same test items. It does not mean equal bits used. ICL sees at most k items, fine-tuning sees all N. This is stated, and a many-shot probe (k=128, ≈3.5k tokens) at N=1000 and 5000 on the cheapest API and one SLM bounds how much the k cap matters [R]. Retrieval cannot leak: the pool is drawn from train minus validation, so test items are never retrievable, and retrieval is the same function for SLM and API arms (§2). Fine-tuning gets no information beyond prompting: the same pool, no validation labels in training (validation labels are used for selection only, see §5 and issue A3-03).

**k limits [V, measured with Qwen/Phi tokenizers]:** BANKING77 spec 476–490 tokens, shot ≈23.5 tokens, so k=32 ≈1.3k, k=128 ≈3.5k. LEDGAR spec ≈413 tokens, shot ≈138 tokens (mean 130 text tokens, p99 589), so k=32 ≈5k. BillSum: see §3. All models and APIs have far larger windows, so k is limited by cost, not context. **k_max = 32 for both classification tasks, 3 for BillSum** (k grid {4,8,16,32}; {1,2,3}).

Alternatives rejected: list only in the API arm (reproduces B1); no list anywhere (weaker than any deployed system, breaks label-space consistency).

## 2. Pools, validation, test, retriever (D2)

- **Splits [V in code]:** validation = 1,000 train rows (`VAL_SEED` 42), fixed, never in a pool; pool order = remainder (9,003 rows for BANKING77); pools are prefixes, so nested across N within a replicate; a fresh order per `pool_seed`. Test: only `scripts/freeze_test_split.py` reads it; the fingerprint file `results/test_split_fingerprint.json` **does not exist yet**.
- **Sampling decision [R]: class-stratified (round-robin) primary, random as sensitivity.** Evidence [V, from HF statistics]: with random pools, the share of test items whose class is absent from the pool is BANKING77 50.5% (N=50), 26.1% (100), 7.5% (200), 0.4% (500); LEDGAR 48.6%, 29.1%, 13.3% (200), 2.4% (500), 0.4% (1,000). A fine-tuned model cannot predict an absent class, so at N≤200 random pools measure class coverage (a ceiling of ≈92% at N=200), not the value of labels; stratified pools guarantee floor(N/C) per class and have lower pool variance. Cost: the pool prior becomes uniform (LEDGAR train counts run 23–3,167), which can lower accuracy slightly and raises macro-F1; this is stated and checked by random-pool runs at N=200 and 1,000 on both tasks (R=3, both arms). Alternatives: random only (draft default; fails the evidence above); proportional with per-class floor (nesting harder to prove; deferred).
- **N < C is dropped.** N=50 (<77 classes) leaves ≥27 classes absent even when stratified. Use N=77 (one per class) as the exploratory floor.
- **Near-duplicates [V measured]:** with bge-small, the median validation item has cosine 0.918 to its nearest N=1,000 pool item and 21% exceed 0.95. BANKING77 is paraphrase-rich, so removing near-duplicates would change the task. Decision: do not delete; compute, per test item, the maximum cosine to the pool (bge) and, for BillSum, 5-gram shingle Jaccard (bills are reintroduced across sessions) [R]; flag exact normalised duplicates and Jaccard ≥0.8; report accuracy by similarity bin and a sensitivity analysis without flagged items, for both arms.
- **Retriever decision [R, with CPU evidence on validation, 3 pools]:** `BAAI/bge-small-en-v1.5` (MIT) primary. A plain unweighted vote over neighbours: bge 1-NN accuracy 0.855 (N=1,000), 0.916 (N=5,000); my quick BM25 (no stop-word list, so a lower bound) 0.58 and 0.73. Revision must be pinned (the harness leaves `revision=None`; A3-04). kNN baseline = best of k∈{1,5,16} chosen on validation; it is *not* trivial: at N=5,000 it is as strong as the pilot's fine-tuned models (0.89–0.90, old prompt), so it must be reported next to every API-retrieval number. k=32 votes with unweighted counts collapse at small N (0.46 at N=200), a reason to use similarity-weighted or 1-NN. BillSum: bge sees only the first 512 tokens of a ≈1.9k-token bill [V, harness docstring]; choose between bge-head, BM25 on full text and TF-IDF by the ROUGE-L between retrieved and gold summaries on 300 validation items, before any test read [R, CPU-only test].
- **Test isolation [R]:** a single `final_eval` entry point refuses to run unless (a) the pre-registration tag exists, (b) the config hash is in a committed `final_configs.json`, (c) it appends (config hash, test hash, UTC time, git commit) to an append-only `test_reads.jsonl`. Re-reads only after a crash, same hash. Everything before uses validation only.

## 3. N levels per dataset (D3)

| Dataset | C | N confirmatory | Exploratory | Eval items | Why |
|---|---|---|---|---|---|
| BANKING77 | 77 | 200, 1,000, 5,000 | 77 (1/class), 9,003 (all) | 3,080 | ≥2.6 per class at 200; 5,000 is 55% of the 9,003 available |
| LEDGAR (LexGLUE) | 100 | 200, 1,000, 5,000 | 100 (1/class) | seeded 3,000 of 10,000 | meaningful from N=200 (2/class); 3 classes have <50 train items |
| BillSum US | – | 200, 1,000, 5,000 | 50 for ICL | seeded 1,000 of 3,269 | style is learned early; 5,000 tests whether FT keeps improving |

LEDGAR facts [V, statistics endpoint]: 100 labels, train 60,000 (min 23 "Books", median 426, max 3,167), no class below 20 in train, test min 2 ("Books"), 3 test classes under 10 items. The "rare classes" worry is smaller than assumed: the small-N problem is coverage under random sampling and unstable per-class F1, so accuracy is primary, macro-F1 descriptive. Majority-class share 5.8%.

**BillSum** [V, 1,000 rows: 700 train, 300 test; real tokenizers]. Document tokens (Qwen / Phi): p5 1,077/1,062; median 1,915/1,871; p90 3,504/3,406; p99 4,398/4,247; max 7,393/7,392; mean ≈2.1k; chars/token 4.9–5.0. Summary tokens: median ≈200, p99 ≈760, max 983. Nothing exceeds 8k tokens (consistent with the original 5k–20k character filter, [U]). Decisions [R]: no truncation of inputs for any arm; `max_length` 8,192 for training; output cap **1,024** for every arm (API template currently 2,048, harness must match: A3-06); k_max=3 (median prompt ≈8.5k tokens, p90 ≈15k; k=5 gives 12.8k/23k and a 5× API bill for little expected gain [U]); eval on 1,000 seeded items. Note the harness assumes chars/4, which overestimates Qwen/Phi counts by ≈20% (A3-10). Classification assumptions do not carry over (no label list or coverage, noisy item-level ROUGE-L/BERTScore instead of 0/1, 5–10× longer sequences), so BillSum tuning uses epochs {2,3,5} and N=5,000 tunes only lr {1e-4, 2e-4} (A5 to cost).

## 4. Replicates (D4)

**Design [R]:** replicate r=1..R has `pool_seed=r`, `train seed=100+r`; every method of a replicate shares the pool (the API arm gets the same pool file). Tuning uses a separate replicate 0 (pool 0, seed 0) that is never analysed. **R=5 minimum (classification), R=3 for BillSum; recommended R=8 / 5.** A variance-component substudy (BANKING77, Qwen3-4B, N=1,000, 5 pools × 3 train seeds, +10 runs) separates pool from training variance. Assumptions given to A4 [pilot + my kNN runs, with the old prompt]: training-seed SD 0.004–0.02 (N=1,000); pool-to-pool SD of a retrieval-only system 0.01–0.02 at N≤1,000, 0.002–0.005 at N=5,000; the API arm is stochastic only through provider nondeterminism. **Caveat [V]:** with 9,003 usable rows, two N=5,000 pools share ≈55% of items, so pool variance at N=5,000 is understated by a finite-population factor ≈0.44 in variance. Do not call those replicates independent there (A3-07).

## 5. Hyperparameters and epochs (D5)

[V in code] `S_min=200`, effective batch `min(16, ceil(N/20))` (3 at N=50, 10 at 200), epoch grid {3,5,10,20,40} with patience 2 and step-count de-duplication, lr grid {5e-5,1e-4,2e-4,4e-4}; rsLoRA grid = base grid / √r = /4 (α/√r = 8 vs α/r = 2, so the initial update scale matches; I confirmed `lr_grid` does this). r=16, α=32, dropout 0.05, AdamW, cosine, warm-up 3%, weight decay 0 locked.

**Amendments [R].** (1) Tune once per (model, technique, N) on replicate 0, then freeze for replicates 1..R; the per-replicate `--tune` in the code would multiply cost by R and inject selection noise into the very contrasts we test. (2) Select on **validation accuracy** for classification (generation over 1,000 items is seconds), validation loss only as tie-break and for BillSum; the current code returns loss only (A3-05), and loss usually prefers fewer epochs than accuracy does [U]. (3) Cap the epoch grid by cost: N=5,000 ≤10 epochs. (4) ICL k and retriever are tuned on the same validation set (k grid {4,8,16,32}); one prompt text for all arms, not tuned (limitation; sensitivity with one paraphrased instruction at N=1,000). **Equal effort:** per (system, N) 12 ± 3 validation evaluations for FT (4 lr × ≤5 epochs, early stop) vs 4 for ICL; report evaluations, GPU-hours and dollars rather than claiming equality. **Disclosure:** 1,000 labelled validation rows exceed N at N≤200; "N" counts pool items only (the true-few-shot critique, A3-03). Sensitivity: select on a validation subset of size N at N=200.

## 6. API protocol (D6)

[V] `configs/api_models.yaml`: `claude-haiku-5-5`, `gpt-6-luna`, `gemini-3.1-flash-lite`, prices read 2026-11-21, thinking disabled (Haiku), `effort none` (Luna), `minimal` (Gemini, cannot be off). **Temperature is sent only for Gemini (0). Haiku and Luna run at provider defaults**, so "temperature 0 everywhere" (stated in `task_spec`) is false for two of three APIs (A3-01) [V by reading `clients.py`].

Rules [R]: (1) record `model_id_returned`, run date, mode, price read date, billed input/output/reasoning tokens per record (already in `extra`); compute cost from billed tokens after the fact. (2) One mode (batch, 50% price) for every confirmatory run; sync-vs-batch agreement checked on 100 items. (3) Self-agreement: after replicate 1 of each (provider, N), re-run a seeded 10% subset twice; report per-item flip rate; **if flip rate >2%, every API cell is averaged over two full runs** (cost ×2, stated in advance). (4) All confirmatory runs for one provider within 72 h; re-check price and id page that day; a changed `model_id_returned` voids the batch. (5) Gemini counts thinking tokens against `maxOutputTokens` [U]: a 32-token cap may truncate; log `finish_reason` and treat MAX_TOKENS as invalid, after the live test in §11.

**Staging [R]:** S0 dry-run cost gate (exists); S1 smoke 20 items × provider × {zero, retrieved} (≈$0.01); S2 200 validation items; S3 validation tuning of k; S4 test, only after the pre-registration tag. **Request counts (BANKING77, per provider):** zero-shot 3,080; retrieved 3 N × 5 reps × 3,080 = 46,200; random-k same; validation k-tuning 3 × 4 × 1,000 = 12,000; repeats ≈1,850; total ≈110k, 330k for three providers (≈1.3k tokens each ⇒ ≈430M tokens, A5 to price). BillSum (R=3, 1,000 items, k=3): ≈14k per provider, ≈120M tokens each. LEDGAR (3,000 items, R=5): ≈100k per provider.

## 7. Parsing, invalid outputs, generation vs head (D7)

Primary protocol [R]: free generation, first non-empty line, case/separator-insensitive exact match to a label, no fuzzy matching; **invalid = wrong in primary accuracy**, always reported as a separate rate per arm × N. Required reporting so that arms with different output behaviour are not compared silently: (a) accuracy (strict), (b) invalid rate, (c) accuracy among valid outputs, (d) for SLMs, accuracy under constrained decoding / label-likelihood scoring (the exploratory robustness check; APIs expose no verified logprobs [U], so no API equivalent), (e) for the bridge to Bertsch et al., one classification-head fine-tune (Qwen3-4B, BANKING77, N=1,000, R=3). (d) should differ from (a) mostly at N≤200 [U]. An edit-distance "repair" metric is a labelled sensitivity, never primary. [V] `reverted_card_payment?` is the only label with trailing punctuation outside the stripped set: a model emitting `reverted_card_payment` is scored invalid (7 of 18,127 pilot invalid outputs). Strip `?` on both sides in the normaliser, keeping the collision check (A3-02).

## 8. Contamination (D8)

BANKING77 (2020), LEDGAR (2020, LexGLUE 2021) and BillSum (2019) are public; no model's training data is documented to exclude them [U, per verification]. Reported: the dataset first-public dates; the zero-shot base accuracy of each model (a high value at zero shot is a flag, not proof); an exact-prefix continuation probe on 300 test texts per dataset for the open models (needs GPU, descriptive); no claim of "no contamination". Optional fresh-data probe [U, licence and availability to verify]: BillSum-style items from 2025–26 bills with their CRS summaries (US-government works), after all model cutoffs. Per-item OLMo overlap (H3) stays exploratory with a placebo stratification.

## 9. Experiment matrix and run counts

Models M=2 (Qwen3-4B, Phi-4-mini; OLMo-3-7B conditional, +50%). R=5 (BillSum 3). "Final" = test-evaluated runs.

| Block | Min viable | Recommended |
|---|---|---|
| FT finals, 3 tasks (LEDGAR conditional) | 2×3×(5+3+5)=78 | 2×3×(8+5+8)=126 |
| FT tuning (rep 0, once per model×N×task) | 18 tuning cells × ≈12 trials = 216 trials | same, + LEDGAR random-pool sensitivity |
| Technique ablation (B77, N=1,000, lr-only grid 4 + 3 reps, QLoRA/DoRA/rsLoRA) | none (exploratory, dropped) | 3×2×7 = 42 |
| Variance-component substudy | – | +10 |
| Head-based FT bridge | – | 3 |
| SLM base ICL (zero, random-k, retrieved-k), per model | 2×(3 tasks × 3N × 3 reps × 2 modes) + zero + val-k | + R as above |
| kNN baseline | free | free |
| API runs (provider × task × N × rep × {retrieved, random-k}) + zero-shot + val-k + repeats | ≈180k requests | ≈460k requests (R=8/5) |

FT finals are 78 / 126 trainings and ≈216 tuning trials; every final is followed by one test evaluation. GPU-hours come from A5 after the loss/throughput test (§11).

## 10. Quality controls and logging

**Controls [R].** Unit test: every gold label of every task parses to itself, no collisions (exists for B77; add LEDGAR 100 labels, verified collision-free [V]). Test-set and pool hashes in every record (exist). Re-tokenise-and-assert max length and non-empty completions. Identical-prompt test between SLM and API arms on a 20-item sample for every task and technique, comparing rendered strings. Fail loudly on non-finite loss (exists). Pilot results never pooled with confirmatory; failed runs stay in the table.

**Per-prediction record** (extends the eval schema): run_id, system, arm, technique, task, n_budget, k, replicate, pool_seed, train_seed, item_id, prediction, parsed, gold, parse_ok, invalid_reason, input/output/reasoning tokens, latency, finish_reason, shot_ids, max_cosine_to_pool, model_id_returned, run_date. **Per-run record:** config hash, git commit (the pilot logged `"unknown"`: A3-08), libs, GPU, seeds, lr/epochs/steps/selection trials, first/last loss, val hash, pool hash, test-read ledger id, price date, wall-clock, energy.

## 11. Needs a GPU or live API (smallest controlled test)

1. First-step loss (I-14): `scripts/diagnose_loss.py`, Phi/Qwen/OLMo LoRA and Phi QLoRA, 16 examples, new prompt. My hypothesis [U]: 16–19 nats came from a confident chat-style prior on label tokens without the label list; expect low single digits now.
2. OLMo-3 DoRA NaN: same script with `--fp32` and zero-row-norm check; the code now stops on non-finite loss, but references a missing `scripts/diagnose_nan.py` (A3-09).
3. Under-training fix: Qwen3-4B LoRA N=200, replicate 0, tuned, validation accuracy and invalid rate (pilot: accuracy 0.17, 74% invalid).
4. Throughput and memory: BillSum 2.6k–8k tokens, micro-batch 1–4, one model, 20 steps.
5. SLM ICL harness: 100 validation items, base Qwen, zero and retrieved k=8, rendered strings equal to the API's.
6. Live API: 20 items per provider, one call each with temperature variations and logprob flags (does Haiku 5.5/Luna accept `temperature`? logprobs?); Gemini `finish_reason` at max 32 with minimal thinking; flip rate on 50 items ×3.
7. `count_tokens` vs the chars/4 estimate on 50 BillSum prompts per provider.
8. Constrained decoding / label scoring in vLLM for 77 labels, 100 items, 1 model.

## 12. New issues (prefix A3-)

| ID | Issue | Sev |
|---|---|---|
| A3-01 | Haiku/Luna called without temperature; "temperature 0 for every arm" untrue; nondeterminism unquantified | HIGH |
| A3-02 | Normaliser keeps `?`, so `reverted_card_payment` is invalid (pilot count small) | LOW |
| A3-03 | 1,000 validation labels exceed N at N≤200 (true few-shot critique) | MEDIUM |
| A3-04 | Missing code: SLM ICL harness, shared few-shot rendering in `task_spec`, kNN baseline, BillSum/LEDGAR data and training, test-read ledger, API repeats, replicate id in API records, embedder revision pin | HIGH |
| A3-05 | Tuning selects by validation loss, runs per replicate, epoch grid too costly at N=5,000 | MEDIUM |
| A3-06 | Output caps differ (BillSum API 2,048 vs shared 1,024; Gemini thinking inside the cap) | MEDIUM |
| A3-07 | N=5,000 pools overlap ≈55%: pool variance understated | MEDIUM |
| A3-08 | Pilot records `git_commit: unknown`; cross-branch prompt test is skipped on the training branch | LOW |
| A3-09 | Missing `scripts/diagnose_nan.py` | LOW |
| A3-10 | API cost estimate uses chars/4 (measured 4.9–5.0 chars/token for Qwen/Phi on BillSum) | LOW |
| A3-11 | The kNN baseline is as strong as the fine-tuned pilot at N=5,000; the paper must show it | MEDIUM |

## 13. Register issues, assessed against the code

- **I-01 [V]:** fixed for BANKING77 zero-shot (byte-identical prompts, same parser). Not fixed for few-shot rendering (only the API builds it), the SLM ICL arm, or other tasks. Keep FIXED-UNVERIFIED until step 3 shows the unparseable rate falls.
- **I-02:** coded, no GPU test; amended in §5 (per-replicate tuning, loss-based selection). FIXED-UNVERIFIED, protocol amended.
- **I-03 [V]:** case bug fixed and tested; residual `?` case (A3-02). RESOLVED for the case issue.
- **I-04:** `pool_seed` replicates exist in code; API records carry no replicate id; R≥5 proposed with caveats. OPEN→IN-REVIEW pending A4.
- **I-08:** validation carve-out and freeze script exist; no fingerprint, ledger or near-duplicate code. OPEN; protocol in §2.
- **I-12:** analysed (§3): 100 labels, min train 23, test min 2; stratified pools; N from 200. IN-REVIEW.
- **I-14:** cannot be settled offline; test in §11. OPEN.
- **I-16:** decided in §7: generation primary, strict + valid-only + constrained + one head bridge. IN-REVIEW.

Alternatives for each decision are listed inline (D1–D8) for the decision log.


## Round 2: reconciled grid (2026-11-21)

Sources: 04, 05, 02, 01. Costs are my arithmetic from A5's per-request prices; A5 to re-price.

### 1. Test sizes
Power depends on n only up to ≈3,000 items (A4). At N=1,000 and a 3-point gap, R=5, first-step α 0.025/0.0083: n=1,000 gives .75/.60, n=3,080 gives .96/.89. So **BANKING77: all 3,080 items for every arm** (API cost is small: ≈1.0–1.2k tokens per request). **LEDGAR: one seeded subset of 3,000 of 10,000** for every arm in every contrast. **BillSum (US test 3,269): one seeded subset of 1,000** for every arm. The metric is continuous: with an assumed paired ROUGE-L SD of 0.06 [U] the item SE is 0.19 points at n=1,000, the replicate SE is 0.4/√5=0.18, so at R=5 the minimal detectable gap is ≈4.6×√(0.19²+0.18²) ≈ 1.2 points (n=500: 1.5; n=3,269: 1.0).
**Free arms** (kNN, SLM, fine-tuned) are also scored on the full test as a descriptive extra; every paired contrast and cost-per-correct figure uses the identical seeded items (seed and hash fixed in the pre-registration).
**Near duplicates (A2):** BillSum 297/3,269 (9.1%) at cosine ≥0.95, 16.9% at ≥0.9; LEDGAR 338 exact (3.4%), 29.2% at ≥0.9; BANKING77 11 exact, 13.5% at ≥0.9. These are against the *whole* train set; against a pool of N they are fewer. They stay in the primary analysis (paraphrase is part of deployment). The novel-text stratum is a sensitivity analysis with effective n ≈ 2,660 (B77), 2,120 (LEDGAR), 830–910 (BillSum): negligible power loss for B77, BillSum MDD ≈1.3 points.

### 2. Replicates
Order runs replicate-major (all methods of replicate 1, then 2, …) so blinded variance re-estimation (A4) is possible: after R=5, estimate variance components only; if power for δ=3 points is <80%, extend to R=8 (B77) or 10 (cap). **B77: R=5 (minimum viable), 8 (recommended). BillSum: N=1,000 R=5 in both grids; N=5,000 R=3 (minimum viable; exploratory, not in the confirmatory family, because at R=3 the t₂ critical value ≈10 gives MDD ≈2.3 ROUGE points) and R=5 in the recommended grid if the §11 throughput benchmark gives ≤2 h per run; N=200 R=3. LEDGAR (extension): R=5.** Confirmatory family therefore: B77 N=1,000 and 5,000, BillSum N=1,000, plus LEDGAR N=1,000 and 5,000 if included: 3 or 5 Holm hypotheses.
**N=5,000 overlap (A3-07):** two pools of 5,000 from 9,003 rows share 5,000²/9,003 ≈ 2,780 items (55%); the pool-effect variance is a finite-population variance, ≈(1−5,000/9,003)=0.44 of the infinite-population value (SD ×0.67). Decision: keep independent permutations, state that the N=5,000 estimand is conditional on these 9,003 training rows, and label the CI accordingly; the crossed-model CI stays valid for it. N=9,003 (all data) is an exploratory ceiling point.

### 3. Final values

| | BANKING77 | BillSum (US) | LEDGAR (extension) |
|---|---|---|---|
| Test items / arm | 3,080 | 1,000 seeded | 3,000 seeded |
| N, minimum viable | 200, 1,000, 5,000 | 200, 1,000, 5,000 | 200, 1,000, 5,000 |
| N, recommended | 77, 200, 1,000, 5,000 | same | same |
| k_max (grid) | 32 ({8,32} / {4,8,16,32}) | 3 ({1,3} / {1,2,3}) | 32 |
| Models | Qwen, Phi (rec.: + OLMo) | Qwen, Phi | Qwen, Phi |
| R (N=77/200/1k/5k), min | –/5/5/5 | –/3/5/3 | –/5/5/5 |
| R, rec. | 3/8/8/8 | –/3/5/5 | as min |
| FT finals, min / rec. | 30 / 54 (+15 OLMo) | 22 / 26 | 30 |
| FT tuning trials (rep. 0), min / rec. | 44 / 64 (+22 OLMo) | 16 / 16 | 44 |
| SLM ICL test passes, min / rec. | 38 / 111 | 30 / 34 | 38 |
| SLM ICL validation passes | 24 / 48 | 18 / 18 | 24 |
| API requests per provider, min / rec. | 66,368 / 135,504 | 10,200 / 18,500 | 64,800 |

Formulas. FT finals = Σ_models Σ_N R_N; tuning = per model 10 trials at each N≤1,000 (lr ×4, epochs {3,5,10}, step-deduplicated, early stopping) + 2 at N=5,000 (lr fixed from N=1,000; BillSum: 4 trials at N≤1,000, none at 5,000). Each classification tuning trial is selected on validation *accuracy*. Run counts: minimum viable fine-tune total = 52 finals + 60 tuning = **112**; recommended = 95 finals + 102 tuning + 42 technique ablation (3 techniques × 2 models × (4 lr + 3 reps)) + 10 variance substudy + 3 head = **252**. SLM ICL test passes per model = Σ_N R_N (retrieved) + random-k (N=1,000 and 5,000 for B77 at R=3 in rec.; N=1,000 R=3 in min) + 1 zero-shot. API requests per provider = n_test × (Σ_N R_N,ret + Σ R_rand + z) + n_val × (|k grid| × |N|) + 0.2 × n_test × |N| (10% repeat, 2 reruns, replicate 1). B77 min: 3,080×(15+3+1)+6,000+1,848 = 66,368. B77 rec.: 3,080×(27+9+2)+16,000+2,464 = 135,504. BillSum min: 1,000×(7+1+1)+600+600 = 10,200 (retrieved: N=200 R=1, N=1,000 R=3, N=5,000 R=3); rec.: 1,000×(12+3+2)+900+600 = 18,500. **Totals, three providers:** min B77+BillSum 229,704 requests; rec. 462,012; LEDGAR +194,400. Indicative cost (standard price): B77 ≈$0.00066 per request summed over three providers ⇒ $44 (min) / $89 (rec.); BillSum ≈ $0.0055 per k=3 request ⇒ ≈$41 / ≈$75; so min ≈ $85, rec. ≈ $165 standard (≈$63 / $120 with batch on B77 and sync on BillSum); LEDGAR ≈ +$56. Recommended exceeds A5's (b) (n_test 3,080, R=8).

### 4. Answers to A5 and A4 items
- **A5-01.** Correction: `tokenize_example` *raises* when an example exceeds `max_length`, so there is no silent truncation, but BillSum would crash on the first bill. Rule: tokenise all 18,949 train bills once on CPU (pre-run check), set `max_length` per task to the maximum plus margin (expected ≈8–9k), never truncate test items; drop train examples above the cap only if <0.1% and log them.
- **A5-02.** `tune()` and `select_trials()` exist in `src/train/train.py` (tests pass); A5 looked for a module. They are untested on GPU and need the §5 amendments: tune on replicate 0 only, accuracy-based selection for classification, the restricted grids above.
- **A5-03 (rewritten API rule).** "Decoding is the most deterministic setting each interface allows: open models greedy; Gemini temperature 0; Haiku 5.5 no sampling parameter (400 otherwise); Luna temperature only if the live test shows it is accepted. Arms at provider-default sampling are labelled *stochastic*, their flip rate is measured on a 10% repeat, and if it exceeds 2% the arm's accuracy is the mean of two full runs." `task_spec.TEMPERATURE` becomes a per-arm record, not a universal claim. Sensitivity: one SLM retrieval cell at temperature 1.0.
- **A5-04.** BillSum fine-tuning: micro-batch 1, accumulation 16, gradient checkpointing (not implemented in `train.py`: code change), loss/LM head over completion positions only (`logits_to_keep`, valid at micro-batch 1 because there is no padding), eval batch ≤8 sorted by length. First GPU step: 20 steps on the longest 1% of train bills, record peak memory; fallback chain: micro-batch 1 + GC → drop the longest 0.1% of train examples (logged).
- **A5-05.** The cost driver is k × (pools × N levels) × n_test. Decision: k_max=3, API retrieval only at N=1,000 and 5,000 for confirmatory (R=3/5) plus N=200 at R=1 (2 in rec.), 1,000 items, sync mode, no per-item truncation of shots.
- **A4-08.** Arm freeze: on replicate 0 (validation only) choose per (task, N): the best FT configuration, per provider the best (k, retriever, technique), and the best provider; write `frozen_arms.json`, commit its hash with the pre-registration tag; replicates 1..R are never re-selected.

### 5. Re-checks against new evidence
- **A4-03.** My protocol never relied on `hierarchical_paired_bootstrap`. Requirement added: every arm (kNN and API included) writes per-item records with `replicate`, `pool_seed`, `item_id`, so A4's crossed model can run; the bootstrap is sensitivity only at R≥8.
- **Spiegelhalter et al. (abstract only [U]): up to 200 labels, crossover ≈50.** My lowest confirmatory level (200) sits at the end of that range. Decision: keep N=77 (1 per class) and 200 as estimation points on all datasets (R as in the table), keep the confirmatory family at N=1,000 and 5,000, and pre-register the *bracket* statement across all N from CIs. Small-N crossover evidence is therefore exploratory; read Spiegelhalter in full first (A1-01).
