# Experiment protocol

Assembled from `agents/03-methodology.md` (A3 body, Round 2, **Round 2b governs**) and `DECISION_LOG.md`. Labels: [V] verified (code read, offline test, measurement), [R] recommendation, [U] unverified. Everything here is a plan; the fine-tuned arm is **unproven under the corrected prompt** until the gated pilot (section 8) passes.

## 1. Information equivalence (D1)
One shared `task_spec` per task defines system text (task sentence + full label list), user turn (`Input: ...\nLabel:`), output cap, decoding and parser. Arms differ only in what they do with the pool of N labelled items: zero-shot (spec only); random-k / retrieved-k (k pool items in the user turn, nearest last, same rendering) for SLM and API; kNN (label of the most similar pool item(s), no LLM); LoRA (trained on the pool with the zero-shot spec as prompt). [V] the zero-shot SLM and API prompts are byte-identical in a scratch merge of the branches. Equal information means same task text, label list, pool and test items, not equal bits used (ICL sees at most k items). A many-shot probe (k=128) at N=1,000 and 3,000 on the cheapest API and one SLM bounds the k cap. Validation labels are used for selection only, never for training or retrieval. k_max: 32 for BANKING77, 8 for LEDGAR, 3 for BillSum (grids {4,8,16,32}; {1,2,4,8}; {1,2,3}).

## 2. Splits, pools, retriever (D2, L-04)
- Validation = 1,000 fixed train rows (`VAL_SEED` 42), never in a pool; pool = remainder (9,003 BANKING77 rows); pools are prefixes of a per-replicate permutation (nested across N), fresh order per `pool_seed`.
- **Random pools for N >= 1,000; class-stratified round-robin only at N=200** (BANKING77, LEDGAR; BillSum random throughout), a separate named condition never pooled with random cells; random-vs-stratified table at N=200, R=3.
- Overlap: two N=3,000 pools from 9,003 rows share about 33% (finite-population factor 0.67 on pool variance); BillSum about 17%. The estimand is stated as conditional on the finite training set (statistics plan).
- Retriever: `BAAI/bge-small-en-v1.5` (MIT), revision pinned (the harness leaves `revision=None`: code change). kNN = best of k in {1,5,16} on val(N), similarity-weighted; it is as strong as the pilot's fine-tuned models at N=5,000 (0.89-0.90, old prompt) and is reported next to every API-retrieval number. BillSum: bge sees only 512 tokens of a ~1.9k-token bill; choose bge-head / BM25 / TF-IDF by ROUGE-L of retrieved vs gold summaries on 300 validation items before any test read.
- Near-duplicates: kept; per test item max cosine to pool (and 5-gram Jaccard for BillSum) stored; reported by bin plus a novel-text sensitivity.
- **Test isolation:** a single `final_eval` entry refuses to run unless the pre-registration tag exists, the config hash is in a committed `final_configs.json`, and it appends (config hash, test hash, UTC, git commit) to an append-only `test_reads.jsonl`. Everything before uses validation only. `results/test_split_fingerprint.json` does not exist yet.

## 3. Grid (Round 2b)
| | BANKING77 | BillSum | LEDGAR (exploratory) |
|---|---|---|---|
| Test items | 3,080 | 1,000 seeded | 3,000 seeded |
| N / R | 200/8 (+3 random), 1k/8, 3k/8 | 200, 1k, 3k / 5 | 200, 1k, 3k / 3 |
| FT tuning trials | 2 models x 3 N x 4 = 24 | 24 | 12 (Qwen only) |
| FT finals | 54 | 30 | 9 |
| SLM ICL test passes | 62 | 38 | 26 |
| API requests, selected provider | 106,128 | 23,200 | about 39,000 |
| API requests, each other provider | 50,688 | 14,200 | not run |

Models: Qwen3-4B-Instruct-2507, Phi-4-mini-instruct; OLMo-3-7B only if time remains. API random-k only at N=1,000, R=3. About 259,104 API requests across three providers (BANKING77 207,504; BillSum 51,600); A5's B77 count differs by 0.15% (extra-cell set), but **BillSum counts disagree** (A3 23,200/14,200 vs A5 28,400/22,400; I-29): one committed `count_requests.py` must produce the counts per provider and cell before the budget is frozen. N=77/100 one-per-class: ICL and kNN only. Replicate r has `pool_seed=r`, `train_seed=100+r`; all methods of a replicate share the pool (the API arm receives the same pool file); run **replicate-major**. Tuning uses replicate 0 (never analysed).

## 4. Tuning (D5, L-05): equal effort, 4 trials per system
- **Fine-tuning:** LoRA r=16, alpha=32, dropout 0.05, AdamW, cosine, warm-up 3%, wd 0; 4 learning rates {5e-5, 1e-4, 2e-4, 4e-4} (rsLoRA grid /4); each trial is one cosine schedule to E_max (N=200: 10; N=1,000: 10; N=3,000: 5; S_min 200 floor) with validation accuracy (loss tie-break; BillSum: loss) at five checkpoints, early stopping patience 2. Chosen (lr, E*) is frozen for replicates 1..R. Tuning on replicate 0 only, per model x N.
- **ICL:** 4 values of k plus (BillSum) the better of two retrievers; prompt text not tuned (limitation; one paraphrase sensitivity at N=1,000).
- **Validation size val(N)=min(1,000, N)**; sensitivity with the full 1,000. Tuning GPU-hours and dollars are charged in cost-per-correct.
- **Tuning bar (L-14, resolved by L-19):** 4 trials per system equals count, not compute (an FT trial is a full training with 5 validation passes, 1.5-2x a final run). Spiegelhalter et al. use 27 configurations x 5 folds. Pre-registered sensitivity inside the F3 pilot window: BANKING77, Qwen, replicate-0 pool, N=200 and 1,000, a 12-configuration FT grid (4 lr x 3 rank/alpha or dropout/weight-decay pairs; about 0.7-2.0 GPU-h, budgeted in FEASIBILITY) and a matched ICL grid (k x retriever x one paraphrased instruction, under $1). If best-of-12 beats best-of-4 by less than m (3 points) on a held-out slice the 4-trial rule is reported as adequate; if by more, the campaign grids are widened at N<=1,000 only (BillSum N=3,000 stays at cut 1). Edge rule: if the chosen value lies on a grid edge, extend by one value on that side, for both arms.
- **Code prerequisite (A5-12):** `select_trials` retrains from scratch per (lr, epochs) pair (at least 18 epochs per lr); in-run early stopping must replace it, otherwise BillSum tuning costs 6-43 GPU-h per model.
- **Arm freeze (A4-08):** on replicate 0 choose per (task, N) the best FT configuration, per provider the best (k, retriever), and the best provider; write `frozen_arms.json`, commit its hash with a **second tag at the end of week 4** (protocol tag in week 3, arms tag after BillSum tuning); replicates 1..R are never re-selected. Provider rule: most accurate on replicate-0 validation; ties within 1 SE go to the cheaper provider; all providers' cost per correct are reported descriptively.

## 5. API protocol (D6, Round 2b section 2): set by a live test, not by documentation
Facts [V]: ids `claude-haiku-5-5`, `gpt-6-luna`, `gemini-3.1-flash-lite`; the Haiku migration page says to omit `temperature`/`top_p`/`top_k` (the red team could not confirm "returns a 400" or `thinking: {type: disabled}` on that page; A5 read them differently, so the live test decides); thinking counts toward `max_tokens`; Gemini thinking cannot be fully off.
1. **Sampling:** send `temperature=0` only where a 50-item test shows it succeeds; otherwise omit, label the arm *stochastic*, measure flip rate on a 10% repeat x2; if above 2% the arm's accuracy is the mean of two full runs (pre-priced: about 49,000 extra BANKING77 requests, about 19,000 BillSum). `TEMPERATURE` becomes a per-arm field.
2. **Reasoning:** lowest accepted setting; *reasoning-off* only if reported reasoning tokens are 0 for every test item, else *reasoning-minimal* with billed reasoning tokens in the cost; never compared silently.
3. **Cap:** if reasoning counts against the cap, `max_output_tokens = 32 + max(64, p99.9 observed reasoning tokens)`, else 32; truncation counts as invalid and a rate above 1% in the smoke test stops the run. BillSum output cap 1,024 for every arm (the API template's 2,048 must change).
4. Record `model_id_returned`, usage fields, price date; all confirmatory cells of one provider within 72 h (batch B77 needs 7-13 sequential waves at a 24 h target for Luna/Gemini, so the 72 h rule is met by the **sync fallback**, priced in FEASIBILITY); a changed returned id voids the batch; one mode (sync or batch) per confirmatory run, agreement checked on 100 items.
5. All arms are "deployed generative systems" (free generation); constrained decoding is reported for SLM arms only, labelled.
Staging: S0 dry-run cost gate; S1 smoke 20 items; S2 200 validation items; S3 validation tuning of k; S4 test after the tag.

## 6. Parsing and invalid outputs (D7)
Free generation, first non-empty line, case/separator-insensitive exact match to a label, no fuzzy matching; **invalid = wrong**; always report invalid rate, accuracy among valid outputs, and (SLM only) constrained-decoding accuracy as a robustness check. One head-based fine-tune (Qwen, BANKING77, N=1,000, R=3) bridges to Bertsch et al. Fix the `?` normaliser (`reverted_card_payment?`) with a gold-label self-parse unit test (all 77 + 100 labels parse to themselves, collision-free [V]).

## 7. Controls and logging
Gold-label parse test; pool and test hashes in every record; max-length assertion (tokenise all 18,949 BillSum train bills once; `tokenize_example` raises rather than truncates; set `max_length` per task, never truncate test items; drop train examples above the cap only if under 0.1%, logged); identical rendered prompt between SLM and API on 20 items per task and technique; fail on non-finite loss; pilot never pooled; failed runs kept. Per-prediction record: run_id, system, arm, task, n_budget, k, replicate, pool_seed, train_seed, item_id, prediction, parsed, gold, parse_ok, invalid_reason, tokens (input/output/reasoning), latency, finish_reason, shot_ids, max_cosine_to_pool, model_id_returned, run_date. Per-run record: config hash, git commit (the pilot logged "unknown"), libraries, GPU, seeds, tuning trials, first/last loss, hashes, test-read ledger id, price date, wall-clock, energy.
Server environment: torch 2.9.0+cu128, transformers 4.57.3, peft 0.18.0, trl 0.26.2, vLLM 0.13.0; `requirements/train.txt` on the pilot branch pins newer, never-installed versions: do not run the bootstrap script blindly. One engine (vLLM) for all evaluation; DoRA must be merged for serving.

## 8. Gated pilot (L-09; exploratory, validation only, test never read)
**Stage 0 (no GPU/API) blockers:** (a) three branches merged to one tagged commit, `git_commit` logged, prompt-equality test unskipped and passing; (b) `?` normaliser fix; (c) tuned-procedure code (checkpoint-epoch evaluation, 4-lr tuner, val(N), replicate-0 freeze, E*); (d) kNN baseline on pinned bge-small; (e) FT-vs-API rendered strings identical on 20 items.
**Stage 1 (about 0.5 GPU-h) diagnostics:** `diagnose_loss.py`, 16 examples, new prompt, Qwen/Phi (LoRA fp32, QLoRA), OLMo (DoRA, LoRA). Pass: finite losses; first-step completion-token loss at most 3.0 (Qwen), 6.0 (Phi), always below ln|V| (about 12); Phi bf16 vs fp32 within 5%, QLoRA within 20%; OLMo DoRA finite for 20 steps after a stated fix, else OLMo dropped and the failure reported.
**Stage 2 (about 1.5-2.5 GPU-h) Qwen3-4B, BANKING77,** N=200 (stratified) and N=1,000 (random); replicate 0 tunes 8 trials, replicates 1-2 are finals (+1 random-pool N=200 final). Gates are evaluated on validation rows **not used for selection** (06c R1: rows 201-1,000 at N=200; a fresh 500-row slice at N=1,000), because the same val(N) rows select lr and checkpoint in this stage and N=200 selection has binomial SE about 3 points. No pilot pass is evidence for any hypothesis (including H1); G2/G3 are sanity gates only:

| Gate | Threshold |
|---|---|
| G1 invalid output | mean under 5% at both N, no run over 8% |
| G2 vs kNN, N=200 | FT >= kNN - 5 points (same pool) |
| G3 vs kNN, N=1,000 | FT >= kNN - 2 points |
| G4 label-budget slope | acc(1,000) >= acc(200) + 8 points |
| G5 tuning sanity | chosen lr not at grid edge at both N, else extend and repeat |
| G6 stability | replicate spread at most 5 points (N=200), 3 (N=1,000) |
| G7 training health | finite loss; final train loss <= 1/3 of first-step loss; time within 2x A5's band |

**Stage 3 (about 1-1.5 GPU-h)** Phi-4-mini, one final per N, gates G1-G5, G7; failing alone means one fix cycle, then Phi is dropped. **Stage 4 (about 0.7 GPU-h)** BillSum memory/throughput, Qwen: 20 steps on the longest 1% of bills, micro-batch 1, gradient checkpointing, completion-only LM head; no OOM; tokens/s inside A5's band. **Stage A (about $1-3, researcher's keys):** 50 validation items per provider x {zero-shot x3, retrieved k=8} + 10 BillSum items; records temperature/thinking acceptance, reasoning tokens, truncations, billed vs `count_tokens`, flip rate; pass: parse failure under 5%, no truncations at the section 5 cap, usage fields present. Total about 3.7-5.2 GPU-h.
**Outcomes:** *Go* means freeze procedure and pre-register. *Fix and repeat* (at most 2 cycles of about 2 GPU-h) if G1/G2/G3/G5/G6 fail with healthy losses (order: lr grid, E_max/S_min, label masking and template, batch size). *Re-scope* if G2 still fails by more than 10 points after two cycles or stage 1 reveals a training-code defect: head-based FT, FT only at N >= 1,000, or a kNN/ICL/API-cost paper.

## 9. Blocking vs deferrable code (A3-04)
Blocks the pilot: tuned-procedure code, kNN + embedder pin, merged tagged commit, `?` fix, BillSum loader + checkpointing (stage 4 only), `diagnose_nan.py` message. Can wait until stage 2 passes: SLM ICL harness (vLLM) and shared few-shot rendering, API repeats and replicate id, LEDGAR loader, test-read ledger and `final_eval`, near-duplicate code, head-based bridge, cost scripts. Engineering is at least 4 of the 8 weeks [I, red team F9]; LEDGAR and OLMo are the first cuts. **Stage-0 exit criterion with a date (end of week 1):** merged tagged commit with working `git_commit`, pinned `pip freeze`, test fingerprint file, `test_reads.jsonl`, and the simulation code committed under `analysis/`.

## 10. Contamination (D8)
See `DATASET_AUDIT.md` section 4. No "no contamination" claim.
