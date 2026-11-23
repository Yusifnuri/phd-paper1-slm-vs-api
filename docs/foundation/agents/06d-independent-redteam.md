# 06d Independent red-team review (second independent reviewer)

Status: complete. The Phase 1 findings (F1-F13) were written before reading 06c, ISSUE_REGISTER, DECISION_LOG, PREREG_CONSISTENCY and NOVELTY_AND_RED_TEAM; the Phase 2 section (verdict, classification, gates, top ten, MUST/MAY, limits) follows them.
Reviewer: a fresh reviewer who drafted nothing. Date of review: 2026-11-23. Repo state: `main`, HEAD `cafc966`, clean tree.
Scope read in Phase 1: PREREGISTRATION.md (all 402 lines), EXPERIMENT_PROTOCOL, STATISTICAL_ANALYSIS_PLAN, FEASIBILITY_AND_BUDGET, RESEARCH_FOUNDATION, DATASET_AUDIT, the three pilot result folders, `results/test_split_fingerprint.json`, and `src/final/*`, `src/train/{tuning,train,billsum,config,env,summarise}.py`, `src/baselines/knn.py`, `src/icl/cli.py`, `analysis/{inference,blinded_variance,build_contrasts,power_sim2}.py` plus `analysis/reference_outputs`, `scripts/{run_stage2,verify_test_isolation,freeze_test_splits}.py`. Not read in detail: `src/icl/{render,retrievers,generate}.py`, `src/api/*` (grepped only), `src/train/{data,evaluate}.py`. Nothing was run on GPU or API.

## PHASE 1 FINDINGS (before reading the internal red-team material)

Severity scale: CRITICAL = a confirmatory result could be invalid or the freeze meaningless; HIGH = a headline claim or the tag's credibility is at risk; MEDIUM = should fix or disclose before the tag; LOW = hygiene.

### F1 HIGH. No code path can produce a confirmatory (test) number for any arm
Evidence: every runner hard-codes validation. `src/train/train.py:358` `"eval_split": "validation"` and `run()` scores `splits.val` only (lines 289-306); `src/baselines/knn.py:163`; `src/icl/cli.py:113,370` (`"split": "validation"`). `src/final/final_eval.py` only returns the loaded test rows after its gates; no runner calls it (grep: only tests, docs, `freeze_test_splits.py`, `billsum.py`). The API runner (`src/api/run.py`) was grepped only, and no API call has ever been made.
Why it matters: the protocol tag is supposed to freeze "the frozen analysis scripts" (PREREG s.7.2). The code that actually touches the test split for FT, kNN, ICL and API will be written after the tag, unfrozen, by the same person/assistant who will see the results. That is the classic forking-path hole, in the one place the gate was meant to close.
Fix: write and unit-test one `run_final` path that (i) takes the config, (ii) calls `final_eval`, (iii) scores with the same functions used on validation, (iv) writes records with `extra.pool_hash` and `extra.replicate`. Dry-run it end to end through `final_eval` with an injected loader that serves validation rows. Commit it before the tag.
Who: assistant builds; researcher confirms the tag only after the dry run passes.

### F2 HIGH. The freeze has no external anchor
Evidence: `final_eval._check_tag` (lines 59-64) accepts any ref `refs/tags/prereg-v1` that is an ancestor of HEAD. It does not check that the tag is annotated, signed, pushed, or dated. Repo tags today are three lightweight local pilot tags; a remote exists (`origin`, GitHub) but nothing says the tag will be pushed. Registry copy is "optional" (D-18, open).
Why it matters: a tag can be created or moved after the fact onto an older commit; `merge-base --is-ancestor` still passes. The "pre-registration" is then unverifiable to a reader, and a reviewer cannot distinguish it from a post-hoc document.
Fix: annotated tag, pushed to origin before any test read, plus an external timestamp (OSF or AsPredicted, or at least a public GitHub release/commit hash posted in the registry). Write the tag object sha into the paper and into `final_configs.json` notes. Same for tag A.
Who: researcher.

### F3 HIGH. All four hypotheses and the six-contrast family are stated against an arm that does not exist
Evidence: PREREG s.2.1 and s.3: H1-H4 "exist only in variant A"; variant B (local comparator) is "not decided" and its hypotheses "would have to be re-worded". The latest commit message says keys will come and the plan stays with the commercial API arm, but the document text still says "unfunded, cannot exist now" (s.2.1, s.7.2, s.14).
Why it matters: if the protocol tag is cut now, the confirmatory content is untestable until an addendum (tag B) exists, and any rewording to a local comparator would be done after the pilot showed the local pattern (local ICL ahead at N=200, FT ahead at N=1,000/3,000, PREREG s.16.4). That is a hypothesis chosen after seeing (validation) data.
Fix, either: (a) freeze now a protocol whose confirmatory contrasts are explicitly "FT vs best-validation retrieval ICL of the comparator named in a dated addendum", with the addendum procedure and its allowed content fixed, and disclose the pilot pattern; or (b) wait to freeze until the API route (provider, key date, budget) is decided. Do not freeze two live variants.
Who: researcher decides; assistant drafts.

### F4 HIGH. "Equal tuning effort" is equal in trial count only, and the asymmetry favours fine-tuning
Evidence:
- FT selection is argmax over lr x checkpoint (5 lr x 5 checkpoints, `tuning.select`, lines 108-120) on the same val(N) rows; ICL is argmax over 5 k values. Candidate sets 25 vs 5 (API arm: 3 providers x 5 k).
- ICL k has almost no leverage: `icl_arms.json` per_k spreads are 1-2 points at every N (N=200: 0.79/0.805/0.79/0.78/0.79). The levers that matter for ICL (instruction wording, retriever, ordering, formatting) are not tuned (PREREG s.7.1: "prompt text is not tuned"). FT's lr grid was repositioned twice after pilot rounds on the very validation rows and replicate-0 pool that the campaign will tune on (grid {5e-5..4e-4} then {1e-4..8e-4} then {5e-5..8e-4}, PREREG s.7.1; `val_hash 6cee7d...` identical in all three frozen_arms files).
- Edge rule applies to FT in practice (N=3,000 chose 8e-4, the top; 1.6e-3 extension = sixth trial) and to ICL only if k=4 or 64 is chosen (it was k=8 at all N). So FT can get 6 trials where ICL gets 5.
- Total labels are N + val(N): at N=200 the arms effectively use 400 labels, at N=3,000 4,000. The "equal pool N" claim and the labelling-cost axis (`N * c_L`) omit the validation labels (PREREG s.12).
Why it matters: a "FT beats ICL" result can reflect a stronger-tuned FT versus a first-try ICL prompt, not the paradigm. The pre-registered 12-vs-4 sensitivity covers FT hyperparameters and one ICL paraphrase, on BANKING77 only.
Fix: (1) state the effort in candidates evaluated and GPU time, not "trials"; (2) give ICL a comparable pilot adjustment on validation (at least instruction paraphrases x retriever, both tasks) before the tag, or pre-register that "ICL is under-tuned by design" in the claims; (3) count validation labels in the headline ("N labelled + min(1000,N) for selection") and in the cost axis; (4) apply the edge rule symmetrically in the text (extension for either arm at the edge, same cap).
Who: researcher decides how much ICL tuning; assistant implements.

### F5 HIGH. GPU training is not reproducible and nothing has been done to measure how much
Evidence: `src/train/env.py:15-30` seeds Python/NumPy/torch only; no `torch.use_deterministic_algorithms`, no `CUBLAS_WORKSPACE_CONFIG`, bf16 autocast (`train.py:93,139`), no attention-kernel pinning (grep for determinism flags: none). Pilot files: same lr, same pool and train seed, rep 2 N=1,000 final = 0.872 (commit f5339262) vs 0.854 (commit ce0dd2a4); rep-0 selection accuracy of the same configuration 0.883 vs 0.871; N=200 finals 0.776/0.769 vs 0.786/0.789. First-step training losses are identical between runs (0.455, 1.377, 1.374, 0.765), so init and batch order match; divergence happens later (kernel non-determinism, or a code change between commits; undecided in the files).
Why it matters: the movement (1-2 points) is as large as the lr-to-lr differences used to select the frozen arm (N=1,000: 0.855/0.861/0.871/0.864) and as the contrasts' MDGs. Consequences: (i) frozen arms are a single noisy draw that cannot be re-derived; (ii) D-15/D-04 retry of "the same seed" yields a different number, so re-runs are a seed-shopping channel unless strictly infrastructure; (iii) the power simulation treats replicate SD as a free parameter and never separates pool variance from nondeterminism; (iv) a reader cannot reproduce a cell. The crossed model itself stays valid if each (pool, seed) is run exactly once.
Fix before tag: rerun the identical N=200 final (71 s) and N=1,000 final (7 min) 3 times at one commit to measure run-to-run SD; decide between deterministic flags (measure the throughput cost) and accepting nondeterminism with disclosure; fix the retry rule so a retry result replaces nothing unless the first produced no result; consider tuning on the mean of 2 train seeds at N<=1,000 (cost 2x of 0.7-1.2 GPU-h per cell).
Who: researcher decides cost trade-off; assistant runs.

### F6 HIGH. The BillSum hypothesis (H3) and RQ2 are not testable as written, and the loss-flat observation says why
Evidence (files): `pilot-session3-2026-11-22/results.jsonl` BillSum N=200 tuning: validation loss across the five learning rates is 0.7464/0.7500/0.7545/0.7549/0.7742 at their best checkpoints (spread 0.03 nats; the minimum is at the FIRST checkpoint, step 40 = epoch 2, for lr 4e-4 and 8e-4, and loss rises to 0.98 and 1.12 by step 120). Final at N=200 on 1,000 bills: 0.7877 (selection set 200 bills: 0.7464, so a 0.04 optimism from selection that is larger than any lr effect). Untuned N=1,000 run: val loss 0.7842, train loss 0.408 (collapsed, overfit). So N=200 to N=1,000 buys ~0.004 nats.
Consequences:
1. FT is selected by validation loss; local ICL by ROUGE-L of generated summaries (`icl/cli.py:25-28,324`); the confirmatory metric is ROUGE-L. Selection criteria are unequal and loss is a weak proxy for ROUGE.
2. The optimum sits at the first of five checkpoints, i.e. a low schedule-length edge that the edge rule does not cover (it only handles "E at E_max"). The tuned BillSum arm is "train ~2 epochs at high lr", resolution coarser than the effect.
3. Learning curve is probably flat in N for FT (and k <= 3 for ICL), so the "crossover/bracket" statement for BillSum is vacuous, and RQ2 "does the answer differ by task" is never a pre-registered contrast: the family contains six within-task deltas and no task-by-N interaction contrast, on different metrics.
4. H3 ("FT beats API retrieval ICL at N=1,000 and 3,000") can still fail, but ROUGE-L F1 is dominated by output length/style calibration, which a FT model trained on 229-token CRS-style summaries gets for free and an API chat model with k<=3 shots may not. A win says little about quality; length covariates are "reported", not controlled.
5. No generation/ROUGE timing exists for any BillSum arm at campaign scale (PREREG s.16.4: "no timing at all"; FT evaluation uses HF `generate` batch 8 on long prompts). The MDGs 0.6/1.0/1.9 rest on assumed replicate SDs of 0.2/0.4/0.8 ROUGE points with no data.
6. The provisional time class for N=3,000 ("72.5 min vs 72 min, cuts 1 and 4") projects full schedules, but the tuned stop step at N=200 was 40 of 200; a loss-stopped final would cost a fraction. The demotion of BillSum N=3,000 to R=3 exploratory looks premature on the evidence, but equally nothing measures it.
Fix: select FT and ICL by the same criterion (ROUGE-L on a fixed val subset, e.g. 200 bills, or add the loss for both); add an early checkpoint (epoch <= 1) and extend the edge rule to a low stop-step edge; run one validation generation pilot (N=200/1,000, 2 replicates) to get ROUGE-L replicate SD and timing; add a length-matched/controlled secondary metric; either add a task-by-N interaction contrast or rewrite RQ2 as descriptive; consider making BillSum N=1,000/3,000 contrasts "confirmatory, direction only, flat curve expected".
Who: researcher decides what H3 claims; assistant implements.

### F7 MEDIUM-HIGH. Some confirmatory claims cannot fail, or fail only by under-powering
- H2 is "supported" if the N=200 outcome is anything except "FT better" (PREREG s.3). With MDG 2.7-5.0 points the expected outcome is (iii) inconclusive, so H2 passes by lack of power. The text half-admits it but still lists it as a hypothesis with a support rule. Also N=200 is "a bracket by rule" (D-12) while still consuming one of the six Holm slots.
- H4 is evaluated at u=1 and sync pricing, the setting most favourable to FT (PREREG s.12 and STAT s.7 themselves say break-even disappears at u<=0.5 or batch), and the pass window is Q* upper CI <= 10^7, while the plan's own sensitivity table gives 9.7k/40k/265k. Its only real failure mode is the non-inferiority proviso. Its inputs are assumed or stale: C_train 0.4-1.15 GPU-h "for 5 runs" (`inference.BREAKEVEN_H`) versus measured tuning alone of 0.70 h at N=1,000 (4 lr, 2,537 s) plus 8 finals of 432 s (~0.96 h) under the stated C_train definition; inference cost `(20/3600/1000, 60/3600/1000)` hours/item is FLOP arithmetic, no FT inference throughput was measured; the GPU rate band is a scenario with no NVL price.
- "Practically equal" (ii) is descriptive and the margin m=3 is "justified as smaller than the gap at which the cost ranking could change", with no computation behind it.
Fix: drop H2 as a hypothesis or recast it as estimation; make H4 a stated-conditions projection or add the u<=0.5/batch scenario to its failure condition; measure FT serving throughput; recompute C_train from the final design.
Who: researcher.

### F8 MEDIUM. Power/MDG logic versus what the code computes
- The MDG table (PREREG s.9.5) comes from `power_sim2.py` with assumed inputs (`LEV` accuracy 0.70/0.80/0.87, pool-effect SD `tau` 1.5-3.0 / 1.0-2.0 / 0.7-1.4 points, `RHO` 0.85/0.93 vs 0.65/0.85). The runtime code (`inference.minimal_detectable_gap`, `blinded_variance`) is a different engine (plug-in, variance components treated as known, df plugged in). Equivalence is stated in a docstring and "proven numerically in tests"; I did not re-derive it. Plug-in power ignores the uncertainty of `s2_rep` estimated from R=5 to 8 replicates (4-7 df) and truncated at 0 (`variance_components` max(...,0)); a replicate component truncated to 0 collapses SE and inflates df (anti-conservative).
- The plan's own type-I table shows conditional uncorrected rejection of 5.2-7.1% at nominal 5% (STAT s.2), so Holm at 0.05 does not control family-wise error at 0.05 under its own simulation.
- BillSum MDG "1.9" at replicate SD 0.8: in `analysis/reference_outputs/sim2_billsum.json`, power at gap 0.02 is 0.766 and at 0.03 is 0.989 for sigma_rep 0.008, so 80% lies near 2.1 points, not 1.9. D-12 dropped the 2-point BillSum rule because "all MDGs are below 2". The number is not reproduced from the committed file (the table may have used a different alpha; unknown).
- Blinded re-estimation: the d-matrix `.npz` built by `build_contrasts` holds the test data and the mean; blinding is by code convention only; with R_obs=5 the variance estimate that drives "extend to 10" is very noisy. Impact small (R=10 buys 0.3-0.6 points).
Fix: add a type-I check at the real design (R=8, n=3,080) with the final code; report SE with a lower bound on `s2_rep` or use a REML/GLMM check; recompute BillSum MDG from the file.
Who: assistant.

### F9 MEDIUM. Routes by which test-derived information reached design choices, and how the gate can be bypassed
1. Descriptive audit used test text before the freeze: DATASET_AUDIT s.3 reports "11 exact test/train duplicates", "417 BANKING77 test items with cosine >= 0.9", "551 / 297 BillSum test items", and coverage by class distribution (which supports stratifying at N=200, L-04). These are test-derived statistics (no accuracy). PREREG s.0 and s.5.3(1) say "no test row is read by any experiment code", which is narrowly true but a reader will read it as "never looked". Disclose the audit and its decisions-influenced list explicitly.
2. `scripts/freeze_test_splits.py` loaded the BANKING77 test split (fingerprint created 2026-11-22T16:22:52Z) and the BillSum test split (via the guarded reader, bypassing `final_eval`) with no ledger line. A hash is arguably not a read, but the ledger starts empty and the study's own rule says every read is logged. Log it, or state the exemption in the protocol.
3. `final_eval.load_banking77_test(revision=None)` loads the default revision, while the fingerprint pins revision `1846984...` for BANKING77 and `3d85104...` for BillSum (file note says so). A silent upstream edit makes the hash mismatch after the data were already read; the mismatch is then logged as a read. Pass the pinned revision.
4. The gate is an honour system: `config` is a dict chosen by the caller (hash is of that dict, not of what was run); one call returns the whole test split, so any number of variants can be scored from one logged read; `loader` is injectable; `load_dataset(..., split="test")` is blocked only by an AST scan of repo `.py` files under `src/`, `scripts/`, `analysis/` (notebooks, `tests/`, other dirs are not scanned; dynamic split names such as `"te"+"st"` pass). The append-only check compares to git HEAD only, so an uncommitted ledger can be rewritten. This is acceptable for a one-person study but must be described as a discipline aid, not a barrier.
5. Validation rows are not fresh: the confirmatory replicate-0 tuning will use the same validation carve-out and pool seed 0 that three pilot rounds already used to reposition grids, choose the ladder, add k=64, and decide the G4 interpretation. Not test leakage, but the frozen arms are partly known in advance and validation accuracies are optimistic.
6. Near-duplicate and similarity features for test items require test text vs pool; that code must also go through `final_eval`; it is not written.
7. Selection (k, retriever) on validation rows whose near-duplicate rate to the pool may differ from test (BillSum: bills re-introduced across Congresses; the val-to-pool rate was never reported while the test-to-train rate is 16.9% at cos >= 0.9). Whether BillSum's train/test split is temporal or random is not stated in the files I read; if temporal, validation carved randomly from train does not mimic the test shift.
Who: assistant for 2-3, 6; researcher for wording of 1.

### F10 MEDIUM. API-arm plan realism
- No key has ever been used. Model ids (`claude-haiku-5-5`, `gpt-6-luna`, `gemini-3.1-flash-lite`), prices and "no temperature" behaviours come from fetch-agent relays, flagged in FEAS as model-relayed; Gemini price "medium confidence from a truncated page". I cannot verify them.
- Request count 274,704 uses 4 k values; decisions use 5 (B77) and 3 (BillSum); dollar columns were priced on older counts and not re-priced (FEAS s.2). Batch-vs-sync is unresolved (D-14 vs PROT s.5 item 4: batch B77 needs 7-13 waves, the 72-hour rule is met only by the sync fallback).
- Stochastic arms (Haiku accepts no temperature control) turn one cell into two full runs (+49k B77, +19k BillSum requests) and make per-item scores 0/0.5/1; with a per-provider drift window of 72 h.
- Model ids can be retired or silently updated before publication; the arm is not reproducible later. Release of predictions/flags only.
- Contamination: BANKING77 (2020) and BillSum (2019) are public and almost certainly in the APIs' pretraining; the exact-prefix probe is described for open models only. API results on these two benchmarks will be inflated by an unknown, non-comparable amount relative to Qwen; "no contamination claim" is stated, but the confirmatory contrast FT vs API rests on it.
- Selecting provider and k on N=200 validation items (SE about 3 points) makes "best API" partly noise at the smallest N.
- Provider terms: paid tier only; competing-model clause not triggered; incorporated policies not all read (DATASET_AUDIT s.5); the release decision is deferred (D-21).
Fix: nothing blocks the protocol tag except F3; the API specifics belong to the tag-B addendum, which must have its own go/no-go on a live 50-item test.
Who: researcher (keys, budget, terms).

### F11 MEDIUM. Things that cannot be undone after the tag
Family of six contrasts and the Holm rule; the estimand (conditional on the fixed training set); R=8/5 and the extension rule; m=3; N grid; the BillSum 1,000-item test subset (seed 42) and its hash; the BANKING77 N=200 stratified-pool convention; Qwen3-4B-Instruct-2507 as the only confirmatory FT model; five-trial rule and edge rule; the retry rules D-04/D-15; replicate numbering and seed formula; that test items are never truncated; the order tag -> tag A -> tag B. Also not undoable: spending the weekly GPU credits (carry-over unknown, D-19), and any API spend. Sixteen of the 21 decisions were taken by the assistant under delegation (PREREG s.17); H1-H4 directions were "accepted as drafted". A tag over a text the researcher has not read line by line should not happen.
Who: researcher.

### F12 MEDIUM-LOW. Numbers and statements not backed by a file in the repo
- GPU rate band $3.6-6.0/h, API prices, model ids, context limits: from fetch agents; no committed source file I read.
- "about 95 s per 1,000-bill pass" and "about 40 GPU-min per N" for BillSum validation (PREREG s.5.2): self-labelled assumption.
- "6 h of 13.3 h used" GPU: assistant's accounting; "billed time may be higher".
- BillSum MDG 1.9 (see F8).
- HANDOFF campaign estimate 20-27 GPU-h and ~28k-request single-provider arm: "no foundation file derives it" (FEAS s.3a, PREREG s.2.1).
- Pilot table values I recomputed from `results.jsonl` and they match: G4 gain 7.4 and 9.8 points; FT-minus-kNN +6.6/+1.95; best-checkpoint accuracies per lr; ICL-minus-FT -3.5/+1.8/+2.0. Good.
- ROUGE variant: PREREG s.11 says "rougeLsum"; code default is `rougeL` (`src/eval/metrics.py:177`, `summarise.rouge_l_scores`). Pin one and the rouge-score version.
- PREREG text is stale versus code in places: s.5.3 says `test_split_fingerprint.json`, `final_eval`, the ledger do not exist (they do); s.16.7 and s.7.1 list BillSum generation/ROUGE, BillSum ICL runner, blinded variance script as not built (commit 8f6686a/aaf25e3 added them). Reconcile before tagging so the frozen text matches the frozen code.

### F13 MEDIUM. Compute plan is unmeasured where the load is
BillSum is ~80% of the planned load (FEAS s.3) and has no generation timing; ICL BillSum k=3 prompts are ~9,300 tokens; FT evaluation uses HF `generate`, ICL uses vLLM with `max_model_len` 40,960 "[U] not checked". Weekly allowance 13.3 h with unknown carry-over. The calendar is "not set". Freezing R and the cut ladder before one BillSum generation timing exists means the cut ladder (which changes confirmatory cells) will be exercised after the tag, as logged deviations.

### Quality gates (Phase 1 view, to be finalised below)
Literature: PARTIAL. Dataset: PARTIAL. Novelty: PARTIAL (depends on API arm). Methodology: NOT PASSED. Statistics: PARTIAL. Economic feasibility: PARTIAL. Reproducibility: NOT PASSED. Adversarial review: this document.

---

## PHASE 2 (after reading 06c, ISSUE_REGISTER, DECISION_LOG, PREREG_CONSISTENCY, NOVELTY_AND_RED_TEAM)

Test suite, run read-only with the given venv: `python -m pytest -q` gives **325 passed in 14.7 s**, no failures. (PC-37 notes HANDOFF quotes 246/247 tests: stale.) The suite proves the offline logic, not the missing pieces in F1; it cannot reach GPU or API behaviour. Nothing else was modified; this file is the only new file.

### Verdict

**Not safe to freeze today. Safe to freeze after a short, mostly cheap list (below, "MUST before the protocol tag").** The design is coherent, the stats code is real and tested, the test-isolation machinery exists, and the team's own documents are unusually candid. But the tag would currently freeze (a) a confirmatory family against an arm that has no key, no run and no code path to the test split, (b) a BillSum hypothesis whose selection criterion, metric and curve shape are mismatched, (c) two hypotheses (H2, H4) that pass almost by construction, and (d) a tag that nothing external timestamps. None of these requires expensive compute to fix; several need one decision from the researcher.

### Classification of my Phase 1 findings against the internal material

| ID | Finding | Classification | What the internal material says / what is still missing |
|---|---|---|---|
| F1 | No code path from frozen arms to a test number | **New** | PROT s.9 lists `final_eval` as "can wait"; the gate was built (src/final) but nobody noted that no runner calls it and all three runners hard-code validation. 06c did not see it (code did not exist) |
| F2 | Tag has no external anchor, movable | **Known, handling insufficient** | D-18 leaves the registry optional; 06c C6 only added the two-tag structure. Nothing addresses annotated/pushed tags or external timestamps |
| F3 | Hypotheses exist only for an unfunded/unbuilt arm | **Known, handling insufficient** | PC-29, L-32, L-38 (keys will come; variant A continues). PREREG s.2.1, s.7.2, s.14 still say "no keys, cannot exist"; no rule for how the tag text treats the arm that will be added later; variant B wording after seeing the pilot pattern not flagged |
| F4 | Equal effort is equal in trial count; asymmetries favour FT | **Known (I-23, L-19, L-35, L-36, 06c s.4), handling insufficient; sub-points new** | New: 25 vs 5 candidate evaluations (lr x checkpoint); ICL k has ~1-2 point leverage; FT grid repositioned on the same validation rows across three pilot rounds; validation labels not counted in N or in the labelling axis |
| F5 | GPU non-reproducibility | **Known (I-30, L-33), handling insufficient** | I-30 is MEDIUM/OPEN, with "optional: deterministic flag". I rate it HIGH: it hits frozen-arm selection (lr gaps are the same size as the noise), the retry rule and cell reproducibility. No repeat-run measurement was done; stage-2 vs 2b is confounded with a commit change (the log says so only partly) |
| F6 | BillSum H3/RQ2 testability, loss-flat | **Partly known (PC-39, s.16.4 warning, PC-36), handling insufficient; most sub-points new** | Known: loss vs ROUGE selection mismatch, flat-loss warning. New: optimum at the first checkpoint (uncovered low edge), 0.04 selection optimism, final-run cost is stop-step-limited so PC-36's class is probably pessimistic, no task-by-N contrast for RQ2, length/style confound of ROUGE-L |
| F7 | H2 and H4 pass by construction; stale H4 inputs | **New** (06c C10/R5 cover BillSum equivalence and N=200 bracket only) | H4 uses u=1 and sync (best case for FT), a 10^7 window, C_train from before measurements |
| F8 | Power/type-I engine mismatch, plug-in df, BillSum MDG 1.9 vs ~2.1 | **Partly known** | STAT s.2 prints 5.2-7.1% conditional type-I; I-24 admits the `mdg` and BillSum sims were not rerun. New: plug-in ignores s2_rep uncertainty/truncation; D-12's "all below 2" does not reproduce from the committed reference file |
| F9 | Test-information routes and gate bypasses | **Mostly new**; item 5 related to PC-31 | Audit used test text (DATASET_AUDIT s.3) yet PREREG says "no test row is read"; fingerprint script reads are unlogged; revision not pinned in the loader; caller-supplied config hash; AST scan scope |
| F10 | API-arm realism | **Known** (PC-33, PC-35, PC-02, I-09, I-17, I-18, L-31, D-21) | New: the contamination probe is open-model-only so FT-vs-API contrast carries an unmeasurable differential; model-id retirement vs reproducibility |
| F11 | Irreversibles; 16/21 decisions were the assistant's | **Known** (PREREG s.17, L-28) | The remaining gap is process: the researcher should read and initial the text before the tag |
| F12 | Unbacked numbers; stale PREREG text; rougeL vs rougeLsum | **Partly known** (PC-37 lists stale text in other files, section G lists unverified numbers) | New: PREREG s.5.3, s.7.1, s.16.7 are stale versus the code; ROUGE variant mismatch; BillSum MDG |
| F13 | BillSum compute unmeasured | **Known** (PC-36, PC-38, I-15) | Still no generation timing |

Agreement with 06c: its CRITICAL (FT arm unproven under the corrected prompt) is now largely addressed by evidence: Qwen/BANKING77 trains sensibly, FT is 2-7 points above kNN on validation, parse failure under 1%. That evidence is exploratory, on selection rows, replicate 0-2, and I re-checked its numbers against the files (they match). I do not think it needs a further pilot before the tag, only an honest statement that G4 was waived (PC-30 still demands a written waiver).

### The eight quality gates (final)

| Gate | Status | Evidence that would change it |
|---|---|---|
| 1 Literature | **PARTIAL** | Spiegelhalter full text is now read (L-21), which is real progress. Still: Semantic Scholar/OpenReview/ACL Anthology never searched; sources read via fetch agents; the "absent in a limited search" wording must stay; a second literature check by someone other than A1 (NOVELTY gate row 3) |
| 2 Dataset | **PARTIAL** | The test-derived audit statistics disclosed in the protocol (F9.1); BillSum train/test split type (temporal or random) stated, and val-to-pool near-duplicate rate reported next to the test rate; incorporated provider policies read (I-18); BANKING77 label noise limitation |
| 3 Novelty | **PARTIAL (variant A) / FAIL-to-incremental (variant B)** | Delivery of the API arm; credit for Mansuri and Trad-Chehab already done (I-28). Without the arm the paper is an incremental extension, as PREREG s.2.1 itself says |
| 4 Methodology | **NOT PASSED** | F1 (test path exists and is dry-run), F4 (effort stated in candidates and GPU, ICL pilot adjustment, validation labels counted), F6 (same selection criterion for BillSum arms, generation pilot), G4 waiver written (PC-30), PC-31/32 decided |
| 5 Statistics | **PARTIAL** | F7 (H2, H4 recast), F8 (type-I at the real design with the final code; BillSum MDG recomputed), a replicate-SD measurement from repeat runs (F5), BillSum ROUGE-L SD from a validation generation pilot |
| 6 Economic feasibility | **PARTIAL** | A derived budget for the decided design (PC-38), one BillSum generation timing, measured FT serving throughput, carry-over answer (D-19), API re-pricing after the grid is final (PC-33/35) |
| 7 Reproducibility | **NOT PASSED** (improved) | Code, fingerprints, ledger and `analysis/` are now in the repo and 325 tests pass; remaining: non-bitwise training (F5), unanchored tag (F2), unpinned loader revision (F9.3), stale PREREG text vs code (F12), `pip freeze` at the tag |
| 8 Adversarial review | **PARTIAL** | This is a second independent review (D-20 satisfied for the text read, not for later edits). It should be re-run on the final text after the MUST list, or the researcher should accept a self-check against this list |

### Findings table (final severities)

| # | Sev | Finding | Fix owner |
|---|---|---|---|
| F1 | HIGH | No code writes a test result for any arm; the tag would freeze the gate but not the evaluator | assistant |
| F2 | HIGH | Tag not annotated/pushed/timestamped; back-datable | researcher |
| F3 | HIGH | Confirmatory family only exists for the unbuilt API arm; text stale vs L-38 | researcher |
| F4 | HIGH | Equal effort nominal; ICL under-tuned by design; val labels uncounted | researcher + assistant |
| F5 | HIGH (06c/I-30: MEDIUM) | Nondeterministic training unmeasured; retry rule becomes seed-shopping | researcher + assistant |
| F6 | HIGH | BillSum: selection criteria differ, first-checkpoint edge, flat curve, no RQ2 contrast, no ROUGE/timing data | researcher |
| F7 | MEDIUM-HIGH | H2, H4 cannot fail meaningfully | researcher |
| F8 | MEDIUM | Plug-in power; type-I 5.2-7.1% under own simulation; BillSum MDG | assistant |
| F9 | MEDIUM | Audit used test text; unlogged fingerprint reads; loader revision; gate is a discipline aid | assistant + researcher (wording) |
| F10 | MEDIUM | API specifics; differential contamination | researcher (addendum) |
| F11 | MEDIUM | Irreversibles incl. 16 assistant-made decisions | researcher |
| F12 | MEDIUM-LOW | Stale PREREG text; ROUGE variant; unbacked numbers | assistant |
| F13 | MEDIUM | BillSum compute unmeasured; cut ladder after the tag | assistant |

### Top ten, ranked

1. F1: build and dry-run the single test-evaluation path (through `final_eval`, with an injected validation loader) and commit it before the tag.
2. F3: decide how the tag treats the API arm (freeze variant A with a defined addendum procedure, or wait); fix PREREG s.2.1/s.7.2/s.14 to L-38.
3. F6: one selection criterion for both BillSum arms, an early checkpoint, a validation generation pilot (ROUGE-L SD, timing), and a rewritten H3/RQ2.
4. F5: three repeat runs of one configuration; decide deterministic flags vs disclosure; tighten the retry rule.
5. F2: annotated, pushed tag plus an external timestamp before any test read.
6. F4: restate "equal effort" honestly; count validation labels; give ICL a comparable validation pilot or pre-register the asymmetry in the claims.
7. F7: drop or recast H2; recast H4 with a failure condition at u<=0.5 or batch; recompute C_train from measured times.
8. F9: disclose test-derived audit statistics; log the fingerprint reads; pin dataset revisions in the loaders; describe the gate as a discipline aid.
9. F12: reconcile PREREG text with code; pin ROUGE variant and version; fix the MDG figure.
10. PC-30/31/32 and F8: write the G4 waiver, decide the N>=1,000 gate rows and the sixth-trial treatment, rerun type-I with the final code.

### MUST be fixed before the protocol tag

- F1 (evaluator path and dry run), F2 (anchor plan), F3 (variant/addendum structure and wording), F6 (selection criterion, H3/RQ2 text, ROUGE variant), F7 (H2, H4 wording), F4 (state effort and validation labels; decide the ICL pilot), F5 (repeat-run measurement and the retry rule), F9 items 1-3 (disclosure, log, revision pin), F12 (stale text), PC-30 waiver, PC-32 treatment. The researcher should read the final text, since 16 of 21 decisions were made by the assistant.

### MAY follow after the tag (as logged additions, not deviations, if they only add exploratory or descriptive material)

F8 (type-I rerun can be a pre-registered sensitivity run before the first test read; it changes no rule unless it fails), F10 API specifics (tag-B addendum with its own live test), F13 budget derivation and calendar, optional deterministic-flags study, near-duplicate bin edges (PC-41), the 12-configuration tuning-bar sensitivity results (they run on validation before tag A).

### Honest limits of this review

I read the listed documents and code; I did not run any GPU or API job, did not re-derive the simulation model, did not verify provider prices, model ids or the existence of "claude-haiku-5-5", "gpt-6-luna" or "gemini-3.1-flash-lite" (they are beyond what I can check), and did not read `src/icl/{render,retrievers,generate}.py`, `src/api/*` or `src/train/{data,evaluate}.py` in detail. Pilot figures I recomputed from the committed `results.jsonl` files; claims about the HF dataset split type (BillSum) and about GPU kernel nondeterminism as the cause of the run-to-run gap are inference, not tested; the stage-2 vs 2b difference may partly be a code change between commits. The review is of a snapshot at HEAD `cafc966`; later edits are not covered. I am a model reviewer with the same blind spots as the drafting assistant; a human statistician reading the crossed-model section and the BillSum selection logic would add value.
