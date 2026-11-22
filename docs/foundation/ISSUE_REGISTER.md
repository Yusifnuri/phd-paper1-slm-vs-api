# Issue register

Maintained by the integration agent (the coordinating session). Specialist agents propose issues in their own file under `agents/`; only the integrator edits this file, so concurrent sessions never collide. An issue is **not** resolved because an agent says so: it needs evidence, and for CRITICAL/HIGH an independent reviewer.

Severity: CRITICAL (could invalidate the paper or its main findings) · HIGH (substantially weakens methodology or claims) · MEDIUM (needs correction, does not invalidate) · LOW (minor).
Status: OPEN · IN-REVIEW · FIXED-UNVERIFIED (changed in code/protocol, not yet checked independently or on GPU) · RESOLVED (evidence + review) · ESCALATED (genuinely human-dependent) · WONT-FIX (shown not to matter).

| ID | Issue | Sev | Claims affected | Responsible | Reviewer | Evidence so far | Proposed resolution | Status | Human input? |
|---|---|---|---|---|---|---|---|---|---|
| I-01 | The two arms saw different information (fine-tuned prompt had no label list, API prompt had) | CRITICAL | every fine-tune vs API claim | A3 | A6 | `verification/05` B1 | shared `task_spec` (code `b12f80c`, branch `claude/pilot-train-226845`) | FIXED-UNVERIFIED | no |
| I-02 | Under-training at small N (12/39/189/939 steps; 90%+ unparseable at N≤200) | CRITICAL | H1 / label-budget curve | A3 | A6 | `verification/05` B2; pilot jsonl | min-steps + per-N validation-tuned epochs/lr (coded, untested on GPU) | FIXED-UNVERIFIED | no |
| I-03 | Parser lower-cased a case-sensitive label set (`Refund_not_showing_up` unscorable) | HIGH | all pilot accuracies | A3 | A6 | re-score of 45 files: +0.0052 mean | shared case-insensitive normaliser, test, re-score script | FIXED-UNVERIFIED | no |
| I-04 | Fixed pool (`data_seed` 42) and 3 seeds: uncertainty understated | HIGH | all variance/CI claims | A3/A4 | A6 | `verification/05` S2 | fresh nested pool per replicate, ≥5 replicates (code coded; count to be justified by power simulation) | OPEN | no |
| I-05 | Planned inference unsound (3-seed bootstrap, seed-majority prediction) | HIGH | all significance claims | A4 | A6 | `verification/03` §3, `05` S3 | item-level paired tests / mixed model; state the estimand | OPEN | no |
| I-06 | rsLoRA vs LoRA confounded by learning rate; ±1pt TOST margin unreachable | MEDIUM | technique ablation | A3/A4 | A6 | `verification/05` S4 | per-technique lr grid; widen margin or estimation only | OPEN | no |
| I-07 | Multiplicity family undefined | HIGH | all p-values | A4 | A6 | `verification/05` S5 | one small pre-registered confirmatory family | OPEN | no |
| I-08 | Test-set discipline; pool-to-test near-duplicates | HIGH | validity of all test results | A3 | A6 | `verification/05` S6 | frozen hashed test, one read per final config, similarity-stratified report | OPEN | no |
| I-09 | API protocol: nondeterminism, no dated ids (Haiku 5.5, Luna), reasoning tokens, tokenizer, temperature removal | HIGH | all API results and costs | A5/A3 | A6 | `verification/04` | pin ids/dates, reasoning off, billed tokens, 10% repeat | OPEN | live API test needs the user's keys |
| I-10 | Novelty overstated; large overlap with Bertsch 2025 (BANKING77, CLINC150, SAMSum, retrieval vs fine-tuning) and Pecher 2025 | CRITICAL | contribution claim | A1 | A6 | `verification/03`, `00-summary` | defensible gap statement from a full-text novelty matrix | OPEN | no |
| I-11 | Licences unresolved: BillSum (CC0 only in card metadata; US vs California), LEDGAR (no original data licence), SAMSum (NC-ND unverified), tweet data excluded | HIGH | dataset inclusion, adapter release | A2 | A6 | `verification/01,02` | primary-source audit incl. rights holders | OPEN | possibly (legal status) |
| I-12 | LEDGAR label structure (LexGLUE: 100 single labels, rare classes; original 12k+ multi-label) and meaningful N | HIGH | LEDGAR results | A3/A2 | A6 | `verification/01` | label-frequency analysis, fixed label list, N levels per class coverage | OPEN | no |
| I-13 | Summarisation evaluation validity (ROUGE alone; need for human evaluation; faithfulness) | HIGH | summarisation claims | A4 | A6 | none yet | metric set plus human-evaluation decision | OPEN | possibly (annotators) |
| I-14 | Phi first-step loss 18.88 (QLoRA 8.5); OLMo-3 DoRA NaN at step 1 | MEDIUM | model/technique cells | A3 | A6 | `scripts/diagnose_loss.py` | needs a GPU run (server off) | OPEN | no |
| I-15 | Compute/cost estimates unreliable (prompt ≈6× longer with label list; BillSum untested; tuning multiplies runs) | HIGH | feasibility, cost claims | A5 | A6 | pilot timings only | scenario budget from measured tokens/s | OPEN | no |
| I-16 | Generated labels vs classification head: comparability with Bertsch/Pecher | MEDIUM | bridge claims | A3 | A6 | Bertsch §2 (head, constrained decoding) | state implications, add sensitivity analysis | OPEN | no |
| I-17 | Contamination of public benchmarks (BANKING77/CLINC public since 2019–20) | MEDIUM | absolute scores, gaps | A2/A3 | A6 | `verification/02` §C | disclose; contamination-free probe if feasible | OPEN | no |
| I-18 | Provider terms on publishing benchmark results / using outputs | MEDIUM | publication, API arm | A2 | A6 | not yet checked | read each provider's terms | OPEN | possibly |
| I-19 | Estimand vs stratified pools: a random-pool estimand cannot be read from class-stratified N=200 pools | HIGH | N=200 contrasts | A3/A4 | A6 | `agents/06-redteam.md` F4; L-04 | stratified only at N=200 as a named condition, random-vs-stratified table (R=3) | FIXED-UNVERIFIED | no |
| I-20 | Cross-draft contradictions between agent files (15 listed by A6) | HIGH | all numbers | integrator | A6 | `06-redteam.md` §1; `06b` C1, C5 | superseding rule (later rounds govern) plus assembled deliverables; API request count (C5) still unreconciled | IN-REVIEW | no |
| I-21 | Confirmatory family excluded the crossover region (N=200) | HIGH | H1 | A4/integrator | A6 | `06-redteam.md` F1; L-03, L-15 | 6-contrast family incl. N=200 with pre-registered underpowered-a-priori rule | FIXED-UNVERIFIED | no |
| I-22 | Labelling cost not in the cost model | MEDIUM | break-even | A4/A5 | A6 | `06-redteam.md` F-cost; `STATISTICAL_ANALYSIS_PLAN` cost section ("labelling axis") | labelling axis added to the break-even; no number yet | FIXED-UNVERIFIED | no |
| I-23 | Tuning asymmetry and validation size at small N; tuning bar vs Spiegelhalter (27 configs x 5 folds) | HIGH | fine-tuning vs ICL gaps | A3 | independent reviewer | L-05, L-14; `06b` C7 | 4 trials per system (equal effort) plus pre-registered wider-grid sensitivity at N=200/1,000 (L-17); noisy selection at N=200 stated | IN-REVIEW | no |
| I-24 | Power-simulation code and API request count not reproducible from the repository | HIGH | statistics and budget numbers | A4/A5 | independent reviewer | `06b` C5, C6 | copy simulations into `analysis/`; reconcile 259,104 vs 3 x 106,436 requests | OPEN | no |

## Changes on 2026-11-22 (integrator; no status below is RESOLVED without an independent review)

- I-10: CRITICAL -> HIGH per L-13, conditional on never claiming "first"; Spiegelhalter paper text still unread (H4), so the novelty claim stays provisional.
- I-01, I-02: stay FIXED-UNVERIFIED; they are tested only by the gated pilot (G1-G7), which has not run.
- I-04: stay OPEN until the replicate design is coded and R is frozen after the validation-split API runs (H1).
- I-05, I-07: the plan now states the crossed mixed model and a 6-contrast Holm family (`STATISTICAL_ANALYSIS_PLAN.md`); IN-REVIEW, no code exists.
- I-11: MEDIUM is acceptable per A6 because no LEDGAR adapter is released (L-08); keep HIGH until `DATASET_AUDIT.md` rights items are confirmed by an independent reader.
- I-13: HIGH stays; ROUGE-L/BERTScore only unless the human validation study (H2) happens (L-10).
- I-15: re-priced by A5 Round 2 (35.5-112 GPU-h needed vs about 54-70 available; API $93-140); stays HIGH because throughput is unmeasured.
- I-09, I-18: stay OPEN (live API tests need H1; provider terms incorporated by reference not fully read).


## Independent second pass (`agents/06c-independent-redteam.md`, 2026-11-22): new issues and actions

| ID | Issue | Sev | Responsible | Action taken / status |
|---|---|---|---|---|
| I-25 | Pilot gates G1-G7 evaluated on the rows that select lr/checkpoint; G2/G3 tolerate -5 points | HIGH | A3 | Protocol section 8 rewritten: non-selection rows, G3 -2 points, pilot pass is no evidence for any hypothesis. FIXED-UNVERIFIED (pilot not run) |
| I-26 | Weekly GPU cap (13.3 h) vs schedule (W5-W6 25-45 h), carry-over unknown, GC code dependency | HIGH | A5 | Cap-aware note added to FEASIBILITY section 4; H3 stays with the researcher. OPEN |
| I-27 | kNN outside the inference family | HIGH | A4 | STAT plan: secondary contrasts and decision table. FIXED-UNVERIFIED |
| I-28 | Novelty text overstated tuning-charged break-even; Trad and Chehab omitted | HIGH | A1 | RESEARCH_FOUNDATION and PAPER_OUTLINE corrected; Spiegelhalter text still unread (H4). IN-REVIEW |
| I-29 | BillSum API request counts differ (A3 23,200/14,200 vs A5 28,400/22,400) | MEDIUM-HIGH | A3/A5 | `analysis/count_requests.py` (tested) reproduces both: A3 used 300 validation items per N and no random-k on the other providers, A5 used min(1,000, N) and random-k on all. Protocol default = A5 validation size, random-k on the selected provider only: BillSum 67,200 requests. Which validation size to pay for is a decision (L-22). FIXED-UNVERIFIED |
| I-23 (update) | Tuning bar L-14: L-17 was unbudgeted, no flip definition, no edge rule | HIGH | A3/A4/A5 | Replaced by L-19 (12-config sensitivity, 0.7-2.0 GPU-h, margin 3 points, edge rule). FIXED-UNVERIFIED |
| C6/C7/C8/C10/C11/C13 | Two tags; a-priori underpowered rule; R extension B77 only; no BillSum equivalence; provider tie rule; sync fallback | HIGH | A3/A4 | Applied in protocol and STAT plan. FIXED-UNVERIFIED |
Remaining OPEN, not fixable by text: FT arm unproven (I-01/I-02), power code outside the repo (I-24), unread Spiegelhalter paper (H4), no merged/tagged code (F9/F13).

- 2026-11-22: H4 closed; Spiegelhalter full text read (`docs/research/spiegelhalter-2026-notes.md`, L-21). I-10 stays HIGH pending a second literature check, but no overlap with the commercial-API, N>200 or summarisation parts. I-28: the tuning-charged break-even is confirmed as not distinguishing.

- 2026-11-22: I-24 reproducibility part closed for statistics: `analysis/power_sim.py`, `power_sim2.py`, `breakeven.py` and A4's reference outputs copied into the repository; `power_sim2.py typeI` rerun on the Mac gave output identical to the saved `sim2_typeI.json` (24 cells, max difference 0.0) and `breakeven.py` reproduces the break-even table of `STATISTICAL_ANALYSIS_PLAN.md`. The `mdg` and BillSum simulations were not rerun. The request-count part is closed by `count_requests.py` (see I-29); the 06b C5 claim of a 60,204 gap was my arithmetic error and is retracted.

- 2026-11-22: I-09 and I-18 cannot progress until funding exists (L-32). I-15: GPU cost has no dollar basis beyond the imputed band (H3 answered).
| I-30 | Training is not bitwise reproducible: the same configuration, pool and seed gave finals that differ by 1-2 points between two GPU runs (N=200 replicate 2: 0.769 vs 0.789; N=1,000 replicate 2: 0.872 vs 0.854; replicate 1 at N=1,000 identical, 0.869) | MEDIUM | every fine-tuned number | A3/A4 | independent reviewer | `results/pilot-stage2-2026-11-22`, `results/pilot-stage2b-2026-11-22` | record the library versions and the non-determinism in the paper; the crossed model's residual term already absorbs it; do not claim seed-exact reproducibility; optional: deterministic algorithms flag, measured cost | OPEN | no |
