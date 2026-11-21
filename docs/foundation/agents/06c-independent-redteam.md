# A6c: Independent red-team second pass (2026-11-22)

Reviewer: a fresh reviewer who wrote none of the drafts and did not write `06b` (which is an integrator consistency check). Labels: **[V]** I checked it myself (arithmetic, simulation, primary source or the file text), **[I]** inference, **[U]** unverified. Read in full: README, ISSUE_REGISTER, DECISION_LOG (L-01..L-18), HUMAN_DECISIONS, PROGRESS, `06-redteam.md`, `06b`, `agents/01..05` including all Round 2 / 2b sections, and the seven assembled deliverables (RESEARCH_FOUNDATION, DATASET_AUDIT, EXPERIMENT_PROTOCOL, STATISTICAL_ANALYSIS_PLAN, FEASIBILITY_AND_BUDGET, NOVELTY_AND_RED_TEAM, PAPER_OUTLINE). Only this file was written; no GPU, no paid API, no commit. Section numbers below are mine.

Verdict: **go for Stage 0 and the cheap live tests; no go for the campaign.** The first-pass structural failures (grid, N=5000, estimand, test sizes, LEDGAR, random-k, cost scope) are resolved on paper. What is left is evidence that does not exist yet (FT arm under the corrected prompt, FT-vs-API discordance, measured throughput, the key paper's text) and a set of internal defects in the Round 2 text, **most of which were copied unchanged into the deliverables** (section 9). No gate is PASS.

## 1. Status of the first-pass findings

| # | First-pass finding | Status | Evidence |
|---|---|---|---|
| F1 | Confirmatory grid beyond the known crossover | **Partly** | L-03/L-15 put N=200 in the 6-contrast family. But FT below N=200 was dropped (L-02), so nothing is said about the 50-100 label region where Pecher/Spiegelhalter place the crossover, and N=200 is Spiegelhalter's maximum. A4 R2.4: N=200 MDG 2.7-5.0 points at R=8, so "inconclusive" is the expected outcome. Acceptable only because L-01 forbids a crossover claim. I confirmed from the authors' repo README: 17 tasks, N 5-200, 27 configs x 5 folds, no commercial API [V, https://github.com/us58/prompting-vs-finetuning-slm]; paper text unread |
| F2 | Contribution narrower than claimed; thin search | **Partly** | L-01 and the "must not claim" list adopted; about 30 queries on arXiv/OpenAlex/GitHub/WebSearch. Semantic Scholar, OpenReview, ACL Anthology never searched. Wording safe, evidence of absence thin (section 5) |
| F3 | FT arm unproven under corrected prompt | **Not resolved** | Gates G1-G7 defined (EXPERIMENT_PROTOCOL section 8), nothing run; gate design weak (C9) |
| F4 | Stratified pools vs random estimand | **Resolved (text)** | Random N>=1,000, stratified N=200 as a named condition, table at R=3 (A3 R2b, A4 R2.1, protocol section 2) |
| F5 | Tuning symmetry, validation size | **Partly** | val(N)=min(1,000,N), 4 trials each. Equal in count only; cut 1 removes BillSum N=3,000 tuning; ICL prompt untuned. L-17 proposes a sensitivity analysis (section 4 and 9) |
| F6 | N=5000 overlap | **Resolved** | N=3,000. Overlap and finite-population factors reproduced [V]; type-I simulation (A4 R2.3) plausible but its code is not in the repo [U] |
| F7 | Power on unobserved quantities; BillSum R=3 | **Partly** | R=8/5, MDG tables, decision rule. FT-vs-API discordance and BillSum replicate SD still assumed; H1 unanswered; rule defects C7, C10 |
| F8 | API fairness | **Partly** | A3 R2b section 2 rule is correct under either Haiku outcome. A5 section 1.1 still marks the temperature 400 "settled [V]" (not retracted); FEASIBILITY table now says "not confirmed by the red team" [V]. kNN is not in the inference family (R6) |
| F9 | Engineering underestimated | **Partly** | 8-week schedule with week-2 gate; C5 dependency inversion remains |
| F10 | Summarisation validity | **Partly** | L-10, A4 R2.7 (60 bills x 3). H2 open; rater cost up to $400 vs $300 ceiling [V]; ROUGE margin conflict (C10) |
| F11 | Cost model | **Mostly resolved** | Labelling axis, u, rate band reconciled; H3 open and it matters (C5) |
| F12 | Release/legal | **Partly** | L-08 adequate; incorporated provider policies unread (A2 confidence M); keep I-18 OPEN |
| F13 | Reproducibility | **Not resolved** | Nothing merged or tagged; fingerprint file absent; sim code outside repo |
| F14 | Minor | **Partly** | `?` fix and `max_length` scheduled; DATASET_AUDIT still says "no fine-tuning below N=77" (section 9) |

First-pass section 1 contradictions: 1 (B77 test 3,080), 2 (BillSum 1,000), 3, 4 (LEDGAR role, k=8), 6 (N levels), 9 (task set), 10 (GPU band), 11 (random-k), 14 (pool sampling) **resolved**. 5 (replicates) resolved except the unfunded extension (C8). 7, 8 (tuning placement, trainings count) **partly** (A3 78/54/21 vs A5 72+54: A3 adds 6 random-pool N=200 finals A5 omits; trial cost 1x vs 1.5-2x, C4). 12 (API requests) **partly**: B77 within 0.3%, BillSum not (C2). 13 (BillSum cap 1,024) resolved in A3/protocol. 15 (engineering) **not resolved**.

## 2. New contradictions between the Round 2 sections

| ID | Contradiction | Where | Why it matters | Fix |
|---|---|---|---|---|
| C1 | A1 R2.3-2.4 wording says "N up to 5,000" after L-02 | `01` R2.3/2.4 vs L-02, `03` R2b, `04` R2, `05` R2 | Abstract would promise a range not covered | "N up to 3,000". RESEARCH_FOUNDATION already corrected this [V] |
| C2 | BillSum API requests: A3 R2b 23,200 selected / 14,200 other (3,600 validation requests, basis unstated); A5 R2.2 28,400 / 22,400 (implies 12,000 validation requests, contradicting val(N)=min(1,000,N)); rule applied gives 2,200 items x 4 evaluations = 8,800 | `03` R2b s.1; `05` R2.2 | 51,600 vs 73,200 requests; BillSum is the largest API cost line (about $0.001-0.0035 per request) | One committed script printing counts per provider and cell |
| C3 | Extra B77 cells differ: A3 adds 3 random-pool N=200 retrieval replicates (+6 FT finals); A5 adds N=77 R=3 instead. Both come to about 9,240 requests, so totals coincide by accident | `03` R2b table; `05` R2.1-2.2 | The L-04 random-vs-stratified table has no priced API arm in A5; N=77 none in A3 | State the union; add the 6 short FT runs |
| C4 | Tuning trial cost: A3 R2b "9 runs per model" (5 finals + 4 trials at 1x) gives 14-38 GPU-h for BillSum N=3,000; A5 prices a trial at 1.5-2x a final: 16-55 h [V, my recomputation: finals 7.2-21 + tuning 8.6-33.6] | `03` R2b "GPU flag"; `05` R2.0-2.1 | The cut-1 saving is the main schedule lever; the go/no-go rests on this cell | A5's figure governs; also define "same schedule" (A3 replicates run the schedule, A5 prices E* epochs) |
| C5 | Weekly credits vs schedule: 200/15 = 13.3 h/week, 8 resets = 106.7 h gross [V]. FEASIBILITY section 4 puts about 25-45 GPU-h into W5-W6 (cap 26.7 h); carry-over unknown (H3). If none, W1-W4 use about 15-30 of 53 available hours. Also W1 includes a BillSum OOM probe while gradient-checkpointing code is a W2 item and A3 says stage 4 needs it | `05` R2.5 = FEASIBILITY s.4; `03` R2b s.3-4 | 54-70 productive hours may be 20-30% too high; week-2 go/no-go needs a BillSum measurement not possible in W1 | Cap-aware per-week table; move GC code to W1 or the probe to W2; ask H3 now |
| C6 | Pre-registration tag in W3, but the `frozen_arms.json` hash must be "committed with the pre-registration tag" and BillSum arms freeze in W4 | `05` R2.5 vs `03` A4-08 | Either arms are not pre-registered or the tag slips | Two tags: protocol (W3), arms (end W4) |
| C7 | "Realised MDG ... declared underpowered **a priori before the test read**", yet MDG is re-estimated from variance components "after R=5" and replicates 1..R are test reads; "largest plausible effect" has no number | L-15; STAT s.3-4 | Self-contradictory: a priori (validation-split) or realised (data-dependent) | Define a priori from the A4-02 validation-split runs, numeric threshold |
| C8 | Extension to R<=10 vs budget: for BillSum +5 replicates x 3 N x 2 models = 30 finals, about +12.5-33 h by A5's own line for 30 finals [V]; A5: "only if time" | L-06; `04` s.1.4 | The pre-registered rule is not executable for BillSum | Extension for B77 only; BillSum capped at R=5 with stated MDG |
| C9 | Pilot gates G1-G7 are "all on validation", but the same val(N) rows select lr and checkpoint in the same stage; G2/G3 tolerate FT 5 points below kNN; at N=200 val SE about 3 points [V: sqrt(.8*.2/200)=0.028] | protocol s.8 | Can pass on selection noise and still not show FT is competitive with the headline comparator. `06b` C9 says "wording in the protocol already calls these sanity gates": I find no such wording (only "G5 tuning sanity") [V grep] | Gate on rows not used for selection; tighten G3 |
| C10 | Margin m = 1 ROUGE-L point (STAT s.4) vs the same plan citing Goyal et al. (about 5-point gaps informative) and BillSum MDG 0.6-1.9. "Practically equal" needs a 95% CI half-width under 1 point (SE about 0.4); justification "cost ranking could change" is meaningless for ROUGE | `04` R2.5-2.6; STAT s.3-4 | Outcome (ii) unreachable except at the optimistic replicate SD | Drop equivalence outcome for BillSum |
| C11 | Comparator "best API" frozen on replicate 0 by accuracy; the paper's question is cost per correct; A5: provider choice moves cost $15-40; other providers run R=3; at N=200 selection uses 200 items | `04` R2.1; `03` A4-08; `05` R2.2 | Selection uncertainty is outside the CI; the most accurate provider may not be the cost-relevant one. L-18 covers only the "frozen" label | Tie rule (within 1 SE choose cheaper); report all providers' cost per correct descriptively |
| C12 | Pilot size: L-09 "about 3 GPU-h" vs protocol 3.7-5.2 vs A5 extras 5-6 | L-09; `03`; `05` | Reconcilable by scope; the log still says 3 | Footnote in L-09 |
| C13 | A3 D6 "one mode (batch) for all confirmatory runs, 72 h per provider" vs A5: batch B77 needs 7 (Luna) to 13 (Gemini) sequential waves at a 24-hour target. Protocol section 5 now says "one mode per confirmatory run" [V] but keeps the 72-hour rule | `03` s.6; `05` R2.2; protocol s.5 | 72 h not guaranteed for batch Gemini/Luna B77 [I] | State sync fallback and price |

## 3. Arithmetic reproduced

| Item | Result |
|---|---|
| Overlap and finite-population factors | N=200/1,000/3,000: 2.2%/11.1%/33.3%; BillSum 3,000: 501 shared (16.7%); factors 0.67, 0.83 [V] |
| Absent-class share, N=200 / 100 | Uniform-class (1-1/77)^N gives 7.5% / 27%, matches the HF-statistics figures [V] |
| A3 B77 API requests | 83,160 + 9,240 + 3,080 + 8,800 + 1,848 = 106,128; others 50,688; total 207,504 [V]. BillSum 23,200 + 2 x 14,200 = 51,600 only if the 3,600 is accepted. Grand total 259,104 [V] |
| `06b` C5 and protocol "rounding" claim | The comparison "3 x 106,436 = 319,308" is wrong: other providers run R=3 (50,688), so A5's own total is 106,436 + 2 x 50,688 = 207,812 for B77, within 0.15% of A3's 207,504. The 308-request gap is 0.1 x 3,080 (a different extra cell set, C3), not "rounding in the repeat term". The **real** discrepancy is BillSum (C2), which neither `06b` nor the protocol identifies [V] |
| A5 GPU hours | Components sum to 35.4 / 112.3 = 35.5-112 [V]; BillSum tuning 14.4-51.2 reproduces [V]; 13.33 x 8 = 106.7 h [V] |
| Prices | Haiku 5.5 $0.10 / $0.50 per MTok, batch $0.05 / $0.25 confirmed on the provider page today [V, https://platform.claude.com/docs/en/about-claude/pricing]. B77 k=32 request $0.000126-0.000169 and BillSum k=3 $0.00105-0.0014 reproduce [V]. Luna and Gemini prices not rechecked [U] |
| API budget | A5 ranges consistent with my per-request arithmetic within about 20%; fits $50-200 in mixed mode only (standard $109-187) |
| Break-even table end points | 9.7k, 22k, 265k reproduce from C_train $1.44-6.9, c_API $0.000126-0.000169, c_FT $2e-5-1e-4 [V]; "central" column depends on unstated midpoints [U] |
| "Labels cost 25-1,000x GPU training" (STAT s.7) | $100-2,000 / $1.44-6.9 = 14-1,400x [V]. Immaterial but wrong, copied into STAT |
| Human-study effort | 180 x 8-13 min = 24-39 h; rater 2 12-20 h; $180-400 at $15-20/h [V] |
| BillSum MDG | A3's "MDD about 1.2" is critical value x SE (4.6 x 0.26), a 50%-power gap; with A4's components my 80%-power value is 1.0-1.2, consistent with A4's 1.0 [V]. Use A4's only |
| Power cell, B77 N=1,000, R=8, n=3,080, opt regime, crossed components + Satterthwaite | My simulation from A4's written model (latent probit, correlations .85/.93, replicate SD 1 point, pool effects correlated .5; 300 sims): delta 1.9 -> power 0.73 (alpha .0083) / 0.88 (.025); delta 2.7 -> .97 / .997. A4 reports MDG 1.9 / 1.8. Agreement within MC error (+-2.6) and unshared details [V, partial]. A4's own code is not in the repo [V: `find` finds no power_sim*, breakeven*, sim2_*]; the type-I table and the other MDG cells are unreproduced [U] |

## 4. The tuning bar (L-14, L-17)

Facts: Spiegelhalter et al. use 27 LoRA configs x 5 folds per cell [V, repo README] plus GEPA prompt optimisation for the ICL side [V, README via A1]. Ours: 4 learning rates, everything else fixed, epochs by in-run checkpoint selection (5 checkpoints), selection on val(N) on one replicate then frozen; ICL: 4 values of k (BillSum plus a retriever choice), no prompt tuning.

1. "4 vs 4" is equal in count only: an FT trial is a full training with 5 validation passes (1.5-2x a final run, A5 R2.0), an ICL trial one validation pass.
2. Selection at N=200 uses 200 items (SE about 3 points), so the chosen lr/provider is partly noise (`06b` C7; agree).
3. The error direction matters: under-tuned FT flatters ICL/API; untuned prompts do the opposite. Both are declared; the headline is still a difference between two partly tuned systems.
4. 27 x 5 folds is neither needed (we have a validation set) nor affordable (135 trainings per cell, about 4-10 GPU-h per cell for one model at A5's per-run times [I]).

Assessment of L-17 (integrator's resolution): **right in kind, under-specified.** Specifically: (a) "up to 27 configurations" at N=200 and N=1,000 costs about 1.6-4.4 GPU-h for one model [I: 27 x (1.4-2.8 + 2.3-7 min)], which is **in no budget line** (A5's gated extras are 5-6 h and are already used); (b) "results that flip are reported as flipping" has no definition of a flip; (c) it does not add the edge-of-grid rule outside the pilot (G5 only); (d) it does not say what happens to the primary 4-trial claim if the wider grid gains more than the margin.
Proposed text: run on B77, Qwen, replicate-0 pool, N=200 and 1,000, a 12-configuration FT grid (4 lr x 3 of rank/alpha or dropout/weight-decay pairs; about 0.7-2.0 GPU-h [I]) and a matched ICL grid (k x retriever x one paraphrased instruction; under $1). Pre-register: if best-of-12 beats best-of-4 by less than m (3 points) on the held-out slice, the 4-trial rule is adequate and reported so; if more, the campaign FT and ICL grids are widened at N<=1,000 only (BillSum N=3,000 stays at cut 1). Add the edge rule (extend the grid by one value on the edge side) to both arms. State that equality is in trial count, not compute. Owners: A3 (protocol), A4 (margin), A5 (GPU line). Do it inside the F3 pilot window.

## 5. Novelty claim against the literature file

Claim: RESEARCH_FOUNDATION gap statement ("none compares tuned small open models with retrieval-augmented cheap commercial APIs on the same labelled pool for N 200-3,000, includes a long-document summarisation curve, **and charges the tuning cost in the break-even**").

| Component | Literature file | Assessment |
|---|---|---|
| Crossover / curve on classification | Pecher (about 100), Spiegelhalter (about 50, N<=200), Bertsch (retrieval vs LoRA on B77/CLINC) | Not novel. N=200 is Spiegelhalter's ceiling; N=1,000-3,000 open-model B77 is Bertsch territory (with test-set checkpointing) |
| Commercial API arm | Pecher: GPT-4o-mini, random ICL; Bertsch: Claude 3.5 Sonnet, one run; Trad and Chehab: Gemini-1.5-Flash retrieval few-shot vs FT encoders | Exists. Our conjunction (retrieval from the same pool, equal-pool LoRA SLM) is narrow. Trad and Chehab is the nearest precedent for the contrast and is **missing from the gap statement** |
| Cost model | Spiegelhalter (setup + per-query, measured throughput), Mansuri ($, break-even) | Not novel as a method |
| "Tuning cost charged in the break-even" | A1 section 1: Mansuri Table 4: about 10,000 reports/year **if the whole 95-GPU-hour sweep is charged** | **Overstated.** RESEARCH_FOUNDATION s.2 (Cost row and gap statement), PAPER_OUTLINE s.1, A1 R2.3-2.4 list as new what A1's own table credits to Mansuri. Our difference: against retrieval-augmented APIs, across tasks and N |
| BillSum N-curve | ELearnFit (XSum, 125 test items, ROUGE-1, no API) | The most defensible new element; search incomplete; metric ROUGE-only unless H2 |
| Methodological repairs | Bertsch App. E test-set checkpointing, 250-item tests | An improvement, not a headline |

Verdict: framing (L-01) defensible; gap statement not yet. Required: credit Mansuri's sweep-charged break-even and Trad and Chehab; read the Spiegelhalter PDF (H4); log direct Semantic Scholar/OpenReview/ACL Anthology searches; keep the workshop-level fallback (if the API arm or BillSum arm fails) in the pre-registration.

## 6. Quality gates

| Gate | Result | Evidence needed to pass |
|---|---|---|
| 1 Literature | **PARTIAL** | Spiegelhalter full text; Semantic Scholar/OpenReview/ACL Anthology searches logged; A1-05 sources checked against PDFs; Mansuri and Trad and Chehab credited correctly |
| 2 Dataset | **PARTIAL** | Similarity-bin reporting implemented and run on validation; B77 label-noise audit or an explicit limitation; incorporated provider policies read (I-18); fix the "below N=77" slip in DATASET_AUDIT |
| 3 Novelty | **PARTIAL** (was FAIL) | Corrected gap statement; paper read; fallback scope pre-registered |
| 4 Methodology | **FAIL** (unchanged: design now coherent, but no evidence the FT arm works and the pilot gate can pass on selection noise) | G1-G7 passed on non-selection rows; merged code; tuning-sensitivity check (section 4); kNN in the inference family (R6) |
| 5 Statistics | **PARTIAL** | Power code committed and reproducible; FT-vs-API discordance from the validation-split API runs (H1) and R re-checked; C7, C8, C10, C11 fixed in the pre-registration text; blinded re-estimation specified (who computes, what is hidden) |
| 6 Economic feasibility | **PARTIAL** (was FAIL) | Measured throughput (A5-11); cap-aware schedule and H3 answered; one request-count script; tuning-sensitivity GPU line; human study funded or ROUGE-only accepted |
| 7 Reproducibility | **PARTIAL** (plan only) | Merged tagged commit with `git_commit`, `pip freeze`, test fingerprint, `test_reads.jsonl`, `analysis/` with sim code, two pre-registration tags. The integrator marks this NOT PASSED; I accept either label, since no artefact exists |
| 8 Adversarial review | **PARTIAL** | This is the independent pass over agents 01-06b and the deliverables; it cannot sign itself off. Needs a re-review after the F3 pilot and after the section-9 fixes |

No PASS. The integrator's NOVELTY_AND_RED_TEAM statuses (4 and 7 NOT PASSED, others PARTIAL) are consistent with mine except gate 7 wording.

## 7. Remaining CRITICAL / HIGH issues, ranked

| Rank | Sev | Issue | Owner | Concrete text change |
|---|---|---|---|---|
| 1 | CRITICAL | FT arm unproven under the corrected prompt; gate evaluated on selection rows, G2/G3 at -5 points (F3, C9) | A3 (gate), A5 | In EXPERIMENT_PROTOCOL s.8: "G1-G6 are evaluated on validation rows not used for selection (rows 201-1,000 at N=200; a fresh 500-row slice at N=1,000). G2: FT >= kNN - 5 at N=200; G3: FT >= kNN - 2 at N=1,000. No pilot pass is evidence for any hypothesis." Keep I-01/I-02 FIXED-UNVERIFIED |
| 2 | HIGH | Tuning bar: L-17 unbudgeted, no flip definition, no edge rule (section 4) | A3, A4, A5 | Replace L-17 by the section-4 text; add the GPU line to FEASIBILITY s.3 |
| 3 | HIGH | Weekly cap vs schedule, GC dependency, carry-over unknown (C5) | A5 | FEASIBILITY s.4: per-week table capped at 13.3 h, "no carry-over" as base case; move GC code to W1 or the BillSum probe to W2; ask H3 now |
| 4 | HIGH | Novelty text overstates (Mansuri sweep-charged break-even, Trad and Chehab omitted); key paper unread | A1 | Rewrite RESEARCH_FOUNDATION gap statement, Cost row and PAPER_OUTLINE s.1: "Existing work prices self-hosting against an API with the tuning sweep charged for one task (Mansuri 2026) ... We add retrieval-augmented commercial APIs from the same pool, N 200-3,000, tasks beyond one, BillSum." Download the PDF (H4) |
| 5 | HIGH | Decision rule inconsistent: a priori vs realised MDG, undefined threshold, ROUGE equivalence unreachable, unfunded R extension, selection uncertainty (C7, C8, C10, C11) | A4 | STAT s.3-4: "A contrast is underpowered a priori iff the MDG from the validation-split replicates (A4-02) exceeds m = 3 points (B77) or 2 ROUGE-L points (BillSum). BillSum has no equivalence outcome. The R extension applies to B77 only (to 10). The CI is conditional on the frozen arms. Provider rule: most accurate; ties within 1 SE go to the cheaper." |
| 6 | HIGH | kNN, the strongest cheap baseline (equal to pilot FT at N=5,000), is outside the inference family; STAT s.4 lists "kNN-vs-FT" as exploratory | A4, A3 | Add: "Secondary, outside the Holm family: FT vs kNN and API-ICL vs kNN per cell with CIs; the decision table reports the best of {kNN, API-ICL, FT} per cell with its cost. If kNN is within m of FT the headline says so." |
| 7 | HIGH | Power rests on unobserved discordance and BillSum SD; code outside repo (I-24) | A4, A5 | Commit `power_sim.py`, `power_sim2.py`, `breakeven.py` to `analysis/` with seeds; run the validation-split API runs ($7.5-11.5, H1) before the protocol tag; "R is frozen after this run" |
| 8 | HIGH | API request counts: protocol and FEASIBILITY disagree on BillSum (23,200/14,200 vs 28,400/22,400); the protocol's "rounding" explanation and `06b` C5 are wrong (section 3) | A3, A5, integrator | One `count_requests.py`; delete the "rounding" sentence; FEASIBILITY header must not claim counts "from A3 Round 2b" while its table is A5's |
| 9 | HIGH | Engineering/reproducibility are plans (F9, F13) | A5, A3 | Make "merged tagged commit + fingerprint + pinned freeze + `analysis/`" the Stage-0 exit criterion with a date (end W1) |
| 10 | MEDIUM-HIGH | Two tags (C6); batch/72 h (C13); incorporated provider policies (I-18) | A3, A2 | Protocol tag W3, arms tag end W4; state the sync fallback; record dated incorporated policies before closing I-18 |

Register: keep I-10 HIGH; I-01/I-02 FIXED-UNVERIFIED; I-04, I-07 IN-REVIEW at best; I-09 OPEN; I-15 HIGH; I-18 OPEN; I-23 and I-24 stay OPEN/IN-REVIEW. Add: I-25 pilot gate on selection data (HIGH), I-26 weekly GPU cap/carry-over (HIGH), I-27 kNN outside the inference family (HIGH), I-28 novelty text overstated (HIGH), I-29 BillSum request-count mismatch (MEDIUM-HIGH).

## 8. Assessment of `06b` (the integrator's consistency check)

Agree: C1 (fixed dangling references), C6 (sim code outside repo; I confirmed), C7 (noisy N=200 selection), C8 (frozen-arm estimand; L-18 handles the label). Disagree or incomplete: C5 is mis-computed (section 3); C9 cites wording ("sanity gates") that is absent from the protocol; C2-C4 "pass" checks were by grep and missed C1-C13 above; L-17 is unbudgeted. `06b` rightly disclaims independence.

## 9. Review of the assembled deliverables (what changed vs what the Round 2 text left broken)

| Deliverable | Verdict | Findings [V by reading] |
|---|---|---|
| RESEARCH_FOUNDATION | Good structure, honest scope; one overclaim | Gap statement and Cost row repeat "tuning charged" as new (section 5). "Spiegelhalter ... for N up to 3,000" fine. Pilot "3.7-5.2 GPU-h" matches the protocol but L-09 still says 3. RQ5 correctly states one classification benchmark |
| DATASET_AUDIT | Accurate to A2 | Section 3 last line "no fine-tuning below N=77" contradicts L-02 / protocol ("below 200"). LEDGAR near-dup at 0.95 present, B77 "-" (A2 gave no count) |
| EXPERIMENT_PROTOCOL | Coherent; defects inherited | "Rounding in the repeat term" is false (C2/C3). Gates unchanged (C9). Section 3 table is A3's BillSum counts while FEASIBILITY uses A5's: two deliverables disagree. Does not state the validation-set size for BillSum or a numeric E_max for FT-final vs trial cost (C4). Retains "arms frozen at tag" without the two-tag fix (C6) |
| STATISTICAL_ANALYSIS_PLAN | Strongest document; defects inherited | s.3-4 keep C7, C8, C10, C11 verbatim; s.7 keeps "25-1,000x" (wrong); kNN-vs-FT exploratory (R6); says "Pre-registration text must copy this section", so these defects would be pre-registered. Correctly flags sim code not in repo |
| FEASIBILITY_AND_BUDGET | Numbers match A5; header misleading | Header says counts come from A3 R2b but the table is A5's. W5-W6 25-45 h vs 26.7 h cap (C5) retained. No budget line for the L-17 sensitivity. Prices consistent with provider page for Haiku [V] |
| NOVELTY_AND_RED_TEAM | Honest; gates conservative | Says remaining contradictions are "API request count, `06b` C5": understated (C1-C13). Gate 8 text correct that independence was outstanding: this file supplies it for agents 01-06b and the deliverables, but not for any later change |
| PAPER_OUTLINE | Fine; no results stated | s.1 repeats the "tuning charged" contribution; s.5.2 six-row table matches the family; limitations list includes the tuning bar and N=200 underpowered |

Overall: the deliverables faithfully assemble the agent files, which is also their weakness: they preserve every Round 2 inconsistency. Fix the section 7 items in the deliverables before the protocol tag; do not tag a pre-registration containing C7/C8/C10.
