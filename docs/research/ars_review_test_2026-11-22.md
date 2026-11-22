# ARS academic-paper-reviewer: test run on the study design (2026-11-22)

Skill under test: `.claude/skills/academic-paper-reviewer/` v1.11.1, applied by hand (text only, no script executed). Mode used: `quick` (field_analyst + Journal-Fit Reviewer; output is an advisory signal, not an editorial decision). Subject: `docs/PREREGISTRATION.md` (draft v2), `docs/foundation/PAPER_OUTLINE.md`, `STATISTICAL_ANALYSIS_PLAN.md`, `EXPERIMENT_PROTOCOL.md`, `RESEARCH_FOUNDATION.md`. The four earlier-review files were not read before Part 1 was written.

Deviations from the skill's protocol, stated up front:
- Checkpoint Rule 1 (stop after Phase 0 and wait for the user to confirm the cards) was not honoured: the caller asked for one continuous output and is not the end user. Cards are shown below and used as written.
- Quick mode has no sprint contract, so the standard (non-contract) Journal-Fit report format was used.
- No manuscript exists, so "Title & Abstract" and "Conclusion" comments are limited to the working title and the outline.
- The five-seat panel, Devil's Advocate, synthesizer, and the script-based attachments (acronym, reference, claim, author-letter checks) belong to `full` mode and were not run.

---

# PART 1. REVIEW (skill format)

## Field Analysis Report (Phase 0)

| Dimension | Result |
|---|---|
| Primary discipline | Applied machine learning / NLP evaluation methodology (LLM adaptation economics) |
| Secondary disciplines | Applied statistics (paired, replicated experimental design); cost engineering / technology economics; summarisation-evaluation validity |
| Research paradigm | Quantitative, confirmatory comparison with a pre-registered protocol |
| Methodology type | Experimental benchmark comparison, replicated, nested pools; statistical modelling (crossed mixed model, Holm, Fieller break-even) |
| Target journal tier | `criteria_binding_unavailable`. The caller said "applied-AI journal, Q2" but gave no confirmed venue/track/article type. Field-general observation: a registered-report-style design for an applied venue; no venue named or benchmarked, per the skill's rule. |
| Paper maturity | Pre-data design stage: protocol drafted, not frozen, pilot partly run (one of five pilot stages). Stage-1 registered-report maturity: not yet. |

## Coverage (Phase 0, "say what you read")

| Part | Status |
|---|---|
| PREREGISTRATION.md sections 1-17, Appendices A-B | read |
| PAPER_OUTLINE.md | read |
| STATISTICAL_ANALYSIS_PLAN.md sections 1-9 | read |
| EXPERIMENT_PROTOCOL.md sections 1-10 | read |
| RESEARCH_FOUNDATION.md sections 1-6 | read |
| Documents the above cite but that were not supplied (`FEASIBILITY_AND_BUDGET.md`, `DATASET_AUDIT.md`, `PREREG_CONSISTENCY.md`, `PREREG_DECISIONS.md`, `DECISION_LOG.md`, `ISSUE_REGISTER.md`, agent files, `results/pilot-*`, simulation code) | not read (not part of the supplied set). Numbers that live only there (prices, token counts, dataset statistics, power-simulation outputs) were taken as the supplied documents state them. Please supply them if a card check below depends on them. |
| Tables, figures, embedded images, appendix data | none (all content is text/markdown tables) |
| Manuscript | none (does not exist) |

## Reviewer Configuration Cards

### Card #1 — Journal-Fit Reviewer (serialized ID: EIC)
**Identity**: Senior applied-ML scholar with editorial experience in empirical NLP/ML-systems work and registered-report formats; no journal named (no confirmed target).
**Review focus**:
1. Check whether the question and claims stay inside what two tasks, two small models and three cheap APIs can support [PREREGISTRATION §2, §3].
2. Check whether the stated contribution is positioned against close prior work without an overstated novelty claim [RESEARCH_FOUNDATION §1-2].
3. Check whether the registration is in a state a Stage-1 reviewer could evaluate [PREREGISTRATION header, §17].
**Cares about**: whether a practitioner could act on the result, and whether the paper survives being "mostly inconclusive".
**Blind spots**: statistical detail (deferred to R1), cost-model internals.

### Card #2 — Peer Reviewer 1 (Methodology)
**Identity**: Experimental-design and applied-statistics specialist (paired/crossed designs, mixed models, multiplicity), experience with ML benchmarking protocols.
**Review focus**:
1. Check whether arm tuning effort is comparable across fine-tuning and in-context arms [PREREGISTRATION §7.1].
2. Check whether the estimand, model and outcome rule match the claims in H1-H4 [§3, §9].
3. Check whether power inputs and the equivalence margin are mutually consistent [§9.4-9.5; STAT §3].
**Cares about**: falsifiability and what is actually frozen.
**Blind spots**: domain literature, practical deployment.

### Card #3 — Peer Reviewer 2 (Domain)
**Identity**: Researcher in parameter-efficient fine-tuning vs in-context learning, including summarisation evaluation.
**Review focus**:
1. Check literature coverage and the novelty wording against the cited close papers [FOUNDATION §2].
2. Check whether ROUGE-L / BERTScore on BillSum can carry a confirmatory hypothesis [PREREGISTRATION §11, H3].
3. Check dataset suitability (public benchmarks, near-duplicates, contamination) [§5.2, §5.3].
**Cares about**: whether the benchmark choice drives the answer.
**Blind spots**: cost economics.

### Card #4 — Peer Reviewer 3 (Cross-disciplinary / practical)
**Identity**: Technology-cost / MLOps practitioner with a procurement-economics angle.
**Review focus**:
1. Check whether the break-even cost model reflects realistic utilisation, hosting and labelling costs [PREREGISTRATION §12; STAT §7].
2. Check who is assumed to benefit and who bears cost: a practitioner who already holds N labels, with GPU access, and API use at scale; what the design leaves out (API fine-tuning, labelling cost, small teams without GPUs) [FOUNDATION §1; PREREGISTRATION §6].
3. Check how long price/model-id facts remain valid [PREREGISTRATION §12, §14].
**Cares about**: decision-grade usefulness over time.
**Blind spots**: statistical inference details.

### Stakeholder Coverage (#1098)
Seat: Card #4 (Peer Reviewer 3), Review Focus item 2
Check: who the design assumes benefits (practitioner holding N labels with GPU access and API budget), who bears the cost, and who is outside the assumed user set; scope stated at [FOUNDATION §1; PREREGISTRATION §6, §12].
Competence basis: procurement-economics and MLOps background.

### Review Strategy Recommendations
- The study is registered-report-like but unfrozen; reviewers should judge the design as a Stage-1 submission would be judged, and say what is missing for that status.
- Tension to watch: R1 (statistical rigour) versus R3 (cost realism) pull on the same H4.

---

## Journal-Fit Review Report (Phase 1, quick mode)

### Reviewer Identity
Senior applied-ML scholar with editorial experience in empirical NLP / ML-systems evaluation and registered-report formats (no venue named; `criteria_binding_unavailable`, so no venue-alignment claim is made).

### Overall Recommendation (advisory signal, not an editorial decision)
**Major Revision** (as a Stage-1 registered-report-style design). The design is unusually careful in several places, but it is not yet a frozen, externally verifiable protocol, two of four hypotheses are close to unfalsifiable as written, and the arm-comparison fairness and power assumptions are not settled.

### Confidence Score
3. Basis: confident on experimental-design and statistics questions; cannot verify model names, prices, dataset statistics or simulation outputs that live in documents not supplied, so those are taken as stated.

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements
Criterion source: `review_criteria_framework.md` section 1 (universal) and 2.1 (empirical research), configured by Card #1 and Card #2.

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty or scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Framework §1 | PARTLY_MEETS | text: FOUNDATION §2 "This is absence in a limited search (about 30 queries" | Honest, modest gap statement; contribution is an extension of cited work | Cannot check the literature myself | yes: contribution is incremental |
| Methodological rigor (design) | Framework §1, §2.1 | PARTLY_MEETS | W2, W3, W4 | Strong statistical core, but fairness of arm tuning and N-axis confound unresolved | Simulation code not supplied | yes |
| Evidence sufficiency (hypothesis testability) | Framework §2.1 "hypothesis clarity" | DOES_NOT_MEET (H2, H4); MEETS (H1, H3 as stated) | W2 | H2 and H4 can be "supported" by low power or by a broad range | none identified | yes |
| Argument coherence | Framework §1 | PARTLY_MEETS | W7, W11 | Question, claims and scope mostly aligned; title and "decision-grade" framing broader than the evidence | none identified | no (fixable by rewording) |
| Statistical reporting plan | Framework §2.1 | MEETS | S2 | Crossed model, Holm, inconclusive rule, MDG tables all pre-specified | Type I error for ROUGE not simulated (authors say so) | no |
| Conclusion conservatism | Framework §2.1 | MEETS | S1, S3 | Forbids "no difference" wording; no "first" claim | none identified | no |
| Significance and impact | Framework §1 | PARTLY_MEETS | W7, W8 | Practically useful question; generalisation limited to 2 tasks and cheap APIs | none identified | yes |
| Pre-registration integrity | Framework §2.1 (internal validity) | DOES_NOT_MEET (at present) | W1, W9 | Not registered, not frozen, enforcement code absent, single researcher | Status is stated honestly by authors | yes |

### Summary Assessment
The planned paper asks when a LoRA-tuned 4B open model should be preferred over retrieval-augmented cheap commercial APIs, given N labelled examples, on BANKING77 and BillSum, with cost charged for tuning. The design is notable for its statistical care: a paired crossed items-by-replicates model, six Holm-corrected contrasts, an explicit inconclusive outcome, kNN as a first-class baseline, a two-tag freeze, and candid limits. As a registered-report-style submission it is not ready. The protocol is a draft with 21 open decisions, no external registry, test-isolation code that does not exist, a pilot that failed one gate and did not run four stages, and power inputs that are assumed rather than observed. H2 and H4 are supported too easily; arm tuning effort is equal in trial count but not in search space; the sampling scheme changes with N, which confounds the crossover bracket; and the BillSum hypothesis rests on ROUGE-L with heavy near-duplicate overlap. The contribution is modest but honest, and a clear "where fine-tuning does not win" result would be useful to practitioners.

### Strengths
1. **S1 Inconclusive-outcome discipline**: three-way outcome with "no difference/equivalent/ties" banned for inconclusive contrasts, plus a bracket-style crossover statement. Evidence Anchor: `text: PREREGISTRATION §9.4 "The words "no difference", "equivalent", "ties" are not used for (iii)."`
2. **S2 Sound inference plan**: rejects pooled McNemar and item-mean t with simulation support, adopts crossed model with a corrected-variance sensitivity. Evidence Anchor: `text: PREREGISTRATION §9.2 "Primary version: uncorrected (conditional label); sensitivity: variance corrected for overlapping pools"`
3. **S3 Honest scoping and failure reporting**: no "first" claim, pilots never pooled, failed runs kept, limits listed. Evidence Anchor: `text: PREREGISTRATION §13 "Nothing is excluded for being inconvenient."`
4. **S4 Strong, fair-minded comparators**: same pool for every arm, byte-identical prompts, kNN reported next to every API number. Evidence Anchor: `text: PREREGISTRATION §6 "kNN | label of the most similar pool item(s), similarity-weighted, no LLM"`
5. **S5 Test-isolation design**: read-ledger, fingerprint, two tags, deviations policy. Evidence Anchor: `text: PREREGISTRATION §7.2 "No test row is read before the arms tag."`

### Weaknesses
1. **W1 Not a verifiable, frozen registration**: no registry, no external timestamp, enforcement code and fingerprint file absent, 21 open decisions, one pilot gate failed and four stages not run. A Stage-1 reviewer cannot approve something whose grids, arms and tests may still change. Suggested fix: freeze and deposit (registry or third-party timestamp), write and commit `final_eval` and the fingerprint before submission, close the section-17 items, and rerun the pilot gates.
   - **Severity**: Major | **Evidence Anchor**: `text: PREREGISTRATION §5.3 "This file does not exist yet"` | **Confidence**: 4 (design-review basis)
2. **W2 H2 and H4 are close to unfalsifiable**: H2 is "supported" when the outcome is not "FT better", yet the authors expect (iii) there, so H2 is supported by lack of power. H4 is supported if Q* is finite and below 10^7 at utilisation 1 and synchronous pricing, the most favourable setting, over a four-decade range that the authors' own projections already fall inside. Suggested fix: define H2 as a bracket statement with a minimum-power condition (or drop it to descriptive), and make H4 a statement about a realistic utilisation range or a tighter volume threshold.
   - **Severity**: Major | **Evidence Anchor**: `text: PREREGISTRATION §3 H4 "the tuning-charged break-even volume Q* exists and lies inside the pre-specified volume range 10^3..10^7 queries"` | **Confidence**: 4 (inference design)
3. **W3 Tuning-effort asymmetry between arms**: effort is equalised in trial count, but fine-tuning searches learning rate, schedule and checkpoints while the in-context arm searches k and (BillSum) retriever with untuned prompt text. The authors acknowledge "equality is in trial count", and the 12-configuration sensitivity covers only BANKING77, Qwen, N=200 and 1,000, with the held-out slice still open. Direction of the bias is not stated. Suggested fix: extend the sensitivity (prompt paraphrase, shot ordering, system text) to the API arm in at least one cell per task, state the expected direction, and decide D-07 now.
   - **Severity**: Major | **Evidence Anchor**: `text: PREREGISTRATION §7.1 "Equality is in trial count, not compute"` | **Confidence**: 4 (experimental-design basis)
4. **W4 Sampling scheme changes with N**: N=200 uses class-stratified pools, N>=1,000 random pools, and the primary crossover statement is a bracket across those cells. The N effect and the sampling effect are confounded in exactly the contrast the question is about; the R=3 random-pool N=200 check is exploratory. Suggested fix: promote a random-pool N=200 condition to a pre-specified sensitivity on the bracket, or state the bracket separately for each sampling scheme.
   - **Severity**: Major | **Evidence Anchor**: `text: EXPERIMENT_PROTOCOL §2 "class-stratified round-robin only at N=200"` | **Confidence**: 4
5. **W5 Power inputs unobserved and margin inconsistent with achievable precision**: power inputs are model-vs-model proxies, "unobserved for FT-vs-API"; at N=200 the pessimistic MDG exceeds the equivalence margin even at R=10, so outcome (ii) is effectively unreachable there; the margin m=3 is justified as "smaller than the gap at which the cost ranking could change", which is a claim, not a derivation. Suggested fix: run the validation-split API replicates before freezing R and m, and derive m from the cost model with the actual numbers.
   - **Severity**: Major | **Evidence Anchor**: `text: PREREGISTRATION §8 "stays above m = 3 even at R=10 (4.4 points)"` | **Confidence**: 4
6. **W6 BillSum hypothesis (H3) rests on ROUGE-L with likely artefacts**: the authors say gaps under about 1 ROUGE-L point are not interpretable; ROUGE rewards style match, which favours the fine-tuned arm; and 16.9% of BillSum test bills have cosine >= 0.9 to training bills, so retrieval and memorisation can copy summaries. No smallest effect of interest is stated for BillSum, so a Holm-significant 0.5-point gap would count as "FT better". Suggested fix: add a minimum meaningful ROUGE-L difference, make the near-duplicate-excluded analysis co-primary, or demote H3 to exploratory unless the human study runs.
   - **Severity**: Major | **Evidence Anchor**: `text: PREREGISTRATION §5.2 "BillSum 551 (16.9%) at >= 0.9"` | **Confidence**: 3 (summarisation-evaluation basis; artefact direction is my inference)
7. **W7 Estimand and framing narrower than the question**: the estimand is conditional on a fixed training set and on frozen arms chosen on 200 validation items at N=200; selection noise lies outside the CI. The title and "decision-grade" framing speak to "a tuned 4B model" versus "a cheap API" generally, while scope is two tasks, two small models, three cheap APIs, one GPU, and no API fine-tuning (the option a practitioner with N labels may actually take). Suggested fix: retitle and reword the research question to the tested scope; state API fine-tuning as a named excluded alternative in the introduction.
   - **Severity**: Major | **Evidence Anchor**: `text: PREREGISTRATION §7.2 "selection uncertainty (noisy at N=200: 200 items, binomial SE about 3 points) lies outside it"` | **Confidence**: 3
8. **W8 Cost model rests on imputed and best-case inputs**: GPU rate is an assumed $3.6-6.0/h band, utilisation 1 is the headline scenario, throughput is not yet measured, labelling cost dominates but is outside the confirmatory family, and API prices and ids were read through fetch agents on one day. The authors' own reading is that "a break-even exists only if the GPU stays busy and the API is called synchronously", so the headline H4 setting is the narrowest case. Suggested fix: make a mid-utilisation scenario co-primary, measure throughput before freezing, and report labelling and operations cost as part of the main cost table.
   - **Severity**: Major | **Evidence Anchor**: `text: STATISTICAL_ANALYSIS_PLAN §7 "a break-even exists only if the GPU stays busy and the API is called synchronously"` | **Confidence**: 4 (cost-analysis basis)
9. **W9 Independence of design decisions and evidence base**: a single researcher computes the "blinded" variance re-estimation and scores all human ratings; design decisions were "taken by the assistant under the researcher's delegation"; literature claims are partly agent-summarised and one key source is unread. A journal reviewer would ask for disclosure and for human sign-off on each decision that changes the confirmatory family. Suggested fix: disclose the AI-assisted workflow, have the supervisor or a colleague run the blinded step, read the key sources directly.
   - **Severity**: Major | **Evidence Anchor**: `text: PREREGISTRATION §17 "taken by the assistant under the researcher's delegation"` | **Confidence**: 3 (process, not statistics)
10. **W10 Positioning rests on a limited search**: the gap statement is acknowledged as "absence in a limited search", and the closest dollar-break-even work already charges tuning cost. The contribution is an incremental extension. Suggested fix: repeat the search with the unsearchable indexes before submission and narrow the contribution list.
    - **Severity**: Minor | **Evidence Anchor**: `text: RESEARCH_FOUNDATION §2 "This is absence in a limited search (about 30 queries"` | **Confidence**: 3
11. **W11 Naming and rule ambiguities**: `H2` is used for the human-study decision in the outline and foundation but for a BANKING77 hypothesis in the pre-registration; it is unclear whether the "selected provider" is chosen per cell or once; "kNN within m of FT" has no stated test (point estimate or CI). Suggested fix: rename and define each.
    - **Severity**: Minor | **Evidence Anchor**: `text: PAPER_OUTLINE §5 "Human faithfulness validation, 60 bills, if H2 resolves"` | **Confidence**: 4

### Detailed Comments

#### Journal Fit
Venue-neutral. The format (a registered-report-style design for an applied-AI venue) is a good fit for the question, provided the registration is actually frozen and deposited. Without a confirmed target, no venue-specific claim is made.

#### Originality
Honest and incremental. The authors themselves list the close prior work and drop the "first" claim. The value is in the equal-pool, stated-effort, tuning-charged comparison against retrieval-augmented cheap APIs.

#### Significance
Useful to practitioners if the result is robust, including a "fine-tuning does not win here" outcome. Limited by W7 and W8.

#### Structural Coherence
Question, hypotheses, estimand and outline line up, except W2 (hypotheses that cannot fail), W7 (title versus scope) and W11 (naming).

#### Title & Abstract
Working title "When does a tuned 4B model beat a cheap API with retrieval?" promises more generality than two tasks support.

#### Conclusion
No manuscript yet; the outline's discussion plan ("bracketed crossover; when kNN or the API suffices") is appropriately conservative.

### Questions for Authors
1. What result would make you declare H2 and H4 false?
2. Which direction do you expect the tuning-effort asymmetry to bias the FT-vs-ICL contrast, and what data would show it is small?
3. Will the near-duplicate-excluded analysis be reported as primary if it reverses a contrast?
4. Who will run the blinded variance re-estimation, and what exactly is masked?
5. How will you decide, before freezing, whether the 3-point margin is the right one, given that the validation-split API data do not yet exist?

### Minor Issues
- Several parameters differ between documents (learning-rate grid, BillSum request counts); the pre-registration lists them as open, which is fine, but they should be reconciled before the tag.
- The research documents use internal ids (L-xx, I-xx, PC-xx) heavily; a Stage-1 submission needs a self-contained text.

---

# PART 2. COMPARISON WITH THE EXISTING REGISTER AND 06c
(Added after reading `ISSUE_REGISTER.md`, `NOVELTY_AND_RED_TEAM.md`, `DECISION_LOG.md`, `agents/06c-independent-redteam.md`.)

Labels: **already known** (the register, decision log or 06c already says it), **new** (not found there), **wrong or unfair** (my finding is stale, overstated or contradicted by material I had not been given). Where a finding is part-known and part-new it is split.

| My finding | Verdict | Where it already sits / reason |
|---|---|---|
| W1 registration not frozen, no registry, enforcement code and fingerprint absent, pilot not complete | already known | 06c F3, F13, gates 4 and 7; I-24; PREREGISTRATION §16-17 say it themselves. |
| W1 "21 open decisions" | wrong or unfair (partly) | The decisions were taken in `PREREG_DECISIONS.md` (L-28, L-29, L-30), which I was not given. The open-items table in the draft is stale, not the study. The real defect is that the draft file was not updated to match. |
| W2 H2 "supported" by low power | already known (in substance) | PREREGISTRATION H2 row says "(iii) is the expected outcome"; 06c F1 accepts N=200 only because no crossover claim is made. My sharper point (H2 cannot fail except one way) is a rewording, not a new finding. |
| W2 H4 true by construction at u=1, sync, over 10^3..10^7 | new | Not in the register or 06c. 06c C-items and I-15/I-22 cover cost-input reliability, not H4's falsifiability. |
| W3 tuning asymmetry (trial count not compute; prompt untuned; sensitivity only on B77/Qwen) | already known | I-23, L-14, L-19, 06c §4. |
| W3 "direction of bias not stated" | wrong or unfair | 06c §4 point 3 states the direction (under-tuned FT flatters ICL, untuned prompts do the opposite). L-19 already includes a paraphrased-instruction ICL grid. |
| W4 stratified N=200 vs random N>=1,000 | already known | F4, I-19, L-04 (named separate condition, R=3 table). |
| W4 bracket straddles the sampling-scheme change | new (small) | Specific consequence for the bracket statement; not discussed in 06c. |
| W5 power inputs unobserved for FT-vs-API | already known | F7, I-24, A4-02, 06c gate 5, PREREGISTRATION §9.5. Note L-32: with no API money, the validation-split API runs cannot happen, so this is worse than I wrote; I missed that. |
| W5 equivalence margin m=3 unreachable at N=200 and circularly justified for BANKING77 | new (partly) | 06c C10 covers the ROUGE margin only; PC-11/D-10 cover the R trigger. The B77 "(ii) unreachable at N=200" and the "cost ranking could change" justification are not examined there. |
| W6 ROUGE-L validity for a confirmatory hypothesis | already known | I-13, F10, 06c C10, L-10. |
| W6 near-duplicates in BillSum (16.9%) | already known | I-08, §5.2 near-duplicate bins and novel-text sensitivity. |
| W6 retrieval/memorisation can copy near-duplicate summaries; ROUGE favours style match; no smallest effect of interest for H3 | new | Not in register or 06c. The artefact direction is my inference, labelled as such. |
| W7 conditional estimand, selection noise outside CI | already known | L-18, 06c C11. |
| W7 title/"decision-grade" framing broader than scope; API fine-tuning as an unexamined option | new (framing) | Limitation is stated in PREREGISTRATION §6 and RQ5, but not raised as a framing or decision-relevance issue. Reviewer-opinion level, not a defect. |
| W8 cost inputs imputed, u=1 headline, throughput unmeasured, labelling dominates | already known | I-15, I-22, F11, L-32 (imputed band, no dollar basis), STAT §7. |
| W8 H4 headline uses the single most favourable utilisation | new | Same as the W2 H4 point. |
| W9 single researcher runs the "blinded" step and scores all ratings | already known | D-09, 06c §9. |
| W9 AI-assisted decisions need disclosure to a journal | new (process) | L-28 records the delegation internally; no item about disclosure or human sign-off. |
| W9 "one key source is unread" | wrong or unfair | Spiegelhalter full text was read 2026-11-22 (L-21; notes file cited in PREREGISTRATION). `RESEARCH_FOUNDATION.md` still says "unread", so the stale sentence misled me; that inconsistency between documents is real and small. |
| W10 novelty rests on a limited search | already known | I-10, I-28, 06c §5. |
| W11 H2/H5 vs HD naming collision persists in outline and foundation | new (minor) | PC-24 renamed it in the pre-registration only. |
| W11 provider selected per cell or once; "kNN within m" has no stated test | new (minor) | 06c C11 touches provider selection but not the per-cell ambiguity. |

Findings in the existing register and 06c that my review did **not** raise: pilot gates evaluated on selection rows (I-25, 06c C9); weekly GPU cap vs schedule (I-26, C5); BillSum API request-count mismatch (I-29, C2); type-I error under overlapping pools as a risk (I-04; I only noted the plan's treatment); licences and provider terms (I-11, I-18); contamination of public benchmarks (I-17); the Phi and OLMo training failures (I-14); and, most important, that the API arm is currently unfunded (L-32). Several of these were out of reach because their sources were not in the supplied set, but a Stage-1 reviewer would have asked about licences, contamination and ethics.

Tally of my 11 weaknesses: 6 mainly already known (W1, W3, W5-power, W7-estimand, W9-blinding, W10), 3 containing a genuinely new element (W2-H4, W6-copying artefact, W4-bracket), 2 minor and new (W11, W9-disclosure); 3 contained a statement that was wrong or unfair (W1 "21 open", W3 "direction not stated", W9 "unread"). No finding contradicts the register's top-level verdict (go for Stage 0, no go for the campaign); my "Major Revision" is consistent with 06c's.

---

# PART 3. ASSESSMENT OF THE SKILL

## What the skill added versus a plain careful review

**Caveat on the test itself.** I read the study design closely and then applied the skill in the same pass, so I cannot separate "what the skill found" from "what a careful reader finds". I did not run a plain review first. Treat the judgement below as one run, one model, no control.

**Structure (useful).**
- The Coverage section made me state which documents I had not read, and the register comparison shows that this mattered (W1, W9 and W5 were each weakened by sources I never saw). A plain review would have been more confident than it should have been.
- The rule that every finding carries Severity, Confidence with a competence basis, and a typed Evidence Anchor (verbatim quote of 25 words or fewer) forced me to ground each claim in the text and made the comparison table fast to build. Every quote traced back to the files.
- "Cards set checks, not conclusions" kept the configuration step from pre-judging the study.
- The `NOT_CALIBRATED` label and the "role separation is not independence" wording are honest about what a simulated panel is.

**New insights (few).** About three: H4 is nearly true by construction under its own assumptions; the BillSum near-duplicate/style-match artefact with no smallest effect of interest; the bracket crossing a sampling-scheme change. Roughly half of my other findings were already in the register or 06c. The skill's contribution to these three was indirect (the "evidence sufficiency" and "so what" prompts in `review_quality_thinking.md`), not a method I could not have used otherwise. I would not claim the skill found anything a competent plain review would miss.

**Noise and mismatch.**
- Quick mode is under-specified. SKILL.md says quick = field_analyst + Journal-Fit Reviewer, but the Journal-Fit agent file is about 29 KB, most of it sprint-contract grammar, dissent rules and structural preflights that apply only to `full` and `methodology-focus`. There is no quick template; I inferred the format from the Journal-Fit output block.
- The Journal-Fit seat is told not to go deep into methodology. For a study design, methodology is the main question, so quick mode points at the wrong lens. I folded methodology into the one report, which departs from the seat's definition.
- The field analyst produces four cards but quick mode runs one reviewer, so cards 2-4 and the stakeholder seat were produced and then unused.
- Journal-fit and originality framing assume a manuscript and a named venue. With neither, those parts shrink to "venue-neutral".
- Cost: SKILL.md (56 KB) plus two agent files (46 KB) plus a few references, roughly 28k tokens of reading before the subject. The review itself was about the same length as a plain one, plus the cards and coverage table.
- `top_journals_by_field.md` lists high-impact general AI journals, not applied-AI Q2 venues; it was not useful (and the skill forbids naming a venue without a confirmed target anyway).

**Unsafe or off (read-only observations from the skill text; nothing executed).**
- The installed copy is incomplete. SKILL.md refers to `scripts/*.py` (synthesis, acronym, reference, claim and letter checkers), `shared/*` (contracts, ground-truth isolation pattern, cross-model verification) and `docs/DATA_FLOWS.md`, none of which exist in `.claude/skills/academic-paper-reviewer/` (only SKILL.md, references, agents, templates, examples). In `full` mode these checks could not run, and the skill's fallback wording ("the section is one line saying the check did not run") makes that visible.
- Data flows in `full` mode, per SKILL.md text only: the suggested-reference check "looks each work up in the four bibliographic indexes", sending suggested titles and identifiers; the claim check sends the five reports plus the manuscript to a subagent; the cross-model track uploads the manuscript to an external provider, gated on a configured environment variable and explicit user consent. The first appears not to be consent-gated in the text (the claim check has an off switch, `ARS_PANEL_CLAIM_CHECK=0`). For an unpublished study design that is a minor but real disclosure to consider before using `full` mode. I did none of these.
- Instruction handling is sound: the skill states repeatedly that manuscript text is data and that directives inside it are findings, not commands. I saw no such directives in the subject files.
- The Iron Rule that the session must stop after Phase 0 and wait for the user suits interactive use and conflicts with a one-shot subagent run; I deviated and said so. Skipping the pause is allowed only on the user's own instruction, which the caller is not.
- The persona examples ("Senior Associate Editor of ...", "20 years of editorial experience") are invented identities by design; the skill restricts naming a journal to a user-confirmed target, which limits the risk of fabricated affiliation claims.
- Nothing in the text I read asked me to install anything or contact an external service during `quick` mode.

**Bottom line.** Worth keeping as a checklist for finding format, evidence anchors and the honesty labels; not worth the reading cost for a quick design review, and quick mode needs its own short template and a methodology-capable reviewer seat before it fits a study-design review. Its value would likely be higher in `full` mode with a real manuscript, which this test did not exercise.
