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
