# Novelty assessment, red-team outcome and quality gates

Written 2026-11-22 by the integrating session. Sources: `agents/01-literature.md` (incl. Round 2), `agents/06-redteam.md`, `agents/06b-redteam-second-pass.md`, `DECISION_LOG.md`, `ISSUE_REGISTER.md`. **Nothing here is a result.** A gate is marked PASSED only when the evidence is in the repository and an independent reader could check it; otherwise it is PARTIAL or NOT PASSED, with what is missing.

## 1. Novelty

- Prior work already shows the fine-tuning/ICL crossover at tens to hundreds of labels for classification and narrow tasks (Pecher 2025; Spiegelhalter, Franke, Hutter 2026; Bertsch 2025) and retrieval ICL competing with LoRA on BANKING77/CLINC150 (Bertsch 2025). The study must not claim a "first" crossover (L-01).
- Defensible contribution, as a **conjunction** and only as "absent in the limited search I could do" (about 30 queries, see `RESEARCH_FOUNDATION.md` §2): tuned small open models vs retrieval-augmented cheap commercial APIs on the same labelled pool, N from 200 to 3,000; a BillSum label-budget curve; a break-even with the tuning cost charged; stated equal tuning effort; honest reporting where fine-tuning does not win.
- Update 2026-11-22: the full text is now read (`docs/research/spiegelhalter-2026-notes.md`, L-21): workshop poster, local Qwen3 0.6-4B only, N <= 200, 17 mostly synthetic tasks, no commercial API arm, no summarisation, RAG excluded from costs; it does charge HPO in LoRA's setup cost, so that is no longer a distinguishing claim. The paragraph below is kept for history.
- (Superseded) Weakest point: the full text of Spiegelhalter et al. 2026 was not read (abstract and public repository only; H4). The authors' repository shows no commercial API arm, no BANKING77/LEDGAR, no summarisation and N ≤ 200 (L-13), but a statement about the paper itself is unverified. I-10 is HIGH, not resolved.
- A mixed or null outcome is a plausible and publishable result (L-01).

## 2. Red-team outcome

- First pass (`06-redteam.md`): verdict "reject as designed". Critical findings: confirmatory grid beyond the known crossover (now in the family via L-03/L-15); fine-tuned arm unproven under the corrected prompt (gated pilot, L-09); contribution narrower than claimed (L-01, L-13); 15 cross-draft contradictions (superseding rule; remaining: API request count, `06b` C5).
- Second pass (`06b`): **consistency check by the integrator, not an independent agent.** It found two dangling references (fixed), an unreconciled API request count, power-simulation code outside the repository, noisy tuning selection at N=200, and proposed the resolution of the tuning-bar question (L-17, L-18).
- The 06b file says outright that it is not independent. An independent second pass is outstanding.

## 3. Quality gates

| # | Gate | Status | Evidence in the repository | Missing before PASSED |
|---|---|---|---|---|
| 1 | Literature | PARTIAL | `agents/01-literature.md`, `RESEARCH_FOUNDATION.md` §2, §5 (primary sources with access dates and quotes live in the agent file) | Spiegelhalter 2026 full text (H4); several sources were read via fetch agents and are partly secondary |
| 2 | Dataset and licences | PARTIAL | `DATASET_AUDIT.md`, `agents/02-dataset-licensing.md`; L-07, L-08 cut the risky data (SAMSum, LEDGAR release) | Independent confirmation of BillSum and LEDGAR rights statements; provider terms incorporated by reference (I-18) |
| 3 | Novelty | PARTIAL | Section 1 above | Spiegelhalter paper text; a second literature check by someone other than A1 |
| 4 | Methodology | NOT PASSED | `EXPERIMENT_PROTOCOL.md` (design complete on paper) | The fine-tuned arm is unproven under the corrected prompt; gated pilot G1-G7 has not run; code branches are not merged; tuned-procedure code does not exist |
| 5 | Statistics | PARTIAL | `STATISTICAL_ANALYSIS_PLAN.md` (estimand, crossed model, 6-contrast Holm, MDGs, Q\* with Fieller) | Simulation code is not in the repository (`06b` C6/I-24); inputs for FT-vs-API discordance and replicate SD are unobserved until the validation-split API runs (H1); continuous-score Type I error not simulated |
| 6 | Economic feasibility | PARTIAL | `FEASIBILITY_AND_BUDGET.md` (35.5-112 GPU-h needed vs about 54-70 available; API $93-140, up to about $220; cut ladder; go/no-go at week 2) | Throughput unmeasured (1-h benchmark); GPU credit value unknown (H3); thinking-token cost unmeasured; request count inconsistency (C5) |
| 7 | Reproducibility | NOT PASSED | Per-prediction and per-run schemas, pre-registration tag plan, `frozen_arms.json` plan in the protocol | Three code branches unmerged; no tagged commit; sims outside the repo; no `analysis/` folder; pinned requirements not yet reconciled with the server |
| 8 | Adversarial review | NOT PASSED | `06-redteam.md`, `06b` | Independent second pass over all agent files and deliverables; CRITICAL/HIGH issues (I-01, I-02, I-04, I-10, I-23, I-24) have no independent sign-off |

Conclusion: the foundation is **ready for the researcher's decisions (H1, H3, H4, H5) and for Stage 0 engineering**, not ready for any GPU campaign or API spending. No gate is marked PASSED; the fine-tuned arm and the novelty claim carry the most risk.

## 4. What would change the plan

- Pilot gates G1-G7 fail: stop, diagnose (Phi first-step loss, OLMo DoRA NaN), do not run the campaign.
- Spiegelhalter 2026 contains a commercial API arm or a retrieval comparison at N ≥ 1,000: re-open I-10 as CRITICAL and revisit the contribution.
- The wider tuning grid (L-17) flips an N=200 or N=1,000 conclusion: report the flip; the primary 4-trial claim is then stated as tuning-budget dependent.
