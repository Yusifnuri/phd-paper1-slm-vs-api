# Research foundation: how this folder works

Goal: a defensible, publication-oriented research foundation before any expensive experiment campaign. Nothing here is a result; there are no confirmatory runs yet.

Roles (each a separate Claude session, so their work is genuinely independent; the coordinator is also one):

| ID | Role | Draft file | Becomes |
|---|---|---|---|
| A1 | Literature and novelty | `agents/01-literature.md` | `RESEARCH_FOUNDATION.md`, `NOVELTY_AND_RED_TEAM.md` |
| A2 | Datasets, licensing, provider terms | `agents/02-dataset-licensing.md` | `DATASET_AUDIT.md` |
| A3 | Experimental methodology | `agents/03-methodology.md` | `EXPERIMENT_PROTOCOL.md` |
| A4 | Statistics and evaluation | `agents/04-statistics.md` | `STATISTICAL_ANALYSIS_PLAN.md` |
| A5 | Compute, API and cost | `agents/05-feasibility.md` | `FEASIBILITY_AND_BUDGET.md` |
| A6 | Red team | `agents/06-redteam.md` | `NOVELTY_AND_RED_TEAM.md` |
| A7 | Integrator (coordinator) | `ISSUE_REGISTER.md`, `DECISION_LOG.md`, `HUMAN_DECISIONS.md` | all nine deliverables |

Rules for every agent: primary sources only (full text, not abstracts or search snippets, for any methodological claim); every factual claim carries source URL, access date, location (section/table/page), a ≤15-word quote and a confidence level; label each statement as verified fact, author interpretation, own recommendation or unverified assumption; never invent a reference, price, licence permission or result. Existing pilot results in `results/pilot-2026-11-21/` are exploratory and must not be overwritten. Each agent writes only its own file under `agents/` and does not commit; the integrator commits. Do not escalate to the human what research can settle: choose the safest defensible option, document the assumption in your file, and list only genuinely human-dependent decisions (legal exposure that evidence cannot settle, budget or credentials you cannot infer, a change of research question).

Earlier material this builds on, not to be repeated from scratch: `docs/research/verification/00-summary.md` (and reports 01–05), `docs/research/2026-11-21-sources.md`, `docs/PLAN.md`, `docs/PREREGISTRATION.md` (draft), `docs/HANDOFF.md`. Treat those as claims to be checked, not as facts.
