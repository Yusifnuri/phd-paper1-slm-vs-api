# Human decisions (draft)

Only items that evidence cannot settle and that need the researcher's access, authorisation or preference. Everything else is decided in `DECISION_LOG.md`. **Naming (2026-11-22):** to avoid the clash with the hypotheses H1-H4 of `PREREGISTRATION.md`, these rows are called **HD-1..HD-5** there and here; `DECISION_LOG.md` and `ISSUE_REGISTER.md` still write them H1..H5 (same numbers). The pre-registration items D-01..D-21 are in `PREREG_DECISIONS.md`; five of them stay with the researcher and are mapped in the table at the end of this file.

| # | Question | What was already done | Why the agents cannot decide | Options and consequences | Recommendation |
|---|---|---|---|---|---|
| HD-1 (H1) | May the agents make live API calls with the researcher's own provider keys (about $0.05 for 50-item tests per provider, then ≈$5–15 for the 5-replicate validation-split API runs that A4's power plan needs)? | A3, A5, A6 documented every question these calls settle (temperature, thinking off, logprobs, token counts, flip rate) | needs the researcher's keys and spending authorisation | yes: settles I-09 and the power plan; no: API protocol stays unverified and R cannot be frozen | **ANSWERED 2026-11-22: not possible now (the researcher has no API keys and no money).** Consequences are in L-32; every API-dependent item stays open |
| HD-2 (H2) | Is there a second annotator (a volunteer or a paid one) for a small human faithfulness study on BillSum summaries (60 bills × 3 systems)? | A4 designed the protocol and costed the large version ($2–5k) | needs a person | yes: validates AlignScore/judge and supports a summarisation quality claim; no: summarisation stays ROUGE/BERTScore-only and exploratory | ask a legal- or policy-trained volunteer; otherwise accept the exploratory framing |
| HD-3 (H3) | What is the credit-to-dollar value of the university allowance and does the unused weekly credit carry over? | A5 could not find either; the rental band $3.6–6.0 per hour is used instead | only visible in the researcher's portal | affects the imputed GPU price in the cost analysis and the schedule | **ANSWERED 2026-11-22: the portal shows no currency value because the university provides the credits; no dollar price exists.** The imputed rental band stays the only cost basis (L-32); carry-over still unknown |
| HD-4 (H4) | Access to the full text of Spiegelhalter et al. 2026 (earlier only the abstract was read) | A1 tried arXiv, OpenReview, Semantic Scholar, authors' pages | access needed a signed-in browser | without it the novelty gate could not pass | **RESOLVED 2026-11-22:** the researcher downloaded the PDF; notes in `docs/research/spiegelhalter-2026-notes.md`, L-21 |
| HD-5 (H5) | Is a preprint plus workshop first, then a journal, acceptable as the plan? | A1 compared venues and review times | depends on the researcher's goals | a journal decision within two months is not realistic on the evidence | **ACCEPTED 2026-11-22** (L-11 stands) |

Legal items that were candidates (SAMSum NC-ND, LEDGAR third-party exhibits, BillSum CRS summaries) no longer need a decision: SAMSum is excluded and only ids/labels/metrics are released for LEDGAR (L-07, L-08). Optional: written confirmation from the dataset authors.

## Researcher-owned items still open (2026-11-22)

| ID | Item | Source | Status |
|---|---|---|---|
| HD-6 | Route for the commercial API arm: provider research or student-credit programmes; a minimal single-provider BANKING77 arm (about 28k requests per `HANDOFF.md`, not reproduced here); or dropping the commercial API arm and re-scoping to open-weight comparisons (Google's unpaid tier is not used for results because it trains on inputs, L-08). Drives which variant of the paper is written | L-32, `PREREGISTRATION.md` s.2.1, `PC-29` | **open; the main decision** |
| HD-2 | Annotator for the BillSum faithfulness study (D-13) | D-13 | open; default is no human study |
| HD-3 (D-19) | The credits carry no dollar value (answered); weekly carry-over still unknown | L-32 | partly answered |
| HD-7 | Authors, affiliation, funding, data-availability statement, registry choice (OSF or none) | D-18 | open |
| HD-8 | Confirmation of provider terms and dataset licences before any release of predictions or adapters; no release before that | D-21, L-31 | open |
| HD-9 | API model id strings, snapshot dates, SDK and price-page versions at the first live call | D-16 | open; needs HD-1 (keys), none exist |
| HD-10 | The decisions listed in `PREREG_CONSISTENCY.md` section F (`PC-29`..`PC-40`): what the G4 miss means, non-selection validation rows at N >= 1,000, the N=3,000 learning-rate edge, the API k grid and the BillSum trial count, the batch-vs-sync base budget, the provisional go/no-go | `PREREG_CONSISTENCY.md` F | open; each row names its owner |
