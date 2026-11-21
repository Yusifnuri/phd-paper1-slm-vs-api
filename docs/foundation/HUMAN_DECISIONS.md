# Human decisions (draft)

Only items that evidence cannot settle and that need the researcher's access, authorisation or preference. Everything else is decided in `DECISION_LOG.md`.

| # | Question | What was already done | Why the agents cannot decide | Options and consequences | Recommendation |
|---|---|---|---|---|---|
| H1 | May the agents make live API calls with the researcher's own provider keys (about $0.05 for 50-item tests per provider, then ≈$5–15 for the 5-replicate validation-split API runs that A4's power plan needs)? | A3, A5, A6 documented every question these calls settle (temperature, thinking off, logprobs, token counts, flip rate) | needs the researcher's keys and spending authorisation | yes: settles I-09 and the power plan; no: API protocol stays unverified and R cannot be frozen | yes, with a hard cap of $20 for this stage |
| H2 | Is there a second annotator (a volunteer or a paid one) for a small human faithfulness study on BillSum summaries (60 bills × 3 systems)? | A4 designed the protocol and costed the large version ($2–5k) | needs a person | yes: validates AlignScore/judge and supports a summarisation quality claim; no: summarisation stays ROUGE/BERTScore-only and exploratory | ask a legal- or policy-trained volunteer; otherwise accept the exploratory framing |
| H3 | What is the credit-to-dollar value of the university allowance and does the unused weekly credit carry over? | A5 could not find either; the rental band $3.6–6.0 per hour is used instead | only visible in the researcher's portal | affects the imputed GPU price in the cost analysis and the schedule | check the portal once |
| H4 | The full text of Spiegelhalter et al. 2026 (only the abstract was read), if every automated route fails | A1 tries arXiv, OpenReview, Semantic Scholar, authors' pages | access may need a signed-in browser | without it the novelty gate cannot pass | download the PDF only if A1 reports failure |
| H5 | Is a preprint plus workshop first, then a journal, acceptable as the plan? | A1 compared venues and review times | depends on the researcher's goals | a journal decision within two months is not realistic on the evidence | accept the staged plan |

Legal items that were candidates (SAMSum NC-ND, LEDGAR third-party exhibits, BillSum CRS summaries) no longer need a decision: SAMSum is excluded and only ids/labels/metrics are released for LEDGAR (L-07, L-08). Optional: written confirmation from the dataset authors.
