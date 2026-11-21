# A6b: Second red-team pass (coordinator consistency check, 2026-11-22)

**Independence caveat (read first).** This pass was written by the integrating session, not by a fresh independent agent. It checks the six assembled deliverables against the decision log and against each other; it is NOT the independent review that gate 8 needs for CRITICAL/HIGH issues. An independent pass over `agents/01..05`, `06-redteam.md` and the deliverables is still outstanding (see `NOVELTY_AND_RED_TEAM.md`, gate 8). Labels: [V] verified by reading the files named, [I] interpretation, [U] unverified.

## 1. Checks and results

| # | Check | Result | Sev |
|---|---|---|---|
| C1 | Deliverables cite only files that exist | **Fail, fixed.** `RESEARCH_FOUNDATION.md` said "assembled from `agents/01..06b`" while 06b did not exist; `EXPERIMENT_PROTOCOL.md` L-14 pointed to a `NOVELTY_AND_RED_TEAM.md` that did not exist [V]. Both now resolve (this file; `NOVELTY_AND_RED_TEAM.md`). | MEDIUM |
| C2 | N grid identical across documents (L-02: 200/1,000/3,000; no N=5,000, N=50 fine-tuning) | Pass in `RESEARCH_FOUNDATION`, `EXPERIMENT_PROTOCOL`, `STATISTICAL_ANALYSIS_PLAN`, `DATASET_AUDIT`; N=5,000 appears only as a dropped/background item [V by grep]. | – |
| C3 | Confirmatory family = 6 contrasts, Holm (L-03, L-15) | Consistent in `STATISTICAL_ANALYSIS_PLAN` §Family, `RESEARCH_FOUNDATION` §3, `PAPER_OUTLINE` §4 [V]. | – |
| C4 | Replicates BANKING77 R=8, BillSum R=5 (L-06) | Consistent in protocol and plan [V]. | – |
| C5 | API request count | **Unreconciled.** Protocol §3 gives about 259,104 requests across three providers and says A5's per-provider figure of 106,436 differs "because of rounding"; 3 × 106,436 = 319,308, which is 60,204 more, not rounding [V arithmetic]. Which count drives the $93–140 budget is not stated. | MEDIUM |
| C6 | Power-simulation code reproducibility | **Open.** `STATISTICAL_ANALYSIS_PLAN` says `power_sim.py`, `power_sim2.py`, `breakeven.py` live in A4's notes folder, outside this repository [V, stated by the document; location not checked]. All MDG and Type I numbers are unreproducible from the repo until copied into `analysis/`. | HIGH (reproducibility gate) |
| C7 | Tuning noise at N=200 | New. Validation size min(1,000, N) (L-05) means 200 validation items at N=200; a binomial SE of about 3 points at accuracy 0.8 [V arithmetic: sqrt(.8*.2/200)=0.028] is comparable to the plausible spread between the four learning rates. Selection of the lr, the ICL k and the best API on replicate 0 can therefore be near random at N=200, and "best-validation" arms are then partly noise. Not fatal (the freeze is applied before replicates 1..R, so there is no leakage), but it lowers the true power of the N=200 contrast beyond the MDGs and should be stated. | MEDIUM |
| C8 | Selection on replicate 0 only | New. The frozen configuration is chosen on one pool; replicates 1..R then measure the frozen choice. This is clean, but the arm "best of four" is an upward-biased optimum on replicate 0 only, not on 1..R. State that the estimand is "frozen tuned configuration", not "re-tuned per pool". [I] | LOW |
| C9 | Gate G2/G3 (FT ≥ kNN − 5 points) | The 5-point tolerance is large relative to the 3–5 point MDGs; passing G2/G3 shows only that fine-tuning is not clearly broken, not that it is competitive. Wording in the protocol already calls these sanity gates; keep that wording and never cite a pilot pass as evidence for H1. [I] | LOW |
| C10 | Evidence labels | Deliverables carry [V]/[I]/[R]/[U]; source URLs, dates and quotes live only in the agent files, not in the deliverables. Acceptable for assembly, but a deliverable cannot be audited without the agent files. | LOW |

## 2. L-14 (tuning bar): resolution proposed

Spiegelhalter et al. tune LoRA over 27 configurations × 5 folds [V, public README only]. Four trials per system (L-05) is defensible as *equal effort*, but a reviewer will say fine-tuning is under-tuned. Because N=200 and N=1,000 fine-tuning runs take seconds to minutes (pilot: N=1,000 LoRA 18–30 s, [V `results/pilot-2026-11-21`]), the cost of widening is small there. Proposal, logged as L-17: keep the 4-trial primary protocol; add a pre-registered sensitivity analysis at N=200 and N=1,000 (BANKING77 only) with a wider fine-tuning grid (lr × rank × epochs, up to 27 configurations) **and** a matched widening of the ICL side (k, retriever, prompt wording variants), reported beside the primary result; any conclusion that flips under the wider grid is reported as such. No change at N=3,000 or BillSum (cost).

## 3. Not checked

I did not re-verify any external source, price, licence or citation in the agent files, did not open `agents/01..05` beyond the sections cited above, and did not run code. Those statements keep the confidence the agent files gave them.
