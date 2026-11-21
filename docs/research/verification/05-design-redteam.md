# Design red-team, 2026-11-21

Sceptical-reviewer review. Numbers are my own recomputations from `results/pilot-2026-11-21/` (curve.jsonl, pilot.jsonl, predictions/). Verdict: not publishable as designed; rescuable by cutting scope.

## Blocking

**B1. The arms see different information, and "equal label budget" is undefined.**
- The fine-tuned prompt (`data.py:29-30,75-76`) gives only an instruction and the query: the 77 label names are never shown. The API prompt (`prompts.py`, CLASSIFICATION) lists every label and parses case-insensitively.
- Scenario: at N=50 about 40 classes have no training example, so the SLM cannot emit them, while the API can pick them from the list. "FT loses at small N" then measures prompt design. At N=5000 the reverse: the API sees 32 of 5000 examples per query, FT absorbs all.
- Decoding also differs (greedy free text with a 32-token cap versus provider defaults).
- Fix: define the budget as "a labelled pool of N available to both arms". Use the same task text and label list in both prompts and in FT training. Fix temperature 0 for all. Report zero-shot and random-k for the API, base-SLM zero-shot and k-shot (this is also the H3 baseline), and a kNN-vote baseline (label vote over the retrieved shots, no LLM). If kNN is near the API, the "API arm" is mostly retrieval. Without this a reviewer can say the API was handicapped at small N and favoured at large N.

**B2. Under-training makes the small-N end of the curve meaningless (H1).**
- Steps are 3 × ceil(N/16): **12 / 39 / 189 / 939** for N = 50 / 200 / 1000 / 5000 (the handoff's "≈9" is a little low). Unparseable outputs: 916–953 of 1000 at N=50 and 734–886 at N=200 (all models, 3 seeds); 65–120 at N=1000. Final train loss at N=50 is 1.9–2.8, i.e. not converged.
- Scenario: a "crossover at N=1000" would be reported when the cosine schedule simply had no steps below that. The curve's shape reflects steps, not examples.
- Fix: protocol below.

## Serious

**S1. A parser bug inflates the "invented labels" and breaks symmetry.**
- `evaluate.py:25-37` lower-cases the candidate, then tests membership in the case-sensitive label set. BANKING77 has one mixed-case label, `Refund_not_showing_up`, so **that class can never be scored correct**. Of the unparseable outputs at N=5000 (11–18 per run), **9–10 are this class** (Qwen 10 of 13–15, i.e. most of them). At N=1000 it is 8–11 of 41–120. The model's output was `Refund_not_showing_up` verbatim.
- The API parser (`prompts.py`, `_norm_label` into a lookup) would score it, so the arms are scored differently. The handoff's "the parser is fine" is wrong.
- Fix: one shared normaliser (map both sides through the lookup), a unit test that every gold label parses to itself, and re-score the saved predictions (no rerun needed). Recompute the 4–12% figure afterwards; at N=5000 true invented labels are only about 0.2–0.8%.

**S2. Seeds change training order only; the pool is fixed, so uncertainty is understated.**
- `data_seed` = 42 for every run, so all three "seeds" share one nested pool. The seed SDs (0.002–0.020) exclude pool-sampling variance, which dominates at small N. The API arm is deterministic (`seed=None` for retrieved shots): one pool, one run, no variance.
- Scenario: "A beats B, p<0.01" holds for one pool draw only.
- Fix: replicate = (pool seed × train seed), a fresh nested pool per replicate, the same pools for both arms. Report pool-to-pool SD.

**S3. The planned inference is unsound with 3 seeds.**
- `stats.py:138-190` resamples seeds with replacement from 3, separately per system (unpaired). A cluster bootstrap with 3 clusters is badly anticonservative.
- "Seed-majority prediction" is a different estimand (a 3-model ensemble nobody deploys), and seeds disagree on **14.5%** of items (Qwen, N=1000).
- Fix: unit = item. Primary test = McNemar on per-replicate predictions, or a mixed logistic model with random item and replicate effects. State the estimand (expected accuracy difference over replicates). Use ≥5 replicates.

**S4. The ±1-point TOST margin at N=1000 is unreachable.**
- Seed SD at N=1000 is 0.020 (Phi), 0.011 (Qwen), 0.004 (OLMo). A paired technique difference over 3 seeds has SE ≈ 1.2 points. A 90% CI with t₂ = 2.92 is about ±3.5 points, before item noise on the test set. It cannot fit inside ±1.
- The pilot's technique evidence is also confounded. rsLoRA at the same lr and α=32 scales the update by α/√r = 8 versus α/r = 2: a 4× difference. Its +4–6 points (0.79–0.81 vs 0.75) go with lower final train loss (0.02–0.03 vs 0.06–0.08), so it is a learning-rate effect, not a technique effect. QLoRA is 0.5–4.5 below LoRA. OLMo-3 DoRA is a NaN-loss failure and must be reported, not dropped.
- Fix: tune lr per technique on validation over the same grid (rsLoRA's grid shifted for √r), widen the margin to about ±3 points with a stated deployment justification, or report estimates only.

**S5. The multiplicity family is undefined.**
- Models × tasks × N × techniques × APIs is hundreds of comparisons. Holm "within task" still leaves 50+ per task, far beyond the power plan (n ≈ 1,480 for about 10 comparisons).
- Fix: one primary family: per task, best-validated FT versus best API at two N values (two contrasts, Holm over 2 × tasks). Everything else is exploratory with CIs and no p-values.

**S6. Test discipline and near-duplicates.**
- The pilot's validation split (from train) is good, but it is now used for model selection and claims. Pool-versus-test decontamination is planned but not implemented; BANKING77 and CLINC contain many paraphrases.
- Scenario: a retrieved near-duplicate carries the gold label, so ICL looks strong for reasons unrelated to capability (it also helps FT, hence report both).
- Fix: freeze and hash the test set; tune on validation only; one test read per final configuration, logged; report pool-to-test nearest-neighbour similarity and accuracy stratified by it.

**S7. API and cost threats are listed but not controlled.**
- Pin dated model ids; run a 10% repeat to measure self-agreement (silent updates, nondeterminism); GPT-6 Luna reasoning tokens bill as output; Claude's new tokenizer gives about 30% more tokens, so use provider-billed usage, never character counts; record the price date.
- Cost per correct answer must be all-in: training, eval, retries, idle time. The H200 rate is imputed (the university box is not billed per hour): report a band ($2.3–5.4/h) and break-even volume as a function of it. One hardware type means no generalisation. NVML readings on a shared container are upper-bound board power, not total energy.

## Minor

- **Generation versus classification.** Free generation is a legitimate deployment protocol and the only one APIs share (no logprobs on at least two of three). Keep it as primary, once S1 and B1 are fixed (the invented-label rate should largely vanish). Report constrained decoding or label scoring for the SLMs as an exploratory robustness check. It can change conclusions only at N ≤ 200.
- **Judge bias (summarisation).** Self-preference and length bias. ROUGE-L and BERTScore primary; judge on a sample only, with a pre-stated agreement threshold (Krippendorff α ≥ 0.6) below which it is dropped.
- **Licences.** Few-NERD (share-alike), Spider (CC BY-SA, unverified) and Twitter data may block publishing adapters. Publish adapters only for CC0 and CC BY tasks (BANKING77, BillSum); ask legal for the rest.

## Training protocol (fix for B2 and S4)

1. Train for `max(epochs·ceil(N/B), S_min)` steps with S_min ≈ 200 (smaller batch at small N so that S_min covers ≥ 10 epochs).
2. Per (model, N), choose epochs ∈ {3, 5, 10, 20, 40} by validation loss with early stopping (patience 2). Choose lr from the same 3–4 value grid for every technique.
3. Equal tuning effort across arms: same number of validation trials for every FT variant (4 lrs) and every ICL variant (k ∈ {4, 8, 16, 32}); report the budget in GPU-hours and dollars.
4. Everything else locked: optimiser, schedule, batch size, max length, r, dropout; α = 2r except rsLoRA as in S4.
5. Diagnose the OLMo DoRA NaN before any claim.

## Hypotheses

- **H1 crossover:** testable after B1, B2, S1. Operationalise as the smallest N where FT ≥ best ICL in the paired confirmatory family, with a CI over replicates.
- **H2 cost:** an estimate, not a test. State utilisation assumptions and show break-even against the rate band.
- **H3 capability redirection:** ambiguous. "Gold in the base model's top-k" depends on k, the prompt and base format failures, and is undefined for free generation. Strata defined by base accuracy are post-treatment, so regression to the mean produces "gains concentrated where the base was right". Needs a fixed independent base score, a pre-stated stratum definition and a placebo stratification (random split, or item length). Make it exploratory, and do not lead with it.
- **H4 contamination:** drop, or one exploratory paragraph. No post-cutoff data exist for these tasks and Qwen3's cutoff is undisclosed.
- **H5 equivalence:** estimation only unless the margin is widened and lr is tuned (S4).
- **H6 format compliance:** trivial at N ≥ 1000 and parser-dependent. Descriptive.

Pre-registration is realistic if short: 2–3 confirmatory hypotheses (H1 in one family, H2 as an estimate, H5 only with a widened margin). It must fix: datasets and split hashes; both prompts verbatim; label-list policy; decoding; training protocol and grids; primary metric and parser; the family and test; exclusion rules (failed runs are reported); the replicate design; the cost model and rate band; API ids and dates. Commit it before any test read. The pilot and curve are exploratory and must not be pooled into confirmatory results.

## Contribution

Prior work (Mosbach 2023, Bertsch 2025, LoRA Land) covers FT versus ICL. The honest novelty is current cheap APIs, equal-pool retrieval ICL and cost per correct answer with break-even volume: a solid empirical paper, not a conceptual one. A five-task, four-model, four-technique, three-API grid gives a thin result per cell in two months.

## (a) Ten fixes, ranked

1. Same task text and label list in both prompts and in FT training (B1).
2. Symmetric case-insensitive label normaliser, unit-tested on all gold labels; re-score all saved predictions (S1).
3. Min-steps plus per-N epoch/lr selection on validation, equal tuning effort across arms (B2).
4. Fresh pool per replicate, shared by both arms (S2).
5. Item-level paired test or mixed model, ≥5 replicates; drop the 3-seed bootstrap and the majority estimand (S3).
6. One small pre-registered confirmatory family; the rest exploratory (S5).
7. Tune lr per technique, or drop equivalence; widen the TOST margin to about ±3 (S4).
8. Add baselines: base-SLM zero-/k-shot, kNN vote, API zero-shot and random-k; report near-duplicate similarity (B1, S6).
9. Lock the API protocol: dated ids, temperature 0, 10% repeat, billed tokens, price date, all-in cost (S7).
10. Cut H4, make H3 exploratory with a placebo, make the judge secondary.

## (b) Recommended minimal design

- **Tasks (3):** BANKING77, BillSum (ROUGE-L and BERTScore primary), plus one clean-licence classification set (CLINC-OOS or LEDGAR). Drop NER, financial sentiment and code (licences, OLMo/MBPP contamination, thin tests).
- **Models:** Qwen3-4B-Instruct-2507 and Phi-4-mini-instruct; OLMo-3-7B only once the DoRA failure is understood; drop Gemma-4-E4B.
- **Techniques:** LoRA main arm. QLoRA, DoRA and rsLoRA (each lr-tuned) as a secondary ablation at N=1000 on BANKING77 only.
- **N:** 200, 1000, 5000 for FT; N=50 for ICL, or FT only with the min-step protocol.
- **Replicates:** 5 (pool seed × train seed), paired across arms.
- **APIs:** Haiku 5.5 and Luna (Gemini if the budget allows), with retrieved-k, random-k and zero-shot. Cost estimate before spend.
- **Protocol:** as above; the test split is read once per final configuration.

## (c) What I could not assess

- Real API behaviour (I read only the harness, no run) and BillSum training code (not written).
- Actual power at final test sizes (needs the discordant rate from API predictions).
- Real near-duplicate rates between pool and test (no embeddings computed).
- Dataset licences, serving, energy and the vLLM experiments (not built).
- Whether the OLMo-3 DoRA NaN is a bf16 problem or a bug (no log inspected).
