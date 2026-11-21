# Verification round, consolidated (2026-11-21)

Five independent verifier sessions (reports `01`–`05` in this folder) re-checked the datasets, literature, API/serving/GPU facts and the design itself. The verifiers are AI agents; they opened primary sources but can still err. Items a human must check are listed at the end. This file supersedes `docs/research/2026-11-21-verified-findings.md` wherever they disagree.

## Errata to earlier claims (including ones made by the coordinating agent)

| Earlier claim | Correct |
|---|---|
| "The pilot parser is fine" (coordinator, after the pilot) | **Wrong.** `evaluate.py` lower-cases the candidate but tests it against a case-sensitive label set, so the one mixed-case BANKING77 label `Refund_not_showing_up` can never be scored correct. Re-scoring the 45 saved prediction files (coordinator, 2026-11-21): accuracy +0.0052 on average (max +0.009); unparseable outputs at N=5000 fall from 11–18 to 2–8 per 1,000. The API parser handles it, so the arms were scored differently |
| "≈9 optimizer steps at N=50" | 12 / 39 / 189 / 939 steps at N = 50 / 200 / 1000 / 5000 |
| "OLMo-3 DoRA likely diverged" | `first_train_loss` is NaN, so it failed at step 1 |
| "Nobody has published break-even curves by label budget" | Refuted for classification: Pecher et al., EMNLP 2025, arXiv 2402.12819 |
| Spider repo licence CC BY-SA | HF card and Yale page say CC BY-SA 4.0, but the GitHub repo LICENSE is Apache-2.0 (code); the databases' per-source terms are unknown. The Spider test set has been public since 2024 |
| API grid cost ≈ $51 standard / $26 batch | $36 / $18 excluding thinking tokens (re-derived by the API verifier); both fit the budget. Prompt caching does not help per-item retrieved shots |
| "Gemini 3.x returns no logprobs"; "Claude has no logprobs" | Not supported by the docs the verifier read: UNVERIFIED either way, needs one live call per API |
| "Vast.ai median H200 $4.44" | Unverifiable; also, every quoted H200 price is SXM/HGX while the university card is an H200 NVL (600 W vs 700 W) |
| DoRA overhead "unknown" | Pilot: +31% training time (Phi), +42% (Qwen), +19% peak memory. vLLM 0.13.0 rejects DoRA in source; the fix for silently mis-served PiSSA/OLoRA/LoftQ is not in 0.13.0 |
| "Dev/test: BillSum training will be 20+ h" | Probably an over-estimate (tokens/s rises with sequence length) but memory risk is real (fp32 logits at 2.6k tokens). Measure before planning |

## Blocking design problems (from the red-team review, checked against the pilot where possible)

1. **The two arms see different information.** The fine-tuned prompt never lists the 77 labels; the API prompt does. "Equal label budget" must be defined as "a labelled pool of N available to both arms", with the same task text, label list and decoding in both.
2. **Under-training at small N** makes the small-N half of the curve meaningless (12 steps at N=50; 92–95% of outputs unparseable). Needs a minimum-steps / per-N validation-tuned protocol, with equal tuning effort across arms.
3. **Parser asymmetry** (above): one shared normaliser plus a test that every gold label parses to itself.
4. **Seeds and statistics.** The pool is fixed (`data_seed` 42), so "seeds" vary only training order; 3 seeds cannot support a seed-level bootstrap; a seed-majority prediction is a different system. Use ≥5 replicates (pool seed × train seed), item as the unit, paired tests per replicate or a mixed model. McNemar needs ≈1,740 items at a 20% discordant rate (not 870); the ±1-point TOST margin is unreachable with the observed seed SD.
5. **rsLoRA vs LoRA is confounded by learning rate** (scale α/√r = 8 vs α/r = 2 at the same lr); tune lr per technique or drop equivalence claims.
6. **Anomalies to explain before trusting any pilot number:** Phi `first_train_loss` is 18.88 for LoRA/DoRA/rsLoRA but 8.5 for QLoRA (same base weights, loss above ln(vocab)); check label masking, padding and chat-template handling.

## Datasets: status

| Dataset | Verified | Role |
|---|---|---|
| BANKING77 | 10,003 / 3,080, CC BY 4.0 (card and repo agree) | Primary (public since 2020; treat as a public benchmark, contamination unknown) |
| CLINC150 plus | 15,250 / 3,100 / 5,500, CC BY 3.0 (repo LICENSE; card prose empty) | Primary or fallback second classification set |
| BillSum | 18,949 / 3,269; CC0 only in card metadata; no original licence file; US-government text | Primary summarisation (long inputs, cost) |
| LEDGAR (LexGLUE) | 60k/10k/10k, 100 labels, chronological; card YAML cc-by-4.0 but prose empty; original LREC release states **no data licence**; SEC filings public but exhibits are third-party text | Candidate, licence gap to clear by a human; single-label like BANKING77, so little task diversity |
| Spider / BIRD | see errata; no hidden, locally scoreable test for either; database execution infrastructure needed | Secondary only |
| SEntFiN | 10,753 headlines, 14.4k entity-level items, repo MIT; headline rights (Economic Times) unaddressed; likely ceiling effect | Secondary unless rights are cleared |
| Twitter Financial News | 9,543 / 2,388 (no test split; card text inconsistent), 65% neutral; tweet text under X terms | Secondary / exclude |
| Few-NERD | 131,767 / 18,824 / 37,648, CC BY-SA 4.0 (share-alike; adapter status is a legal question) | Secondary NER |
| Broad Twitter Corpus | 5,342 / 2,002 / 2,002; ships raw tweet JSON; X terms apply | Exclude from the commercial claim |
| CoNLL-2003 | text is Reuters RCV1; commercial use not cleared | Exclude |
| MBPP+ / HumanEval+ | test sizes 378 / 164 (underpowered); 108 of 378 MBPP+ ids are in MBPP train, 224 in test (reproduced); OLMo-3 has semantic duplicates of all MBPP | Exclude |
| Self-OSS-Instruct | 50,661, ODC-BY, no test split | Training source only |

X's developer terms (read 2026-11-21): "X prohibits any use of the X APIs and/or X Content to fine-tune or train a foundation or frontier model with the exception of Grok"; only Post IDs may be redistributed. Whether a 4B LoRA adapter is a "foundation or frontier model" is a legal question.

Contamination: no documented overlap of the planned datasets with the models' pretraining data was found, which is absence of evidence only; infini-gram probes covered OLMo-2-era indexes, not Dolma 3, and neither Phi nor Qwen.

## Literature and novelty

- All 22 cited arXiv ids resolve. Corrections: LoRA Land has no label-budget axis; the low-code paper's venue is unconfirmed; "OLMo forgets early data" is really *How Much Can We Forget about Data Contamination?* (Bordt et al., ICML 2025).
- **Closest prior work (must cite):** Pecher et al., EMNLP 2025 (2402.12819); Mosbach et al. 2023 (2305.16938); Bertsch et al., NAACL 2025 (2405.00200); Mansuri et al. 2026 (2609.40236, cost break-even, one task); plus 2404.11018, 2405.00732, 2405.05904, 2408.06663, 2602.12413, 2410.03249, 2602.04998, 2602.06370, 2610.24260, 2311.16863. Scoop risk: Pecher-style extensions to NER/APIs/cost; Olmo-3 contamination groups adding fine-tuning. Re-run the search before submission.
- **Novelty holds only for the conjunction:** equal-pool retrieval ICL on cheap commercial APIs vs PEFT small models, several task families, cost per correct answer with break-even volume, and (exploratory) per-item pretraining overlap. "To our knowledge, as of Nov 2026."
- Per-item pretraining-overlap analysis (H3) is the least contested contribution but the most fragile statistically (post-treatment strata, regression to the mean): keep it exploratory with a placebo stratification.
- Statistics: McNemar sample size confirmed as a normal approximation (870 / 1,476; exact ≈920 / ≈1,520); at a 20% discordant rate ≈1,740. Majority-vote McNemar over seeds and a bootstrap over 3 seeds are unsound. Judge agreement ">80%" is GPT-4 pairwise preference, not graded scoring: validate the judge on our own labels (Krippendorff α).

## Resolved conflicts between reports

- **Code replacement.** Dataset verifier: LEDGAR (licence gap). Red team: drop code and use a clean-licence classification set. Resolution: do not use Spider/BIRD; code generation is dropped for methodological reasons (underpowered tests, MBPP leakage, OLMo-3 contamination), not because the APIs win. Second classification set: CLINC150 if a clean licence matters most, LEDGAR if a legal-domain task is worth a human licence check.
- **Scope.** Red team recommends 3 tasks (BANKING77, BillSum, one more classification set), 2 models (Qwen3-4B, Phi-4-mini; OLMo-3 after the DoRA failure is understood; drop Gemma), LoRA as the main arm with QLoRA/DoRA/rsLoRA as an N=1000 BANKING77 ablation, 5 replicates, and 2 contrasts per task as the only confirmatory family. NER and finance fall out because of licences (CoNLL, BTC, Few-NERD share-alike, SEntFiN rights, tweets) and thin per-cell results in 2 months. **This scope cut is the user's decision.**
- **H3 as a lead contribution.** Literature verifier calls it the least contested contribution; red team says do not lead with it. Resolution: exploratory secondary contribution with a placebo; lead with the equal-pool, cost-normalised crossover.
- **GPU-hour estimate.** Pilot tokens/s rose with sequence length; do not trust the 20+ h BillSum guess in either direction until measured at 2.6k tokens (watch memory).

## Fixes before any confirmatory run (ranked)

1. Same task text, label list and decoding in both arms; add base-SLM zero/k-shot, kNN-vote and API zero-shot/random-k baselines.
2. Shared case-insensitive label normaliser, unit-tested on all gold labels; re-score saved predictions (done for the pilot above: +0.5 points average).
3. Minimum-steps plus per-N epoch/lr selection on validation, equal tuning effort across arms.
4. Fresh pool per replicate, shared by both arms; ≥5 replicates.
5. Item-level paired tests or a mixed model; one small pre-registered confirmatory family; everything else exploratory.
6. lr tuned per technique; widen or drop the equivalence margin.
7. Explain Phi's first-step loss of 18.88 and OLMo DoRA's step-1 NaN.
8. API protocol: dated ids where they exist (none do for Haiku 5.5 and Luna), temperature 0 where supported (Claude 4.7+ removed temperature; Haiku 5.5 unconfirmed), reasoning/thinking off, billed token counts, price date, a 10% repeat for self-agreement, split batches below the 10M/20M enqueued-token limits.
9. Pin CodeCarbon's country, use NVML energy directly for the GPU, state that CPU/RAM are modelled; report H200 NVL against an SXM price band.
10. Pre-registration (`docs/PREREGISTRATION.md`) before any test read.

## A human must check personally

Licence pages of BANKING77, BillSum and every dataset finally chosen; LEDGAR's release terms (SWITCHdrive) and EDGAR reuse; SEntFiN's headline rights; X's terms and whether a LoRA is a "foundation or frontier model"; whether CC BY-SA reaches adapters; Reuters/RCV1 for CoNLL; one live logprobs/temperature call per API; prices on the day of each run.
