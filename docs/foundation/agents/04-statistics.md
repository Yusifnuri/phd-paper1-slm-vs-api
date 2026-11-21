# A4: Statistical analysis plan (draft of STATISTICAL_ANALYSIS_PLAN.md)

Written 2026-11-21, before any confirmatory result. Labels: **[V]** verified from a source I read, **[I]** interpretation, **[R]** recommendation, **[U]** unverified. Source access date for every URL below: 2026-11-21. Code and raw output: `…/side-session-notes/local_552bee1b…/a4-statistics/` (`power_sim.py`, `pilot_an.py`, `pilot_f1.py`, `sim_power.json`, `sim_typeI.json`). Nothing was committed or run on a GPU or paid API.

## 1. Replicates, unit of analysis, power

### 1.1 What the pilot says [V, own computation, `pilot_an.py`]
All 45 pilot files score the same 1,000 validation items from one fixed pool (`data_seed` 42). Seeds therefore vary training order only.
- Within-model seed disagreement (pairwise share of items where two seeds differ in correctness): **≈10%** at N=1000 (acc ≈0.75), **3.9–5.7%** at N=5000 (acc ≈0.90). 14.5–15.5% of items are not unanimous across 3 seeds at N=1000. Seed SD of accuracy at N=1000: 0.4–2.0 points.
- Between-model discordance (LoRA, seed-matched): **13.6–18.6%** at N=1000, **5.4–6.7%** at N=5000.
- **Not observable:** fine-tuned-vs-API discordance (no API predictions exist), and pool-to-pool variance (pool fixed). Both are the quantities that decide power. [I]

### 1.2 Estimand and unit [R]
Estimand: δ = E[acc_FT − acc_ICL] over the population of labelled pools of size N (pool seed), training seeds and test items, i.e. "what gap would a practitioner holding one random pool of N labels expect". Both arms are scored on the same test items and the same pool draws. The correct analysis is a paired, **crossed items × replicates** model of d_ir = c^FT_ir − c^ICL_ir. Primary implementation: a linear (identity-link) mixed model, or equivalently the closed-form variance of the mean difference, Var = σ²_item/n + σ²_rep/R + σ²_res/(nR), by ANOVA components with Satterthwaite df (code in `power_sim.py::tests`, analysis A3). Sensitivity analysis: logistic GLMM with random item and replicate effects (and a random item-by-arm term), same contrast. Replicate = (pool seed, training seed) with a fresh nested pool per replicate; ICL arms use the same pool as the FT arm of that replicate.

**Why majority-vote McNemar over seeds is unsound [I, supported by `verification/03` §3, `05` S3]:** the majority of 3–5 seeds is a different system (an ensemble nobody deploys) with fewer errors; it discards the replicate variance that the claim needs.
**Why a bootstrap over 3 seeds is unsound:** `stats.py` resamples seeds from 3 with replacement, separately per system. A 3-cluster bootstrap has at most 10 distinct resamples, no usable tails, and the seed-unpairing wastes the pairing. Item bootstrap alone ignores replicate variance (type I error 8–22% below). Both are shown wrong by simulation.

### 1.3 Simulation (assumptions explicit)
Latent-probit model, `power_sim.py`: Z = μ_arm + √ρ_b·b_i + √(ρ_w−ρ_b)·g_{arm,i} + τ·w_{arm,r} + √(1−ρ_w−τ²)·e_{arm,i,r}; correct iff Z>0. ρ_b is calibrated so that between-arm discordance equals D_b, ρ_w so that within-arm replicate discordance equals D_w (exact bivariate-normal formula). Assumptions:
- A1 *pilot* regime = pilot levels: N=1000: acc_ICL 0.75, D_b 0.16, D_w 0.10, replicate SD of accuracy τ 1.0 pt; N=5000: acc 0.90, D_b 0.06, D_w 0.045, τ 0.5 pt.
- A2 *pessimistic* regime: D_b and D_w ×1.5, τ ×2 (fresh pools add variance the pilot lacks; FT-vs-API discordance likely exceeds model-vs-model).
- A3 pool effects correlate 0.5 across arms; A4 probit shape; A5 items are an i.i.d. sample from the test population; A6 test sizes 1,000 and 3,080 (BANKING77 test); R ∈ {3,5,8,12,20}; 1,000 simulations per cell (MC SE ≤1.6 pts); effects δ = 1, 2, 3, 5 points.
- Per-contrast α is the *first* Holm step: 0.025 (2 contrasts) and 0.0083 (6 contrasts: 3 tasks × 2). Power for later steps is higher, so this is conservative.

**Type I error (δ=0, n=1500, 1,500 sims, nominal 5%):**

| Analysis | R=3 | R=5 | R=10 |
|---|---|---|---|
| A1 pooled McNemar over n·R pairs | 18–35% | 23–36% | 31–45% |
| A2 item-mean paired t (ignores replicates) | 10–22% | 9–16% | 8–13% |
| **A3 crossed components, Satterthwaite** | 5.5–8.1% | **5.1–6.8%** | 5.5–6.2% |
| A4 replicate-level t, df=R−1 (conditions on the test set) | 6–8% | 9–14% | 14–27% |

A3 is close to nominal and slightly liberal at R=3 (truncated variance components, approximate df); use R≥5. A4 is valid only if the test set is treated as fixed.

**Power of A3, first-step α 0.025 / 0.0083 (selected cells; full grid in `sim_power.json`):**

| Regime, n_test | δ | R=5 | R=8 | R=12 |
|---|---|---|---|---|
| N=1000 pilot, 3080 | 2 pts | .69/.53 | .85/.74 | .93/.85 |
| N=1000 pilot, 3080 | 3 pts | .96/.89 | 1.0/.99 | 1.0/.99 |
| N=1000 pilot, 1000 | 3 pts | .75/.60 | .84/.73 | .89/.78 |
| N=1000 pessimistic, 3080 | 3 pts | .58/.41 | .80/.63 | .93/.84 |
| N=1000 pessimistic, 3080 | 5 pts | .92/.80 | 1.0/.98 | 1.0/1.0 |
| N=1000 pessimistic, 1000 | 5 pts | .85/.72 | .96/.90 | .98/.94 |
| N=5000 pilot, 3080 | 1 pt | .67/.51 | .80/.67 | .90/.83 |
| N=5000 pilot, 1000 | 2 pts | .90/.82 | .98/.94 | .98/.95 |
| N=5000 pessimistic, 3080 | 2 pts | .80/.64 | .95/.87 | 1.0/.97 |
| N=5000 pessimistic, 1000 | 2 pts | .66/.51 | .79/.66 | .88/.77 |

A 1-point difference at N=1000 is undetectable at any R≤20 (≤0.41/0.25 even with 3,080 items). Beyond ≈3,000 items, power is limited by R, not by n.

### 1.4 Are 5 replicates enough? [I from the simulation]
- **Yes, for large gaps:** ≥3 points at N=1000 (pilot-like variance, 3,080 items) and ≥2 points at N=5000. Fine for "FT clearly better/worse than ICL".
- **No, for the claims a crossover paper wants near the crossover**, where the gap is small by definition: ≤2 points at N=1000, any gap under pessimistic variance, and a 6-contrast family. R=8 reaches ≥80% at 3 pts (pessimistic, first-step 0.025) and R=12 at 2 pts at N=5000 (pessimistic).
- **Recommendation [R]:** R=8 for BANKING77 (small training runs, API calls cheap); R=5 minimum for BillSum and LEDGAR, stated as such with the detectable gap (≈3 points). Pre-register blinded re-estimation: after R=5, estimate only the variance components (not the effect) and extend R up to 10 if the planned power is below 80% for δ=3 points. Do not describe a null at R=5 as "no difference"; report the CI and the minimal detectable difference.

Continuous metrics (BillSum): same crossed model on per-item scores; with 3,269 bills the replicate term decides. Analytic [I, unsimulated]: detectable gap ≈ t(α/2)·SD_rep/√R; R=5, α=0.0083 (t₄≈4.6), SD 0.4 ROUGE-L points ⇒ ≈0.8 points. SD unknown until replicates exist.

## 2. Metrics

**Classification [R].** Accuracy = correct / N_test over all items, an unparseable output (parser returns None after the shared case-insensitive normaliser, unit-tested to map every gold label to itself, I-03) counts as wrong, never dropped. Report separately: unparseable rate; invalid-label rate (parseable but outside the label set), and **accuracy among parseable outputs** as a diagnostic only. Macro-F1: unweighted mean of per-class F1 over the *full* fixed label set, with outputs outside the set counting as misses for the gold class (this is the harness definition; it differs from sklearn's default union of labels). Micro-F1 equals accuracy for single-label tasks and is not reported; report it only for NER, which is out of scope. Per-class results are descriptive only: BANKING77 test has ≈40 items per class (3,080/77), so a per-class accuracy of 0.75 has a Wilson 95% interval of roughly ±13 points; LEDGAR rare classes are worse (I-12). Calibration: only for arms that supply confidences (token log-probs; verbalised confidences are a separate, labelled, weaker signal). Where available: ECE with 15 equal-width bins (Guo et al. 2017, arXiv 1706.04599) and an adaptive-bin variant plus Brier score, because binned ECE is bin-sensitive (Nixon et al. 2019, arXiv 1904.01685; [V] only from `verification/03`, not re-read). No confidence ⇒ no calibration claim; API logprob availability is [U] (`00-summary`).

**Intervals [R].** Primary CI = the crossed-model interval for δ (t, Satterthwaite df). For descriptive per-arm accuracy: two-stage percentile bootstrap, resample replicates then items within replicate (or, with R<8, report the replicate range plus an item-only bootstrap and say so), 10,000 resamples, seed fixed. Never a bootstrap over 3 seeds. Paired comparisons: always on identical items and pools.

**Multiplicity [R].** One confirmatory family: per task, best-validation fine-tuned configuration vs best-validation in-context configuration of the best API, at N=1000 and N=5000. 3 tasks ⇒ 6 hypotheses (4 if LEDGAR is dropped); Holm at α=0.05. "Best" is chosen on the validation split only, before the test read. The N=50/200 points, technique ablation, model comparisons, per-class results, H3 and cost are exploratory: estimates with CIs, no p-values, and any p-value shown is labelled unadjusted. Two-sided tests; "better" and "worse" are both reported.

**Equivalence [R].** No TOST in the confirmatory family. An equivalence claim needs a deployment-justified margin. With pilot seed SD 0.4–2.0 points, a technique difference over 5 replicates has SE ≈0.6–1.2 points and the 90% interval (t₄=2.13) is ±1.3–2.6 points before item noise: ±1 is unreachable, ±3 is borderline. Report the interval; claim equivalence only if it falls inside a margin fixed before the data (suggest ±3 points, to be justified by cost-per-correct impact).

## 3. Learning curves and break-even

**Crossover N [R/I].** The design has 3 N levels per task (200, 1000, 5000; 50 for ICL/BANKING77 only), a factor of 5 apart. Linear interpolation of Δ(N) in log N between two points cannot be validated and the pilot curve is strongly nonlinear (accuracy 0.17 → 0.75 → 0.90 for Qwen). So:
1. *Primary statement:* the bracket. "ICL significantly better at N_j, FT not worse at N_{j+1}", from the confirmatory family and equivalent CIs at each N. This is what the data support.
2. *Secondary estimate:* N* = root of piecewise-linear Δ(log N). Uncertainty by Monte Carlo draws of (Δ(N_1), Δ(N_2), Δ(N_3)) from a multivariate-t whose covariance comes from the crossed model with pools nested and shared across N; report N* percentile interval and P(no sign change in range). Interpret as a conditional-on-interpolation estimate.
3. *Parametric fit* (a − b·N^−c) only as sensitivity: with 3–4 points it is saturated (no residual df, no uncertainty).
4. *Extension:* pre-register N=500 and N=2000 for any task whose bracket spans a factor of 5 (exploratory). N=50 is in no confirmatory contrast.

**Costs (all components shown) [R].**
- API per query: Σ billed input tokens (uncached) × p_in + cached input × p_cache + output tokens (including billed reasoning tokens) × p_out, × batch factor if used, for the whole prompt including k retrieved shots; retries billed (each failed/unparseable call counts in cost; the item is wrong if the final answer is unparseable); price date recorded.
- Self-hosted FT: training cost C_train = (all tuning trials + final training) GPU-hours × rate; inference per query = measured GPU-seconds/query × rate ÷ utilisation u; plus retries. Rate r is a **band** ($2.3–5.4/h from `05` S7 [U, A5's domain]); report results at low/mid/high r and for u ∈ {1, 0.5, 0.2}. Not sampled in the bootstrap; shown as sensitivity rows.
- **Cost per correct prediction** = (C_train + Σ_q c_q) / Σ_q correct_q at a stated deployment volume Q (training amortised over Q), reported for Q ∈ {10³ … 10⁷}.
- **Cost per successful summary:** same numerator over successes, where success = parseable, within a length bound, and passes a pre-registered quality criterion. Because ROUGE has no natural pass mark, define success on the validated metric (§4): factual-consistency score above a threshold calibrated on the human sample to ≥90% precision for "faithful". Without that, report **cost per 1,000 outputs next to quality**, not a ratio. [R]
- **Break-even volume** (marginal-cost form): Q* = C_train / (c_API − c_FT), per query, valid only when c_API>c_FT; per-correct form: Q* = C_train / (a_FT·(c_API/a_API) − c_FT), a=accuracy. If the denominator's CI contains 0 or changes sign, Q* has no finite interval: use a Fieller-type interval or report P(denominator ≤0) and "no break-even within range". Never a percentile CI of an unbounded ratio.
- **CI for ratios:** resample items as pairs (cost_i, correct_i) for the ratio of means (harness `bootstrap_ratio_of_means`), and resample replicates for C_train and accuracy; the harness function cannot take a fixed cost, so add it after resampling (see A4-04). Ratio-of-means, not mean of ratios.

## 4. Summarisation evaluation

Findings from full text I downloaded and read (arXiv PDFs):
- ROUGE is not a quality measure on its own. Goyal et al. state ROUGE/BERTScore "unreliable when small improvements are reported" and that ≈5-point gaps are considered informative [V; arxiv.org/abs/2209.12356, §1; confidence high]. Bhandari et al.: ROUGE-2 "de-correlates with humans" when evaluating only top systems [V; arxiv.org/abs/2010.07100, summary of Exp-II]. SQuALITY (2205.11465) reports that common metrics do not reliably measure quality on model-generated long summaries [U: quote verified only from a search summary; the PDF's contribution list is truncated in my extract; confidence medium].
- BERTScore: use `microsoft/deberta-xlarge-mnli`, rescaled with baseline, report the hash; rescaling is "with respect to its empirical lower bound" (arXiv 1904.09675 §3, [V]); rescaling changes scale, not ranking. It is reference-based and does not check the source.
- Factual consistency: SummaC (Laban et al. 2022, arXiv 2111.09525) segments into sentence units and aggregates NLI scores; its benchmark datasets are built on news (Source: "C for CNN/DM, X for XSum") [V, Table 1 caption; high]. AlignScore (Zha et al. 2023, arXiv 2305.16739 §3.2) "breaks a context into coarse-grained chunks" of ≈350 tokens and the claim into sentences, designed for long text [V; high]. QAFactEval (Fabbri et al. 2022, 2112.08542) "outperforms the best-performing entailment-based metric" on SummaC [V; high]. **I found no validation of any of these on long legislative text** (all benchmarks I read are news/dialogue/general) [I]; MiniCheck/LLM-AggreFact (2404.10774) long-document coverage not verified [U].
- LLM judge: Zheng et al. (2306.05685) report "over 80% agreement" for GPT-4 on pairwise MT-bench/Arena votes and name "position, verbosity, and self-enhancement biases" [V; Abstract; high]; G-Eval flags "bias towards the LLM-generated texts" (2303.16634, Abstract). Chiang & Lee (2305.01937) validate LLM ratings on stories, not summaries. No source validates a graded judge on legislative summaries.
- Human evaluation: the LongEval survey of 162 summarisation papers: most "annotating less than 50 summaries" and "only 12 report inter-annotator agreement" among those with multiple annotators [V; 2301.13298 §2; high]. RoSE: "achieving a high inter-annotator agreement among annotators can be difficult" (2212.07981, §1) [V]. BooookScore estimates human evaluation at "$15K USD and 500 hours" for book-length summaries (2310.00785 Abstract) [V]. I could not retrieve ACL/ARR checklist wording on summarisation or on human evaluation; the checklist section D concerns human-subject details [U, search result only]. **Whether venues *require* a human evaluation is unverified**; the evidence supports "reviewers will doubt ROUGE-only claims", which is an interpretation.

**Recommended metric set [R].** (i) ROUGE-1/2/L F1 (`rouge-score`, stemming, rougeLsum for multi-sentence; version pinned) and BERTScore F1 (rescaled), primary for the *pre-registered* contrast but worded "on ROUGE-L"; (ii) length covariates (output words, compression ratio) reported with every score, since ROUGE F1 and judge scores confound with length; (iii) AlignScore as automatic factual consistency with source chunking, SummaC-ZS secondary; (iv) a **human evaluation** on a sample (below); (v) LLM judge exploratory only, with a pre-stated drop rule: Krippendorff α (ordinal) with the human consensus < 0.6 ⇒ not reported (carried over from the pre-registration; α thresholds are conventions [U]).
**Is human evaluation necessary?** For a *credible* claim that the small model's summaries are as good as an API's, yes: ROUGE does not measure faithfulness, and no factual metric is validated here. [I] A human check also validates AlignScore/judge on BillSum.
**Protocol [R]:** 100 test bills (stratified by input length tercile; fixed seed), 4 outputs each (best FT and best API arm at N=1000 and N=5000) = 400 summaries, shown blind and shuffled. Two annotators for all 400 plus a third on disagreements; annotators are law or policy trained. Items: faithfulness at claim level (LongEval-style FINE, supported / unsupported per summary sentence), and a 5-point overall acceptability. Agreement: Krippendorff α (ordinal for the 5-point, nominal for the binary), bootstrap CI; adjudicate by majority. Sample-size logic: this size is a **validation and descriptive** sample, not a powered test: a paired proportion with ≈30% discordance has a CI of ≈±11 points at n=100, so it detects only large differences; with 400 ratings the CI of a metric–human Spearman ρ≈0.5 is ≈±0.08 (own approximation). Cost [assumption]: LongEval paid $15–20/h; at ≈8 min per summary for 2 annotators ≈107 h ≈ $1.6–2.1k plus adjudication ≈$2.5k. Annotators are a human decision.

## 5. What the design can and cannot support

**CAN:** (a) a paired δ with valid CI at N=1000/5000 per task (gaps ≥3 points at N=1000, ≥2 at N=5000 for R≥5); (b) a crossover bracket within the grid plus a conditional interpolated estimate; (c) cost per correct/1,000 outputs and break-even Q as functions of a price date and a GPU-rate band; (d) a descriptive ROUGE/BERTScore/faithfulness picture with a human-validated subsample.
**CANNOT:** (a) 1–2-point differences at N=1000; (b) equivalence at ±1 point; (c) a point estimate of N* with a CI sharper than the grid spacing; (d) any claim that "FT ≥ API" for summarisation quality on ROUGE alone; (e) a claim about the *population* of small models or APIs: two models, three APIs, one GPU type; (f) per-class conclusions; (g) pooled or confirmatory use of the pilot (validation-split, fixed pool, under-trained at small N, no API arm).

## 6. New issues (A4-)

| ID | Issue | Sev | Proposed fix |
|---|---|---|---|
| A4-01 | Power is replicate-limited: at R=5, gaps ≤2 points (N=1000) or any gap under pessimistic variance fall below 80% power | HIGH | R=8 for BANKING77; blinded variance re-estimation; report minimal detectable gap |
| A4-02 | FT-vs-API discordance D_b and pool variance unobserved; simulation rests on model-vs-model proxies | HIGH | Run API arms on the validation split with the first 5 replicate pools; rerun `power_sim.py` before freezing R |
| A4-03 | Harness `hierarchical_paired_bootstrap` resamples seeds unpaired; with R<8 it is unsound | HIGH | Replace by the crossed-model analysis; bootstrap only as sensitivity with R≥8 |
| A4-04 | `bootstrap_ratio_of_means` cannot hold a fixed training cost or a deployment volume Q | MEDIUM | Add a function for (C_train + Q·c) / (Q·acc) with replicate resampling; Fieller/sign check for break-even |
| A4-05 | No validated factual-consistency metric for long legislative text (all validations I read are news/dialogue) | HIGH | Validate AlignScore/SummaC against the human sample before using them; otherwise report as exploratory |
| A4-06 | Human summarisation evaluation (400 ratings, ≈$2–5k) is a budget and annotator decision | MEDIUM | Human decision; fall back to 60 bills × 2 systems and claim only metric validation |
| A4-08 | "Best arm" selection must use validation only; selecting on test inflates the contrast | HIGH | Freeze arm selection on validation; log; the test read is one per final configuration (I-08) |
| A4-09 | Cost-per-success for summarisation needs a success definition; none has a pass mark | MEDIUM | Calibrate on the human sample; otherwise cost per 1,000 outputs plus quality |

## 7. Assessment of register issues

- **I-04 (replicates):** keep HIGH, partly resolved on paper. 3 seeds and a fixed pool understate uncertainty (pilot confirms seeds only reorder items). R≥5 with fresh pools is necessary; R=5 is enough only for gaps ≥3/2 points; recommend R=8 on BANKING77. FIXED-UNVERIFIED until the nested-pool code runs.
- **I-05 (inference):** plan now defined (§1.2, tests verified by simulation: A1/A2 anticonservative, A3 near nominal). IN-REVIEW until `stats.py` implements it (A4-03).
- **I-06 (rsLoRA/TOST):** agree. The ±1 margin is unreachable (§2); the pilot rsLoRA gain (+4–6 points) is confounded with lr. Estimation only, lr-tuned per technique; ±3 only with a stated justification.
- **I-07 (multiplicity):** resolved in design: one family of 4–6 tasks × {N=1000, N=5000} contrasts, Holm, everything else exploratory; wording to be copied into the pre-registration (A4-10).
- **I-13 (summarisation validity):** keep HIGH, OPEN→IN-REVIEW with the metric set and human protocol above; ROUGE-only claims are not defensible; factual metrics are unvalidated for this text; human evaluation is recommended and is the one genuinely human-dependent decision (annotators, ≈$2–5k).
