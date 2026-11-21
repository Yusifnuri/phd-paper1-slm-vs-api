# Verification 03: literature, novelty, statistics, metrics (2026-11-21)

Method: arXiv API metadata (ids below resolve), abstracts, the bert_score README and the Zheng et al. PDF text, plus web search. Items marked *(from memory)* are standard references I did not re-fetch; check before citing.

## 1. Citation check (sources.md, Literature)
All 22 arXiv ids resolve and the titles match the claims. Corrections:
- **LoRA Land (2405.00732)**: the abstract says "outperform... GPT-4 by 10 points on average" over 31 tasks and 10 base models, against zero-shot GPT-4. It has no label-budget axis, so do not present it as a crossover study.
- **Low-code workflows (2505.24189)**: the abstract says "fine-tuning improves quality by 10% on average"; the arXiv comment names a workshop, and I could not confirm the "KDD 2025" label (UNVERIFIABLE).
- **"OLMo forgets early data" (2410.03249)**: the actual title is *How Much Can We Forget about Data Contamination?* (Bordt et al., ICML 2025). Models up to 1.6B plus OLMo-7B continual pre-training. The claim is a loose paraphrase, so cite it by its title.
- **S-LoRA (2311.03285)**: arXiv gives no venue (MLSys 2024 from memory, unverified).
- **Zheng et al. (2306.05685)**: NeurIPS 2023 Datasets & Benchmarks. The "over 80%" claim is a pairwise-preference claim (see §4).
- **Olmo 3 / Soft contamination (2602.12413)**: confirmed, "the Olmo corpus contains SemDs of... 100% of MBPP's".
- Spider/Findings-EMNLP-2024 row stays NOT VERIFIED; I did not check it.

## 2. Novelty search

**(a) Fine-tuning small vs ICL/prompting by label count**
- Pecher, Srba, Bielikova, *100 Labelled Samples to Achieve Break-Even Performance*, EMNLP 2025, arXiv 2402.12819. Fine-tuning, instruction-tuning, prompting and ICL on 8 models and 8 text-classification tasks, with break-even counts that account for variance (about 100 labels on average; 100–200% more once variance is counted). No NER, summarisation, API LLMs, cost or knowledge analysis. **The earlier researcher missed this; it is the closest prior work.**
- Mosbach et al., Findings ACL 2023, 2305.16938. Fair comparison at matched sizes with few examples. No curve beyond few-shot, no API, no cost.
- Bertsch et al., NAACL 2025, 2405.00200. Many-shot ICL vs retrieval vs fine-tuning as data grows ("finetuning is more data hungry than ICL but can exceed long-context ICL"). Open models, no commercial API, no cost, no knowledge analysis.
- Bucher & Martini, 2406.08660, and Wang et al., 2411.05050. Fine-tuned small models vs zero/few-shot GPT on classification. Two or three data points, not curves.
- Agarwal et al., NeurIPS 2024, 2404.11018. ICL scaling only.

**(b) Cost-aware comparisons**
- Mansuri et al., 2609.40236 (Sep 2026). Fine-tuned open model vs GPT-4o on radiology extraction, five training sizes, and a break-even volume against standard and Batch API (per the search summary; I did not read the paper). One task, a 100-report test set.
- Valdes Gonzalez, 2602.06370. Fine-tuned encoders vs prompted LLMs, multi-objective with latency and cost. Four sentiment/topic benchmarks, no label curve.
- Mathur et al., 2610.24260. RoBERTa vs GPT-3.5 on survey classification, cost counted with annotation effort. One task.
- Luccioni et al., FAccT 2024, 2311.16863. Energy per 1,000 inferences, not accuracy-adjusted.
- Alizadeh et al., 2307.02179. Annotation budget advice for open LLMs.

**(c) Retrieved/kNN demonstrations vs fine-tuning on the same pool**
- Bertsch (above) is the best match. Trad & Chehab, 2512.04106, find that retrieval-augmented few-shot beats a fine-tuned Gemini on vulnerability detection (single task). Luo et al., Dr.ICL, 2305.14128, is retrieval only. No source varies N for both arms against a commercial API.

**(d) Does fine-tuning add or elicit knowledge?**
- Gekhman et al., EMNLP 2024, 2405.05904: new-knowledge examples are learned slowly. Sun & Dredze, *Amuro and Char*, 2408.06663: fine-tuning gains concentrate on tasks not learned in pretraining, at dataset level on OLMo checkpoints. Wu et al., 2404.12957: latent-knowledge estimation. Spiesberger et al., 2602.12413: soft contamination in Olmo 3's post-training data. Bordt et al.: contamination forgetting.
- I found **no work that stratifies per item by open-pretraining-data overlap (Dolma/OLMo) and relates it to fine-tuning-vs-ICL gains**. The search was not exhaustive (I did not query OLMoTrace or WIMBD papers directly).

**(e) PEFT variants**
- Lee et al., *Learning Rate Matters: Vanilla LoRA May Suffice*, 2602.04998: with tuned learning rates all variants peak within 1–2%. He et al., *A Unified Study of LoRA Variants*, 2601.22708. Wang et al., FinLoRA, 2505.19819 (DoRA worse, rsLoRA best on some financial tasks). Biderman et al., 2405.09673. None sweeps N ∈ {50…5000} across classification, NER and summarisation, and I found none covering NER. **Design implication: tune the learning rate per variant or the variant comparison is confounded.**

**Verdict on "the combination is new":** partly true. Each ingredient exists, and Pecher et al. + Bertsch et al. + Mansuri et al. cover the curve, retrieval and cost-break-even parts separately. I found nothing combining (i) retrieved-ICL from the same N-pool against a cheap commercial API, (ii) several task families, (iii) cost per correct answer, and (iv) a per-item pretraining-overlap analysis. "Novel" is defensible only for that conjunction, and only as "to my knowledge, as of Nov 2026". The search was not exhaustive.

**Strongest framing:** a paired, equal-pool, cost-normalised label-budget study that (1) locates the crossover N per task, (2) prices it as break-even query volume, and (3) tests per item whether fine-tuning gains come from items the base model's pretraining data already covered. Item (3) is the least contested contribution.

**MUST-CITE:** 2402.12819, 2305.16938, 2405.00200, 2404.11018, 2405.00732, 2405.05904, 2408.06663, 2602.12413, 2410.03249, 2602.04998, 2609.40236, 2602.06370, 2610.24260, 2311.16863.
**Scoop risks:** Pecher et al. extending to NER/APIs/cost; Mansuri-style cost papers adding more tasks; any Olmo-3 contamination paper that adds fine-tuning (the Spiesberger group). Re-run the search before submission, since 2609/2610 ids show the field is moving monthly.

## 3. Statistics
- **McNemar sample size.** Connor (1987), Biometrics 43:207–211, DOI 10.2307/2531961: n = [z_{α/2}√ψ + z_β√(ψ−δ²)]² / δ², ψ = discordant proportion. My calculation for ψ=0.10, δ=0.03, power 0.8: α=0.05 → **869.7 (≈870)**; α=0.005 → **1,476 (≈1,480)**. Claim confirmed as a normal approximation. Exact conditional binomial power (my simulation, scipy) needs **n≈920** and **n≈1,520**. The extra ≈5% matters at small margins. Caveats: (i) ψ=0.10 is optimistic when comparing a fine-tuned 3B model with an API LLM, whose answers typically differ on far more items. At ψ=0.20 n≈1,740, at ψ=0.30 n≈2,610. (ii) α=0.005 is a worst-case Holm bound (Holm's first step only), which is fine as a conservative choice.
- **Paired bootstrap:** Koehn (2004) EMNLP *Statistical Significance Tests for MT Evaluation*; Berg-Kirkpatrick et al. (EMNLP 2012, aclanthology D12-1091): confirmed. Dror et al. ACL 2018 (P18-1128) in sources.md. Holm (1979, Scand. J. Stat. 6:65–70), Schuirmann (1987, J. Pharmacokinet. Biopharm. 15:657–680, TOST = the 1−2α CI inside ±margin; confirmed via Zenodo copy), Efron & Tibshirani (1993) *(Holm, Efron from memory)*.
- **Questionable plan items:** (1) *Majority vote over seeds then McNemar*: it discards seed variance and changes the system under test. Prefer per-seed predictions, a paired bootstrap that resamples items and treats seed as a random effect, or report the seed-averaged correctness per item. (2) *3 seeds as a sample*: three seeds cannot estimate a variance; use items as the inference unit and show seed range descriptively. Bootstrapping over both items and seeds is the multi-bootstrap of Sellam et al., MultiBERTs, ICLR 2022, 2106.16163; it needs many seeds. Pecher et al. show variance inflates break-evens, so report it. (3) Permutation tests for differences of group means are valid under exchangeability only; do not use them across heterogeneous tasks. (4) TOST needs a pre-registered margin; a bootstrap CI inside ±margin is equivalent to TOST (Schuirmann).

## 4. Metrics
- **BERTScore**: README says "the best model is `microsoft/deberta-xlarge-mnli`", and `--rescale_with_baseline` is supported (v0.3.0+); report the hash code. Rescaling changes range, not ranking.
- **ECE**: Guo et al. ICML 2017 (1706.04599) uses 15 equal-width bins. Criticisms: Nixon et al. 2019 (1904.01685) show bin-count sensitivity and propose adaptive binning; Roelofs et al. 2020 (2012.08668) show a bias in the binned estimator. Report it with bin count and a debiased/adaptive variant. *(Nixon/Roelofs claims from titles/abstracts only.)*
- **Typed NER micro-F1**: seqeval default mode replicates conlleval (exact span + type); strict mode is separate. Report the mode and the encoding; Palen-Michel et al., SeqScore (2107.14154), show scoring differences are significant. Multiset exact-match for generative NER is a different metric from conlleval; label it distinctly.
- **LLM judge (Zheng et al.)**: ">80%" is for GPT-4 vs human pairwise preferences on MT-bench/Arena: 66% with ties, **85% without ties**, against 63% / 81% human–human. It does not support graded summary scoring. Validate your own judge on your own labels with Krippendorff's α (ordinal for graded scales); thresholds like ≥0.80 are conventions, and α is misleading on skewed labels (arXiv 2509.12382).

## Claim table
| Claim | Verdict | Source |
|---|---|---|
| n≈870 / ≈1,480 | Confirmed (normal approx.); exact ≈920 / ≈1,520 | Connor 1987; my recomputation |
| Majority-vote McNemar over seeds | Questionable | §3 |
| "Nobody published break-even curves by label budget" | **Refuted for classification** | 2402.12819 |
| LoRA Land beats GPT-4 by 10 pts | Confirmed, zero-shot GPT-4, no N axis | 2405.00732 |
| Judge >80% agreement | Partly: pairwise, GPT-4 only | 2306.05685 |
| BERTScore deberta-xlarge-mnli | Confirmed | bert_score README |
| ECE 15 bins | Confirmed; known bias | 1706.04599 |
| Combination is novel | Partly; conjunction only | §2 |

## Recommended novelty statement
Prior work shows that small fine-tuned models overtake prompted or in-context LLMs after roughly 100–1,000 labels on classification (Pecher et al., 2025), and that retrieval and many-shot ICL compete with fine-tuning on open models (Bertsch et al., 2025). We add, to our knowledge, the first equal-pool comparison across classification, NER, and summarisation of PEFT-tuned small open models against retrieval-augmented cheap commercial APIs. It reports paired crossover points, break-even query volumes and cost per correct answer, and tests per item, using Olmo's open data, whether fine-tuning gains arise on items already covered by pretraining.
