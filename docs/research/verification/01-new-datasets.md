# Verification 01 — Spider/BIRD, LEDGAR, SEntFiN (2026-11-21)

Method: primary pages fetched directly (HF cards/API, GitHub, papers via pdftotext). Verdicts: C = CONFIRMED, P = PARTLY, R = REFUTED, U = UNVERIFIABLE. Not legal advice; I report what texts say.

## 1. Spider (and BIRD)

| Claim | Verdict | Source / evidence |
|---|---|---|
| HF `xlangai/spider` is CC BY-SA 4.0 | C | huggingface.co/datasets/xlangai/spider card: "licensed under the CC BY-SA 4.0" |
| Yale page licence | C | yale-lily.github.io/spider: "distributed under the CC BY-SA 4.0 license" |
| `taoyds/spider` GitHub LICENSE is the same | **R** | GitHub LICENSE is **Apache-2.0** (code repo; README states no data licence). `taoyds/test-suite-sql-eval` is also Apache-2.0 |
| Paper carries a data licence | R | arXiv 1809.08887 states none (only "publicly available") |
| Databases/SQL carry the same licence | U | Yale page says "a copy of the dataset" is CC BY-SA, but ~70 DBs came from course/tutorial/textbook sources, ~40 from DatabaseAnswers (schemas), 90 built on WikiSQL/Wikipedia (paper §3). No per-DB licence given. Share-alike would reach derivatives; whether it reaches a LoRA adapter is a legal question |
| Train 7,000 / dev 1,034 | C | HF size API: train 7000, validation 1034. Paper: "8659 train, 1034 dev, 2147 test" (8,659 includes 1,659 extra questions from older datasets; HF has only 7,000 `train_spider`) |
| Test (2,147) held back | **R (now)** | Yale page, 02/05/2024: "test set of Spider 1.0 has already been released"; leaderboard closed. Not on the HF card (only train+validation) |
| Frozen test set | — | Dev 1,034 < 1,000 margin is thin: it passes ≥1,000 only barely, and dev is public on GitHub since 2018. Released test (2,147, public labels) gives more items but equal contamination risk. Neither is hidden |
| Execution-accuracy tooling | P | `test-suite-sql-eval` README: databases "download... from here" = a **Google Drive link**; no licence stated for that archive; the HF parquet has **no SQLite files** |
| Typical prompt length with schema | U | I did not measure it; no source fetched |
| Contamination | P | Ranaldi et al. 2024 (arXiv 2402.08100) report GPT-3.5 has prior knowledge of Spider dev; Rajkumar et al. 2022 (arXiv 2204.00498) argue it did not drive Codex results. Mixed, single-model evidence (from search summaries, not re-read) |
| "DTS-SQL reaches GPT-4 parity" | **P** | aclanthology 2024.findings-emnlp.481, Table 2 (Spider **test** EX): DTS-SQL+DeepSeek-7B 85.5 vs DAIL-SQL+GPT-4 84.4, DIN-SQL+GPT-4 74.2. Caveats: trained on the **full** train set (not 50–5,000), the GPT-4 rows are 2023 prompting methods, and the model is code-pretrained. Abstract says "comparable", not parity |

**BIRD** (arXiv 2305.03111; README; bird-bench.github.io): train 9,428 / dev 1,534 / concealed test 1,789 (paper appendix: "1,789 instances for the concealed test set"; test databases newly curated, README: scoring by emailing the organisers). Licence: site and README: "we now modify the license of our data to CC BY-SA 4.0" (Apr 27 2024); the repo is MIT (code). Databases: 32% Kaggle, 48% CTU Prague Relational repository, 20% built from open tables; paper says "accessible with the appropriate licenses" — per-source terms **U**. Databases total 33.4 GB (site), so schema/value context is heavy and evaluation is slower; dev labels are public, so BIRD's hidden test cannot be scored locally.

**Fit:** BIRD is cleaner on hidden test and size (train ≥5,000, dev 1,534), but its test cannot be scored without the organisers, so the frozen test would have to be dev (public, cleaned in 2023/2025 per the site). Spider is easier to run but its test is public and probably seen by pretraining. Neither gives a truly hidden item set.

## 2. LEDGAR / LexGLUE

| Claim | Verdict | Evidence |
|---|---|---|
| 60,000/10,000/10,000, 100 labels | C | HF size API: train 60000, test 10000, validation 10000; stats API: 100 classes. LexGLUE paper: "training (60k, 2016–2017), development (10k, 2018), test (10k, 2019)" |
| CC BY 4.0 on the LexGLUE card | P | YAML says `cc-by-4.0`, but the card's "Licensing Information" section reads "[More Information Needed]". LexGLUE paper says only datasets are "available for re-use and re-share with appropriate attribution" |
| Original LEDGAR licence | **U** | LREC 2020 paper: **no data licence stated**; page footer "licensed under CC-BY-NC" is the *paper's* copyright line; corpus hosted on a SWITCHdrive link. Do not assume CC BY 4.0 comes from the authors |
| EDGAR filings public domain | P | Paper: "crawled and scraped from the public domain (SEC filings)". SEC FAQ (via search): "EDGAR public filing content are free to access and reuse". But exhibit text is written by third parties; a 2010 law-firm post reports a court finding copyright issues in EDGAR-sourced documents. SEC pages blocked my fetch (rate limit) |
| Label imbalance | C | train frequencies (HF stats): largest "Governing Laws" 3,167, smallest 23; top-5 sum 14,235 |
| Metrics | C | LexGLUE table: TF-IDF+SVM μ-F1 87.2 / m-F1 82.4; BERT 87.6 / 81.8 |
| Input length | C | train chars: median 532, mean 708, max 7,803 (HF stats) |

LEDGAR is **single-label** in LexGLUE (multilabel with 12k+ labels in the original); the 100-label set was derived by LexGLUE keeping the top-100 labels. Provisions are boilerplate-heavy; neighbours like "Notices" vs "Counterparts" are easy, "Governing Laws" vs "Jurisdictions" vs "Venues" overlap. Few-shot with 100 classes needs ≥1 example per class, so N=50 cannot cover all labels. Relative to BANKING77 it is a similar single-label, large-label-set classification, so it adds domain (legal) but little *task-type* diversity.

## 3. SEntFiN

| Claim | Verdict | Evidence |
|---|---|---|
| 10,753 headlines | C | CSV: 10,753 rows (github.com/pyRis/SEntFiN); paper abstract |
| Repo MIT | C | GitHub licence API: MIT; "Copyright (c) 2021 Ankur Sinha, ..." |
| MIT covers headline text | **U / probably not** | Paper: headlines "sampled from ... Indian business news provider The Economics Times" (2011–2015) [sic]; **no statement of publisher permission or rights**. MIT from the authors cannot license text they did not write. Annotations are theirs |
| Single CSV, no official split | C | Repo holds `SEntFiN.csv` (columns S No., Title, Decisions, Words) and `entity_list_comprehensive.csv`; the paper uses random 80/20 splits, 31 repeats |
| 2,847 multi-entity | C | My count from CSV: 2,847 with ≥2 entities; 1,233 conflicting (matches paper) |
| Annotations count | P | Paper: 14,404; my regex count: 14,405 (±1) |
| Label balance | C | Paper: pos 35.23 / neg 26.48 / neu 38.29 (CSV: 5,074 / 3,814 / 5,517). Single-entity headlines (7,906): 2,837 / 2,376 / 2,693 |
| Agreement | C | Paper: pos–neu agreement 80.36% (lowest); no kappa seen |

Formulation: per-(headline, entity) instance, 14.4k items, 3 classes; headline-level labels are ill-defined for 1,233 conflicting headlines. Split **by headline** to avoid leakage: e.g. 7,753 train / 1,000 test headlines would give ~10.4k / ~1.3k instances, so ≥5,000 train and ≥1,000 test is feasible, but little headroom (no separate dev pool). Short (≈10 words), RoBERTa/finBERT ≈93% F1 reported, so a ceiling effect is likely. Known issues: single geography/publisher, 2011–2015, entity spans are given (not predicted), ±1 count discrepancy.

## Twitter Financial News (`zeroshot/...`)
HF size API: train 9,543 / val 2,388 (the card's own text says 11,932 docs and 9,938/2,486 — **internally inconsistent**). Label bins: 1,442 / 1,923 / 6,178 (neutral 65%) — imbalanced. Card: MIT; "collected using the Twitter API". Tweets fall under X's developer terms, and MIT from the uploader cannot grant rights to others' tweets; the card cites no annotation protocol (`annotations_creators: other`). Text length: mean 86 chars.

## Summary table

| Dataset | Use | Main risk | Human must check |
|---|---|---|---|
| Spider | secondary | Public test, contamination, DB licence unclear, test DBs on Google Drive | Licence of DB sources (DatabaseAnswers, WikiSQL); whether adapters count as derivative under CC BY-SA |
| BIRD | secondary (alt.) | Hidden test unscoreable locally; 33 GB DBs; Kaggle/CTU source terms | Kaggle and CTU Prague terms; run time budget |
| LEDGAR | primary (legal) | No explicit original licence; third-party exhibit copyright; 100 labels vs N=50 | Terms of the SWITCHdrive release; EDGAR reuse policy |
| SEntFiN | primary only if publisher rights cleared, else secondary | Economic Times headline rights; shallow task | Contact authors/ET; confirm headline reuse |
| Twitter fin. | secondary | Tweet rights; card inconsistencies; 65% neutral | X/Twitter terms; macro-F1 reporting |

## Recommendation
- **Replace code generation with: LEDGAR**, not Spider/BIRD. Reason: ≥60k train, 10k test chronological and frozen, text classification keeps tooling light, and legal is a real enterprise domain. Its weakness is the licence gap (flagged above). Spider/BIRD both require database execution infrastructure and neither has a hidden, locally scoreable test. If code generation must stay, use BIRD dev as test with the contamination caveat disclosed.
- **Finance primary: SEntFiN** is the better fit by schema (human-annotated, balanced, per-entity), *but* its headline-rights question is unresolved; the Twitter set has tweet-rights and imbalance issues of its own. Choose SEntFiN only after the human check; neither is licence-clean.
