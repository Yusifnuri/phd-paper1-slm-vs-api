# Dataset audit

Source: `agents/02-dataset-licensing.md` (A2, accessed 2026-11-21), amended by decisions L-07 and L-08. Not legal advice: it records what texts say. VF = verified fact (primary text read), INT = interpretation, REC = recommendation, UNV = unverified; confidence H/M/L. "Own check" = computed by A2 from downloaded files with TF-IDF cosine (near-duplicate counts are indicative). Rights are kept apart: dataset licence, underlying text, model licence, API terms, academic vs commercial use, release of adapters and predictions.

## 1. Role in the study and eligibility

| Dataset | Role (L-07) | Dataset licence | Underlying text | Commercial claim | Release |
|---|---|---|---|---|---|
| BANKING77 | **Confirmatory** | CC BY 4.0 (card and repository LICENSE; multi-dataset repo, README does not scope it) VF/H | provenance not stated (UNV) | yes with attribution (INT) | adapters, predictions, statistics |
| BillSum, US split only | **Confirmatory** | **No licence from FiscalNote**; CC0 appears only in HF card metadata, licensing section empty VF/H | US federal works; CRS summaries are federal-employee work (statement specific to CRS summaries not found, UNV) | yes for US text (INT) | adapters, predictions, statistics |
| LEDGAR (LexGLUE top-100) | **Exploratory** | MIT (LREC list) and CC BY 4.0 (LexGLUE); archive page shows none VF/H | third-party contracts filed with the SEC | probably (INT), risk low not zero | **ids, labels, metrics only**; no raw text, no adapters |
| CLINC150 | Not run (no budget slot) | CC BY 3.0 | crowd-written | yes with attribution | n/a |
| SAMSum | **Excluded** | CC BY-NC-ND 4.0 | linguist-written | **no** | n/a; adapter as "Adapted Material" legally open |

## 2. Key evidence

| # | Fact | Source / location | Conf |
|---|---|---|---|
| B1 | BillSum CC0 is card YAML only; repo has no licence file (GitHub API 404) | huggingface.co/datasets/FiscalNote/billsum; api.github.com/repos/FiscalNote/BillSum/license | H |
| B2 | Card ties CC0 to GovInfo/GPO, not to FiscalNote | same, "Initial Data Collection" | H |
| B3 | 18,949 train / 3,269 test US bills; California test 1,237 (separate rights holder, not used) | arXiv 1910.00523 §3 | H |
| B5 | Federal works not copyrightable; embedded third-party material possible | govinfo.gov/about/policies | H |
| L1 | LREC list gives "MIT License" for LEDGAR; the paper states no licence | lrec2020.lrec-conf.org/en/shared-lrs/; aclanthology.org/2020.lrec-1.155 | H |
| L5/L6 | SEC says filing content is free to reuse; a secondary source reports risk for embedded third-party copyright | sec.gov FAQ; Pattishall blog | H / M |
| L8 | LexGLUE keeps 100 most frequent labels, single label, 60k/10k/10k chronological | arXiv 2110.00976 §3 | H |
| M1/M2 | Qwen3-4B-Instruct-2507 Apache-2.0, Phi-4-mini-instruct MIT (LICENSE files); OLMo-3-7B-Instruct Apache-2.0 in card, **no LICENSE file** | HF repositories | H |
| P1–P3 | Anthropic, OpenAI, Google each ban using service/outputs to build competing models; Google's **unpaid tier trains on inputs** | provider terms pages | H |
| P4 | No clause restricting publication of benchmarks found in pages read; incorporated policies not all read | same | M |

Effects: evaluating API outputs is permitted; **never train adapters on API outputs** (all fine-tuning labels are gold); paid API tiers only; record terms version/date in released result files; cite BillSum as "US federal text, public domain by statute", not "CC0 from the authors".

## 3. Dataset facts that drive the design

| | BANKING77 | BillSum (US) | LEDGAR |
|---|---|---|---|
| Sizes | 10,003 train, 3,080 test, 77 classes | 18,949 train, 3,269 test | 60,000/10,000/10,000, 100 classes |
| Usable pool rows | 9,003 (1,000 validation removed) | about 17,950 | train minus validation |
| Test items used | 3,080 | 1,000 seeded | 3,000 seeded |
| Length (tokens) | query mean 12.8; 77-label prompt 476 | bill mean 2,111, max 7,393; summary mean 229, p99 760 | provision mean 136; label list 413 |
| Exact test/train duplicates | 11 (0 label conflicts) | 1 | 338 (3.4%; 2 conflicting) |
| Near-duplicates cosine ≥0.9 / ≥0.95 | 417 (13.5%) / – | 551 (16.9%) / **297 (9.1%)** | 2,922 (29.2%) / 1,849 (18.5%) |
| Other | templated paraphrases; some partially overlapping intents; no label-noise audit (UNV) | bills re-introduced across Congresses; retrieval may copy a prior CRS summary | train min class 23, test min 2 |

Treatment: near-duplicates are **kept** in the primary analysis and reported stratified by similarity bin (BillSum also 5-gram Jaccard); a novel-text sensitivity analysis excludes flagged items. Per-class results are descriptive only (~40 test items per BANKING77 class).

Coverage [V, HF statistics]: with random pools, the share of test items whose class is absent from the pool is BANKING77 50.5% (N=50), 26.1% (100), 7.5% (200), 0.4% (500); LEDGAR 13.3% at N=200. Hence stratified pools at N=200 only, as a named condition (L-04), and no fine-tuning below N=77.

## 4. Contamination (I-17, MEDIUM)

All datasets public for three to seven years. Phi-4-mini documents n-gram decontamination against unnamed benchmarks; Qwen3: no statement found; OLMo-3: decontamination targets its evaluation harness. "None found" is absence of evidence. Report absolute scores as public-benchmark scores, give first-public dates and zero-shot base accuracy per model, add a descriptive exact-prefix continuation probe (300 test texts per dataset, GPU) and optionally a post-cutoff BillSum-style probe (availability and licence unverified).

## 5. Remaining gaps

1. Third-party copyright in LEDGAR exhibits: avoided by releasing no raw text and no LEDGAR adapter (L-08).
2. No statement specific to CRS bill summaries reachable (Congress.gov 403): statutory basis only.
3. Incorporated provider policies, HF `Samsung/samsum` card, SWITCHdrive file list, Dolma 3/Dolci contamination search not done.

None is a human decision under the current release plan.
