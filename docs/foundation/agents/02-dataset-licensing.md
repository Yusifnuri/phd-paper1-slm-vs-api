# A2: Datasets, licensing, provider terms

Agent A2, 2026-11-21 (access date of every source below). Not legal advice: it records what texts say and where evidence cannot settle a question. Labels: **VF** verified fact (primary text read), **INT** interpretation, **REC** recommendation, **UNV** unverified. Confidence H/M/L. "Own check" = I computed it from downloaded files (TF-IDF cosine on character or word n-grams: near-duplicate counts are indicative).

Rights kept apart: (1) dataset licence, (2) underlying text, (3) model licence, (4) API terms, (5) academic vs commercial use, (6) adapters and released predictions/statistics.

## 1. Eligibility matrix

| Dataset | Dataset licence | Underlying text | Academic | Commercial claim | Release adapters | Release predictions, statistics | Verdict |
|---|---|---|---|---|---|---|---|
| BANKING77 | CC BY 4.0 | provenance not stated (UNV) | yes | yes, attribution (INT) | yes | yes | **Include** |
| CLINC150 | CC BY 3.0 | crowd-written for Clinc | yes | yes, attribution (INT) | yes | yes | **Include** |
| BillSum (US split) | none from FiscalNote; CC0 only in card metadata | US federal works, public domain by statute | yes | yes for US text (INT) | yes | yes | **Include with conditions** |
| LEDGAR (LexGLUE) | MIT (LREC list) and CC BY 4.0 (LexGLUE metadata) | third-party contracts filed with SEC | yes | probably (INT) | yes | ids and labels, not raw text (REC) | **Include with conditions** |
| SAMSum | CC BY-NC-ND 4.0 (authors' files) | linguist-written | yes, non-commercial | **no** | uncertain (ND) | statistics yes; predictions uncertain | **Include with conditions** |

## 2. Evidence ledger

| # | Fact | Source (location) | Quote (≤15 words) | Conf | Label |
|---|---|---|---|---|---|
| B1 | BillSum CC0 appears only as card YAML; licence section empty; repo has no licence file (GitHub API 404) | https://huggingface.co/datasets/FiscalNote/billsum README; https://api.github.com/repos/FiscalNote/BillSum/license | "Licensing Information ... [More Information Needed]" | H | VF |
| B2 | Card ties CC0 to GovInfo, not FiscalNote | same, "Initial Data Collection" | "Govinfo service provided by the ... GPO under CC0-1.0 license" | H | VF |
| B3 | Composition; no data licence in paper | arXiv 1910.00523 §3 | "18,949 train bills and 3,269 test bills" | H | VF |
| B4 | US summaries by CRS; CA summaries by Legislative Counsel | same §1, §3 | "summary from the Congressional Research Service (CRS)" | H | VF |
| B5 | Federal works not copyrightable; embedded third-party material possible | https://www.govinfo.gov/about/policies, "Public Domain & Copyright Notice" | "Copyright protection ... is not available for any work of the United States Government" | H | VF |
| B6 | California declares its bill data public domain | https://leginfo.ca.gov/ home notice | "within the public domain and the State of California retains no copyright" | M | VF |
| B7 | A statement covering CRS bill summaries specifically | not found (Congress.gov returned 403) | – | L | UNV |
| L1 | LREC shared-resources list gives MIT | https://lrec2020.lrec-conf.org/en/shared-lrs/ entry LEDGAR | "License MIT License ... Conditions of use Freely Available" | H | VF |
| L2 | Paper states no licence; "CC-BY-NC" is the article's ELRA copyright | LREC paper p.1, https://aclanthology.org/2020.lrec-1.155 | "licensed under CC-BY-NC" | H | VF |
| L3 | Archive on SWITCHdrive; page text shows no licence (JS-rendered, file list unread) | https://drive.switch.ch/index.php/s/j9S0GRMAbGZKa1A | title "LEDGAR" only | M | VF/UNV |
| L4 | Authors call source public domain | paper abstract | "crawled and scraped from the public domain (SEC filings)" | H | VF |
| L5 | SEC reuse statement | https://www.sec.gov/about/webmaster-frequently-asked-questions | "EDGAR public filing content are free to access and reuse" | H | VF |
| L6 | Third-party copyright may survive in EDGAR documents (ISDA v. Socratek, secondary blog) | https://www.pattishall.com/pdf/8-17-10_Copyright_Infringement_in_Edgar_Docs_Blog_Post.pdf | "mass resale of SEC filings that include copyrighted material ... may violate" | M | VF (blog), INT |
| L7 | LexGLUE card: CC BY 4.0 in YAML, prose empty; paper terms | https://huggingface.co/datasets/coastalcph/lex_glue; arXiv 2110.00976 "Licensing & Personal Information" | "available for re-use and re-share with appropriate attribution" | H | VF |
| L8 | LexGLUE derivation | arXiv 2110.00976 §3 | "considering only the 100 most frequent categories" | H | VF |
| S1 | SAMSum is CC BY-NC-ND 4.0 per the authors; licence.txt is the full legal code | https://arxiv.org/src/1911.12237v2/anc/corpus.7z (`README.txt`, `licence.txt` only); paper §1 | "non-commercial licence: CC BY-NC-ND 4.0" | H | VF |
| S2 | Licence core | `licence.txt` §2(a)(1) | "produce and reproduce, but not Share, Adapted Material" (NonCommercial only) | H | VF |
| S3 | Provenance | README.txt | "created and written down by linguists fluent in English" | H | VF |
| N1 | BANKING77: CC BY 4.0 on card and repo LICENSE; the file sits at the root of a multi-dataset repo and the README does not scope it (INT) | https://huggingface.co/datasets/PolyAI/banking77; https://github.com/PolyAI-LDN/task-specific-datasets | "Attribution 4.0 International" | H | VF |
| N2 | Paper does not say how queries were produced | arXiv 2003.04807 §3 | "13,083 customer service queries labeled with 77 intents" | H | VF (absence) |
| N3 | CLINC licence | https://github.com/clinc/oos-eval LICENSE | "Attribution 3.0 Unported" | H | VF |
| N4 | CLINC provenance, split rule | repo README FAQ 6; arXiv 1909.02027 §2.3 | "all queries from a given crowd worker in a single split" | H | VF |
| M1 | Qwen3-4B-Instruct-2507 Apache-2.0 (LICENSE file); Phi-4-mini-instruct MIT (LICENSE file) | HF repos | "Apache License Version 2.0"; "Permission is hereby granted, free of charge" | H | VF |
| M2 | OLMo-3-7B-Instruct: Apache-2.0 in card, **no LICENSE file**; Ai2 guidelines exempt research | HF card "License"; https://allenai.org/responsible-use | "Research Use" exempt from disallowed-use list | H | VF |
| P1 | Anthropic Commercial Terms D.4; customer owns outputs; no training on API data | https://www.anthropic.com/legal/commercial-terms | "build a competing product or service, including to train competing AI models" | H | VF |
| P2 | OpenAI business terms: competing-model ban with exception for non-distributed classifiers; customer owns output | https://openai.com/policies/business-terms/ | "use Output to develop artificial intelligence models that compete with OpenAI" | H | VF |
| P3 | Google Gemini API terms; unpaid tier used for training, humans may read | https://ai.google.dev/gemini-api/terms | "may not use the Services to develop models that compete" | H | VF |
| P4 | No clause on publishing benchmarks/comparisons found (searched "benchmark", "compet", "publish" in the three texts and Google API terms) | pages above | – | M | VF (absence) |

## 3. Per dataset

### BillSum
- Original: Kornilova & Eidelman 2019; https://github.com/FiscalNote/BillSum (data on Google Drive). US train 18,949 / US test 3,269 / California test 1,237 (B3; HF sizes match).
- **CC0 cannot be established as a licence granted by FiscalNote** (B1, B2). What the sources support instead: US federal works are outside copyright (B5) and CRS summaries are federal-employee work (B4; INT, B7 unconfirmed for summaries). So the basis is statute; cite it as such. Caveats: embedded third-party material (B5); §105 operates in the US, and foreign status of US federal works is unsettled but low-risk. California is a separate holder (B6), not needed: use US `test` only (REC).
- Length, US test, characters (HF statistics): min 5,004, median 9,316, mean 10,268, max 19,998 (construction range 5,000 to 20,000). Tokens (own check, 400 random bills): Qwen3 mean 2,109, median 1,899, p90 3,361, p99 4,248, max 4,871; Phi-4-mini mean 2,045; OLMo-3 mean 2,047 (about 5 chars/token). Reference summary mean 238, max 892 tokens. A 5-shot prompt is about 10k to 12k tokens before the test bill.
- Leakage (own check): 1 test text exactly in train; 26 duplicate texts inside train; word-TF-IDF cosine to nearest train bill at least 0.95 for **297** test bills (9.1%), at least 0.9 for 551 (16.9%), at least 0.8 for 901 (27.6%): re-introduced bills across Congresses.
- Contamination: crawlable government text; one weak infini-gram hit in the earlier report; no model-family statement (UNV).

### LEDGAR
- **Reviewer claim verified**: the LREC list says "MIT License" (L1). The paper (L2) and readable SWITCHdrive page (L3) state none, so the MIT label comes from the LREC resource entry, not a licence file I could read. LexGLUE says CC BY 4.0 (L7). Both are permissive; attribute Tuggener et al. and LexGLUE and keep the MIT notice (INT).
- Underlying rights: contracts are third-party exhibits. The SEC says filing content is free to reuse (L5); a court found copying EDGAR documents containing third-party copyright may infringe (L6, secondary). No source addresses it for short provisions; risk judged low (INT), not zero.
- Derivation (L8): top 100 labels from about 12.5k, single label, 60k/10k/10k chronological 2016-17 / 2018 / 2019.
- Label frequency (HF statistics, 3 small calls): train min 23 (Books), 31 (Assigns), 47 (Qualifications), then 106; max 3,167. Test: min 2; 31 classes under 50 items; 3 under 10; no test label absent from train. Test chars median 484, max 6,542. N=50 cannot cover 100 classes; report micro and macro F1 with CIs (REC).
- Leakage (own check, full parquet): **338 test texts (3.4%) exactly equal a train/validation text** (2 with a different label); 920 duplicate texts inside train; char-4-gram cosine at least 0.95 for 1,849 test items (18.5%), at least 0.9 for 2,922 (29.2%), and at those levels the nearest train item has the same label 97.6% of the time. Boilerplate overlap favours kNN and retrieval-ICL as well as fine-tuning. Report overall and novel-text results (I-08).
- Contamination: public EDGAR text; nothing found for the three models (UNV).

### SAMSum
- **Licence established** (S1, S2): CC BY-NC-ND 4.0 in the authors' README, full legal code and paper; the HF mirror `knkarthick/samsum` agrees (cross-check only). The official HF page returned 401.
- INT: academic non-commercial evaluation and fine-tuning allowed; statistics are not licensed material; unaltered gold summaries may be shared non-commercially with attribution. Whether released predictions or LoRA weights are "Adapted Material" is an open legal question the licence text does not answer. Enterprise-use claims cannot rest on SAMSum.
- Prior use: Bertsch et al. 2025 use SAMSum (§2) and discuss no licence (no "licen" in the text).
- Leakage (own check, authors' files 14,732/818/819): **42 test dialogues exactly match a train dialogue**; 48 at cosine at least 0.9; no test summary at 0.9 (3 at 0.8). About 5% of test.
- Contamination: public since 2019; no document naming it for these models found (UNV).

### BANKING77
- Casanueva et al. 2020; PolyAI repo; CC BY 4.0 (N1). Residual risk small: repo scoping (N1) and unstated query provenance (N2); no other rights holder identified.
- Sizes 10,003/3,080 confirmed from raw CSVs. Leakage (own check): 11 test items exactly in train (0 label conflicts); 9 duplicate texts in train; 417 test items (13.5%) at cosine at least 0.9, 1,013 at least 0.8 (templated paraphrases, inherent to the design).
- Known issue: paper §3 says some intents "partially overlap". No label-noise audit found in primary sources (UNV).
- Prior use: Bertsch 2025 (Table 1). Pecher 2025 uses AG News, TREC, SNIPS, DBPedia, GLUE tasks, not our datasets.

### CLINC150
- Larson et al. 2019; `clinc/oos-eval`; CC BY 3.0 (N3); HF metadata `cc-by-3.0`, card prose empty. Crowd paraphrase and scenario tasks; lowercased, punctuation removed, duplicates removed; per-worker splits (N4).
- Leakage (own check, `data_full.json`): 26 of 4,500 in-scope test queries exactly in train (2 with different label); 523 (11.6%) at cosine at least 0.9.
- HF "plus" adds OOS items (15,250 / 5,500 total): choose the 150 in-scope classes and fix OOS handling explicitly (I-16). Prior use: Bertsch 2025 (Table 1).

## 4. Models and API terms

| Item | Text says | Effect |
|---|---|---|
| Qwen3-4B-Instruct-2507, Phi-4-mini-instruct, OLMo-3-7B-Instruct | Apache-2.0, MIT, Apache-2.0 (OLMo: card only) (M1, M2) | adapters and results may be released |
| Anthropic, OpenAI, Google | each bans using the service/outputs to build competing models (P1 to P3); customers own outputs (Anthropic, OpenAI) | evaluating API outputs is fine; **never train adapters on API outputs**; all fine-tuning labels must be gold |
| Google unpaid tier | inputs and outputs used for training and read by humans (P3) | use the **paid** tier for all dataset text |
| Benchmark publication | no restricting clause found (P4) | publishing comparisons not barred by texts read; confidence M (incorporated policies not all read) |

Record the terms version and date in released result files (REC).

## 5. Contamination by model family

Phi-4-mini documents n-gram decontamination against unnamed "benchmark datasets" (Phi card). Qwen3: no statement found. OLMo-3: decontamination targets its evaluation harness, none of our datasets (earlier report, not re-read). Dolma 3 and Dolci could not be searched: the HF search index was still loading. All five datasets are public for 3 to 7 years: treat scores as public-benchmark scores (I-17); stratify by the leakage numbers above and add a small label-permuted or post-cutoff probe (REC). "None found" is absence of evidence.

## 6. Conclusions; what only the human can settle

BANKING77, CLINC150: include. BillSum: include, US test only, "public-domain federal text" wording, near-duplicate stratification. LEDGAR: include, cite MIT and CC BY 4.0, no raw-text redistribution, stratify duplicates, class-coverage rules. SAMSum: include non-commercially, no adapter release, exclude the 42 exact-overlap test items from the headline set.

Genuinely human-dependent:
1. Whether a SAMSum-trained adapter or its predictions are "Adapted Material" (no authority resolves it; the safe course is not to release; the human decides only whether that loss matters).
2. Whether third-party copyright in LEDGAR exhibits reaches short-provision reuse (SEC statement, authors' MIT entry and one court decision point different ways; only counsel or the authors could close it).
3. Optionally, written confirmation from the BillSum/LEDGAR authors or Samsung (no CRS-summary-specific statement was reachable).

## 7. New issues and register views

| ID | Issue | Sev | Proposal |
|---|---|---|---|
| A2-01 | SAMSum test: 42 exact, 48 near train dialogues | HIGH | remove or stratify; into I-08 |
| A2-02 | LEDGAR test: 338 exact, 2,922 near-duplicates | HIGH | overall and novel-text reporting; I-08, I-12 |
| A2-03 | BillSum: 297 test bills at cosine ≥0.95 to train | MEDIUM | same |
| A2-04 | BillSum "CC0" is not FiscalNote's licence | MEDIUM | statutory wording; US only |
| A2-05 | LEDGAR licence stated in two places that differ; archive shows none | LOW | cite both |
| A2-06 | Gemini unpaid tier trains on inputs | MEDIUM | paid tier only; tell A5 |
| A2-07 | OLMo-3 repo has no LICENSE file | LOW | archive the card |
| A2-08 | Provider terms unversioned | LOW | record dated terms |
| A2-09 | Related work uses data without licence discussion (Bertsch SAMSum; Pecher "not able to determine the license") | LOW | no norm to lean on |

**I-11**: downgrade to MEDIUM and split: BANKING77, CLINC150, BillSum, LEDGAR are settled as far as evidence goes; SAMSum is a known NC-ND restriction, not an unknown; human items 1 and 2 remain.
**I-17**: keep MEDIUM; I have leakage counts, not memorisation proof.
**I-18**: close after recording dated terms: no benchmark-publication clause found; live risks are the competing-model clauses and the Google unpaid tier.

Not done: SWITCHdrive file list, Dolci/Dolma 3 searches, Congress.gov and California site terms beyond the LegInfo notice, HF `Samsung/samsum` card (401).
