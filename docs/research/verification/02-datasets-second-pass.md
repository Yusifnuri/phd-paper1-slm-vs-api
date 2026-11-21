# Datasets: second independent pass (2026-11-21)

Verifier: AI agent, read-only. Checked via HF hub API, datasets-server, raw card/repo files, docs.x.com. Not legal advice. Verdicts: CONFIRMED / PARTLY / REFUTED / UNVERIFIABLE.

## A. Licences, sizes, splits

| Claim | Verdict | Evidence |
|---|---|---|
| banking77 10,003/3,080, CC BY 4.0 | CONFIRMED | [card](https://huggingface.co/datasets/PolyAI/banking77): "Creative Commons Attribution 4.0 International"; repo LICENSE says same |
| clinc_oos plus 15,250/3,100/5,500, CC BY 3.0 | CONFIRMED (metadata + repo) | Card metadata `cc-by-3.0`; card prose says "[More Information Needed]". [Repo LICENSE](https://github.com/clinc/oos-eval/blob/master/LICENSE): "Attribution 3.0 Unported". Test is 5,500 in all configs |
| billsum CC0-1.0, 18,949/3,269 | PARTLY | Sizes right (also `ca_test` 1,237). Licence only in metadata `cc0-1.0`; card's Licensing section is empty. Original repo has no licence file (GitHub API 404). Underlying text is US-government bills. Mean text: 10,173 chars on 300 rows spread across train (range 5,022–19,969), consistent with ≈10,268 |
| few-nerd supervised 131,767/18,824/37,648 | CONFIRMED | [Repo](https://github.com/thunlp/Few-NERD): "distributed under the CC BY-SA 4.0 license" (code Apache-2.0) |
| BTC 5,342/2,002/2,002, CC BY 4.0 | PARTLY | Sizes confirmed. [Repo README](https://github.com/GateNLP/broad_twitter_corpus): "released as CC-BY 4.0"; but it ships "the raw tweet JSON from twitter". GitHub lists no licence file. I did **not** read the "2015 guidelines" wording: UNVERIFIABLE here |
| twitter-financial-news 9,543/2,388, MIT | PARTLY | API rows 9,543 train/2,388 validation (**no test split**). Card prose says 9,938/2,486, which is wrong. Card: "released under the MIT License"; "collected using the Twitter API". MIT covers the labels, not Twitter's rights in the text |
| self-oss-instruct 50,661, ODC-BY | CONFIRMED | Metadata `odc-by`, 50,661 train rows. Generated with a StarCoder2 pipeline; model-output terms not checked |
| mbppplus 378, humanevalplus 164, Apache-2.0 | CONFIRMED | Metadata `apache-2.0`, test only; evalplus repo Apache-2.0. Upstream MBPP is CC BY 4.0 (card metadata) |

**CoNLL-2003:** PARTLY usable only with caveats. [HF card](https://huggingface.co/datasets/eriktks/conll2003) (licence "other"): "Because of copyright reasons we only make available the annotations" and Reuters Corpus "can be obtained for research purposes without any charge". Reuters stories "are under the copyright of Reuters Ltd and/or Thomson Reuters"; the individual agreement is for researchers. I did not read the NIST agreements, so commercial use is UNVERIFIABLE and should be treated as not cleared. Exclude from an enterprise-licence claim.

**X terms (read 2026-11-21).** [Restricted use cases](https://docs.x.com/developer-terms/restricted-use-cases): "X prohibits any use of the X APIs and/or X Content to fine-tune or train a foundation or frontier model with the exception of Grok." Redistribution ([agreement page](https://docs.x.com/developer-terms/agreement-and-policy)): "you may only distribute Post IDs, Direct Message IDs, and/or User IDs"; limited hydrated sharing (up to 500 Post Objects per user per day, or 50,000 per recipient per day, non-automated only). Open points: the fine-tuning clause is on the restricted-use page, not in the agreement text I fetched; whether a 4B LoRA is a "foundation or frontier model" is a legal question; and tweets in BTC/twitter-financial-news were obtained under older terms. Redistributing tweet text in either dataset is not covered by the ID-only rule.

## B. MBPP / MBPP+ overlap (reproduced)

datasets-server `rows`, `google-research-datasets/mbpp` full config (train 374, validation 90, test 500, prompt 10) vs `evalplus/mbppplus` (378 unique ids, 2–809). **Train 108, test 224**: both CONFIRMED. Also validation 39 and prompt 7; 108+224+39+7 = 378, none outside full MBPP. So 147 of 378 are in MBPP's train+validation(+7 prompt = 154 in few-shot/train-like splits). Any MBPP-trained or MBPP-ICL pipeline must exclude these ids.

## C. Contamination (evidence level)

Evidence reachable:
- OLMo 3: decon removes "matches of any split of any benchmark dataset that are part of in our evaluation harness" ([report](https://arxiv.org/abs/2512.13961)), focused on midtraining; the harness lists MBPP, HumanEval(+), no classification/NER/summarisation sets from our list. [Soft-contamination paper](https://arxiv.org/abs/2602.12413) reports semantic duplicates of 100% of MBPP (as cited in the sources file; I read only search snippets).
- Dolci SFT card lists FLAN v2, Evol CodeAlpaca, Tulu-3 persona sets, OpenThoughts3; none of our datasets named. FLAN v2's contents were not enumerated by me.
- Phi-4-mini report: only "same processing and decontamination" (line mention); no dataset list. Qwen3 report and 4B-2507 card: no contamination statement found; data undisclosed.
- infini-gram exact-match probes (Dolma 1.7 and OLMo-2-mix indexes; 12 random rows per set, 8 for BillSum summaries): BANKING77 train 0/12, test 1/12 (count 3–16); CLINC train 2/12, test 4/12 (short generic queries, counts up to 8,506: non-diagnostic); Few-NERD 0/24; twitter-fin train 1/12 (count 1); BillSum 1/8 (count 2). These indexes are OLMo-2-era, **not Dolma 3**, and cover neither Phi nor Qwen.

| Dataset | Phi-4-mini | Qwen3-4B-2507 | OLMo-3-7B |
|---|---|---|---|
| BANKING77 | none found (public since 2020, web-scraped risk) | none found | none found; probes ≈0 |
| CLINC150 | none found | none found | none found; probes non-diagnostic |
| BillSum | none found; US bills are web/gov text, likely seen as raw text | none found | indirect (probe 1/8) |
| Few-NERD | none found; Wikipedia source likely seen as raw text, labels not | none found | none found (0/24) |
| BTC | none found | none found | none found (not probed) |
| twitter-fin | none found | none found | indirect (1/12 weak) |
| Self-OSS-Instruct | none found | none found | none found (not in Dolci card) |
| MBPP+ / HumanEval+ | none found (public since 2021; synthetic code data undisclosed) | none found (data undisclosed) | **documented**: decon target; plus soft contamination of MBPP |

"None found" is absence of evidence, not evidence of absence. Consequence for FT-vs-ICL: contamination of labelled train/test lets both arms recall answers, inflating absolute scores and shrinking the FT–ICL gap unevenly (large API models likely more exposed than a 4B model). For BANKING77/CLINC (public 2019–20) treat results as "public benchmark" scores; confirm any headline with a contamination-free probe (a small private or post-cutoff set, or label-permuted intents).

## D. Test-set size (frozen test, ≥1,000 paired items?)

BANKING77 3,080 yes; CLINC 5,500 yes; BillSum 3,269 yes (long docs, API cost); Few-NERD 37,648 sentences, subsample ≥1,000 yes; BTC 2,002 yes; twitter-fin 2,388 (validation, no test split) yes; Self-OSS-Instruct no test split (carve ≥1,000 hold-out from 50,661, but no execution-based held-out check); MBPP+ 378 NO; HumanEval+ 164 NO.

## Summary table

| Dataset | Verified facts | Commercial licence + derived adapters | Contamination | Role |
|---|---|---|---|---|
| BANKING77 | 10,003/3,080; CC BY 4.0 | Yes, with attribution (repo+card agree) | Low–unknown | Primary |
| CLINC150 plus | 15,250/3,100/5,500; CC BY 3.0 | Yes, attribution; card prose empty | Low–unknown | Primary |
| BillSum | 18,949/3,269; CC0 only in metadata | Likely yes; no original licence file | Unknown | Secondary (long input) |
| Few-NERD sup. | 131,767/18,824/37,648; CC BY-SA 4.0 | Share-alike: human to judge adapter status | Unknown | Secondary |
| BTC | 5,342/2,002/2,002 | Unclear: tweet text under X terms | Unknown | Exclude from commercial claim |
| twitter-fin | 9,543/2,388 (no test) | Labels MIT; tweet text under X terms | Weak signal | Secondary/exclude |
| Self-OSS-Instruct | 50,661; ODC-BY | Yes; check generator terms | Unknown | Train source only |
| MBPP+/HumanEval+ | 378/164; Apache-2.0 | Yes | OLMo documented | Exclude as primary (n<1,000; leakage) |
| CoNLL-2003 | via Reuters | Not cleared | Likely high | Exclude |

## Human must check personally
1. Reuters RCV1 agreements (NIST) for commercial use; likely blocks CoNLL.
2. X terms wording and date, and whether "foundation or frontier model" covers a LoRA on a 4B model.
3. BTC's "2015 guidelines" text (not verified by me).
4. Whether CC BY-SA 4.0 share-alike reaches a LoRA trained on Few-NERD.
5. BillSum's original licence with FiscalNote/GovInfo.
6. Generator terms of Self-OSS-Instruct's StarCoder2 outputs.
7. Dolma 3 / Phi / Qwen contamination: no index reachable here.
