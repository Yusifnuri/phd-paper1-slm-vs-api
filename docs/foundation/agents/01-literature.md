# A1: Literature and novelty (accessed 2026-11-21)

Labels: **[V]** verified fact (read in the full text), **[I]** author's interpretation, **[R]** my recommendation, **[U]** unverified. Confidence H/M/L. Locations are sections, tables or figures; the PDFs have no stable page numbers.

## 0. What was and was not read

| Paper | Read | Not read |
|---|---|---|
| Bertsch 2405.00200 | PDF pp. 1-8, 13-28: body, Tables 1-6, App. A-G | Models appendix H |
| Pecher 2402.12819v4 | PDF pp. 1-18: body, limitations, App. A-C | App. D |
| Mansuri 2609.40236 | PDF pp. 1-14: body, all tables, limitations | Supplement S1-S2 (not in the PDF) |
| Mosbach 2305.16938 | PDF §3-4 results, §6-8; §1-4.2 verbatim via a fetch agent | App. A-D, §9 (**[U]**) |
| Spiegelhalter et al. | **Abstract only** (neurips.cc listings 170044, 160515). OpenReview refused automated access | Full text. **Treat every claim below as abstract-level** |
| Valdes Gonzalez 2602.06370, Trad & Chehab 2512.04106, Bohnet 2511.00130 | HTML bodies, read by fetch agents (partly model-summarised) | Figures, later sections |

## 1. Extracted methods (four required papers)

**Bertsch et al., NAACL 2025.** https://arxiv.org/abs/2405.00200
- [V,H] Data (Table 1): TREC, TREC-fine, NLU, Banking-77 (train 10,003), Clinic-150 (15,250), SAMSum (14,732). Models: Llama-2-7B (4k/32k/80k), Mistral-7B-v0.2, Qwen2.5-7B (§2).
- [V,H] Test: "we subsample 250 examples from the test set of each dataset" (§2). Accuracy and macro-F1 for classification; SAMSum scored with BERTScore, with ROUGE as a check (§2).
- [V,H] Decoding: constrained, by adding a large constant to valid-label logits (App. D). Without it Banking-77 accuracy is about 14% against about 88% (Fig. 14). The authors say it is "generally not possible for API-access models".
- [V,H] Fine-tuning uses a classification head, not generation (§2). Llama-2-7B, full and LoRA, 30 epochs, r=8, alpha=32, lr 1e-3, batch 32 (App. E).
- [V,H] **Checkpoints are picked on the test set**: "choosing the checkpoint with the highest test accuracy" (App. E). The authors say this favours fine-tuning. Hyperparameters were tuned on Banking-77 only (App. E). Fine-tuning seeds: 5 (Fig. 15) or 3 (Fig. 16). ICL: 10 random shuffles (§3.1).
- [V,H] SAMSum appears only in the ICL table (Table 2, "up to at least 250-shot"). I saw no SAMSum fine-tuning curve in §3.3 or App. C. Retrieval sometimes underperforms random selection there (§3.2).
- [V,H] Two frontier models on Clinic-150 only: Claude 3.5 Sonnet and Llama 3.1 405B. They ran without constrained decoding and "use only a single run" (§4, Fig. 6). No cost analysis beyond qualitative remarks (§3.3, fn. 6).
- [V,H] Findings: Banking-77 retrieval gain falls from 51.5 to 4.9 points at 1500-shot (§3.2). "ICL generally outperforms LoRA finetuning on the same model" at small N. LoRA mostly never exceeds long-context ICL, while full fine-tuning sometimes does (§3.3). Limitation: "We focus primarily on classification tasks" (§8).

**Pecher, Srba, Bielikova, EMNLP 2025.** https://arxiv.org/abs/2402.12819
- [V,H] 8 classification sets (SST2, MRPC, CoLA, BoolQ, AG News, TREC, SNIPS, DBPedia), 60/20/20 splits, a random 1,000-item test subset (§3, App. A.1, Limitations).
- [V,H] Fine-tuned models are BERT and RoBERTa (full fine-tuning). Instruction-tuning is full for Flan-T5 and LoRA for Mistral-7B and Zephyr-7B. General models: Flan-T5, LLaMA-2-13B, LLaMA-3-8B, Mistral, Zephyr and **"GPT-4 (4o-mini-2024-07-18 version)"** (§3). So a cheap commercial API *is* included, for prompting and ICL only.
- [V,H] ICL uses "as many samples per class as the context-size" allows, chosen randomly. There is no retrieval and no label-budget curve for ICL. Zero-shot has no N. Same prompt for all models.
- [V,H] Runs: "repeated 100 times for fine-tuning and 20 times for the remaining approaches"; GPT-4o-mini only 10 times. One fixed seed per repeat, shared by all models. Second break-even uses mean ± SD (§3).
- [V,H] Cost is FLOPs only (App. A.3, Table 3). There are no dollars and no break-even volume. Test-set contamination is acknowledged (§3.1, Limitations).
- [V,H] Authors' advice: prompting is preferable when the task "requires some kind of generation (e.g., translation, summarisation)" (§4). Generation was not tested. That is an untested claim our summarisation arm can test.

**Mansuri et al., arXiv Sept 2026 (preprint, no venue).** https://arxiv.org/abs/2609.40236
- [V,H] **The body does contain a cost and break-even analysis** (Results, "Training time and cost", Tables 3-4). The earlier register note that it has none was right only for the abstract.
- [V,H] Task: multi-label ICH acuity classification, not span extraction. Gemma-3-12B, 4-bit LoRA, on one L40S. Design: classification head vs generative IFT, crossed with distilled vs synthetic data, n = 200, 500, 1000, 1500, 2000 (Fig. 1). Training samples drawn with "a fixed seed (42)". Test: 100 reports, with only 5 and 6 positives for two classes ("Evaluation used 100 test reports from a single institution").
- [V,H] Baselines: GPT-4o zero-shot and untuned Gemma. No ICL or retrieval arm. Cost: GPT-4o $0.0024/report (standard) vs L40S $0.00016/report. Break-even is "roughly 900 reports/year" (one-time $6, 3-year amortisation) and about 10,000/year if the whole 95-GPU-hour sweep is charged (Table 4).
- [V,H] Statistics: paired bootstrap (1,000) and McNemar. One seed per cell.

**Mosbach et al., Findings ACL 2023.** https://arxiv.org/abs/2305.16938
- [V,H] OPT 125M-30B. MNLI, RTE, QQP in domain; HANS and PAWS-QQP out of domain. n = 2-128 examples, ICL mainly at 16. "we sample 10 different sets of examples for each n", 3 patterns, so 30 runs per cell. Fine-tuning: 40 epochs, lr 1e-5. Accuracy. Welch's t-test. No retrieval, API or cost. Findings: matched-size fine-tuning is comparable to or better than ICL in and out of domain (§4, §8).

## 2. Search log and further close work

**Search.** About 9 queries with the standard WebSearch tool (US-only), plus the 2026-11-21 verification report and arXiv/neurips.cc fetches. I queried no ACL Anthology, Semantic Scholar or OpenReview search directly. Terms: "fine-tuning vs in-context learning labeled examples summarization LoRA break-even"; "cost per correct output self-hosted fine-tuned vs GPT API break-even"; "retrieval-augmented ICL vs LoRA same training pool"; "BillSum fine-tuned small LLM GPT-4o-mini"; "many-shot ICL vs fine-tuning Banking77 LoRA"; ARR/TMLR scope queries. [I] This is far from exhaustive and cannot support "first".

**Closest new find: Spiegelhalter, Franke, Hutter**, "When Does Fine-Tuning Overtake Prompting? An Accuracy and Cost Analysis of Small Language Models on Narrow Tasks" (NeurIPS 2026 workshop poster, OnDevice Intelligence; a variant is in the LIGHT workshop). [V,M, abstract only] 17 tasks, Qwen3 0.6B/1.7B/4B, three ICL selection strategies, LoRA with per-task hyperparameter optimisation, GEPA, budgets "up to 200 labeled samples". Quote: "approximately 50 labeled samples suffice for fine-tuning to overtake all ICL strategies". It also has a "throughput-calibrated two-component cost analysis". [I] This covers equal-pool crossover plus cost with Qwen3-4B, our main model. [U] Whether it includes summarisation, BANKING77, commercial APIs, or N above 200.

| Paper (venue/year; URL) | Data | Models | Methods | N | Metric | Cost? | Finding / limitation | Overlap and gap for us |
|---|---|---|---|---|---|---|---|---|
| Bertsch, NAACL'25; arXiv 2405.00200 | TREC(-fine), NLU, Banking-77, Clinic-150, SAMSum | Llama-2-7B, Mistral-7B, Qwen2.5-7B; Claude 3.5 / Llama 405B (1 run) | Random/BM25 ICL, full+LoRA FT (head) | ICL to ~2k; FT 32 to ~16k | Acc, BERTScore | No | ICL beats LoRA at small N. Test-set checkpointing, 250-item test, classification focus | **Direct overlap on BANKING77, CLINC, retrieval vs FT.** Gap: generative FT, API arm, cost, long-document summarisation, equal tuning |
| Pecher, EMNLP'25; 2402.12819 | 8 classification sets | BERT, RoBERTa, Flan-T5, Llama-2/3, Mistral, Zephyr, GPT-4o-mini | FT, instruction-tuning (LoRA), prompting, random ICL | 10 to full | Macro-F1 ±SD | FLOPs only | Break-even ~100 labels, variance +100-200%. English classification only | Overlap: crossover with variance. Gap: retrieval ICL, LoRA SLM, summarisation, dollars |
| Mansuri, arXiv'26; 2609.40236 | ICH reports, 100 test | Gemma-3-12B, GPT-4o | LoRA head vs generative, distil vs synthetic | 200-2000 | Macro-F1 | **Yes**, $/report, break-even volume | Distilled FT ≈ GPT-4o. 1 seed, 100 test items, no ICL arm | Overlap: pricing method. Gap: ICL, multi-task, multi-seed |
| Mosbach, Findings ACL'23; 2305.16938 | MNLI, RTE, QQP (+OOD) | OPT 125M-30B | PBFT vs ICL | 2-128 | Acc | No | Matched FT ≥ ICL | Methodology only |
| Spiegelhalter et al., NeurIPS'26 workshops; OpenReview XFSEKRwBKO **[abstract only]** | 17 narrow tasks | Qwen3 0.6/1.7/4B | 3 ICL strategies, LoRA+HPO, GEPA | ≤200 | n/a | **Yes**, throughput-calibrated | Crossover ~50 labels | **Main scoop risk**; read before submission |
| Bohnet, arXiv'25; 2511.00130 | 13 tasks, no summarisation | Gemma 3 | SFT, LoRA, ICL | ICL 1-128; LoRA 8-8192 | Acc | No | Single run per config, one model | Context only |
| Valdes Gonzalez, arXiv'26; 2602.06370 | IMDB, SST-2, AG News, DBPedia | BERT-family; GPT-4o, Claude 4.5 | FT encoders vs prompting | full data | F1 | **Yes** ($/1M requests; no training cost) | Encoders cheaper; no label curve | Cost framing. Gap: N axis, break-even |
| Trad & Chehab, arXiv'25; 2512.04106 | CWE labels, size unstated | Gemini-1.5-Flash, DistilBERT, CodeBERT | Retrieval few-shot vs FT | k 1-20; FT 2,000 | Micro-F1 | Training $ only | Single run, unequal budgets | Retrieval arm precedent |

Other claimed items I did **not** read in full: ELearnFit (2405.02710, news summarisation, LoRA, N sweep; abstract/search snippet only), Valdes/Mathur/Luccioni (see verification 03). **[U]**

## 3. RQ analysis

| RQ | Known | Unknown | Why it matters | Can our design answer it? |
|---|---|---|---|---|
| RQ1 data efficiency | Classification crossover at tens to ~100 labels (Pecher; Spiegelhalter abstract). ICL beats LoRA at small N (Bertsch) | Crossover versus cheap APIs with retrieval from the same pool, for generative FT, N up to 5,000 | The practical label budget | Yes, if arms are equal-pool, with equal tuning, ≥5 replicates (I-01, I-02, I-04) |
| RQ2 task dependence | Pecher advises prompting for generation, untested. Bertsch has SAMSum ICL only | N-curve for long-document summarisation with FT vs ICL vs API. No BillSum N-curve found (limited search) | Prior curves are classification-only | Partly. One summarisation task, ROUGE/BERTScore validity needs human check (I-13) |
| RQ3 economic efficiency | Mansuri: $/report, one task, GPT-4o. Valdes: $/1M requests. Spiegelhalter: cost model [abstract] | Cost per correct output across tasks and several cheap APIs with measured throughput | Decisions are made in dollars | Yes, if token counts and GPU rates are measured and ranged |
| RQ4 break-even | Mansuri: ~900/yr, ~10,000/yr with sweep charged. Spiegelhalter says data availability, not volume, drives the choice [abstract] | Break-even volume as a function of N and task | Turns accuracy into an operational rule | Yes. Charge the tuning sweep or the number is misleading |
| RQ5 generalisation | Pecher: variance and contamination vary by dataset | Whether the crossover transfers across tasks and datasets | Avoids one-benchmark conclusions | Only with ≥2 classification sets plus summarisation. Three tasks is thin [I] |

**Direct answer on "minor extension of Bertsch".** [I] The classification half, as planned (BANKING77 and CLINC150, ICL vs fine-tuning, retrieval), would be read as a replication with an API arm: yes, a reviewer could dismiss it. Pecher plus Bertsch cover crossover and retrieval, and Spiegelhalter (once read) covers Qwen3 LoRA and cost for small N. What would change that: (i) the BillSum N-curve; (ii) the API arm with retrieval from the same pool; (iii) cost per correct output with a break-even in which the tuning sweep is charged; (iv) fixing what the prior work leaves open (test-set checkpoint selection, single-dataset hyperparameters, 250-item tests, no tuning symmetry); (v) honest reporting where fine-tuning does not win. "First" claims are not defensible from my search.

## 4. Venues (unverified items marked)

| Venue | Scope and expectations | Fit | Dates / limits |
|---|---|---|---|
| TMLR (jmlr.org/tmlr) | Acceptance rests on whether claims are "supported by accurate, convincing and clear evidence"; modest contributions are acceptable [V,M]. Rolling, journal-style, code encouraged | **Best fit for a journal target and honest empirical work** | Rolling. No hard page limit; over 12 pages main text loses the short review timeline [V,M]. Review typically takes months [U] |
| ACL / EMNLP via ARR | Empirical NLP, reproducibility checklist [U] | Competitive only with a clear novelty beyond Bertsch/Pecher | ACL 2027 ARR deadline reported as 4 Jan 2027 (LINGUIST List, not the official page) [V,L]. Too early for full experiments. NAACL 2027 (ARR 12 Oct 2026) and EACL 2027 (ARR 3 Aug 2026) have passed [V,M]. Long paper 8 pages [U] |
| NeurIPS-type workshops (e.g. the small-model workshops where Spiegelhalter appeared) | Short, preliminary results welcome | Realistic for a first outlet or preprint | Per-workshop; check yearly [U] |
| Applied-NLP journals (Natural Language Engineering, Information Processing & Management, Engineering Applications of AI) | Applied empirical comparisons, usually long format | Plausible. I could not verify current scope pages | Rolling; review 3-6 months typical [U] |
| Computational Linguistics, JAIR | Archival, longer; novelty bar higher | Unlikely for an incremental empirical study [I] | [U] |

[R] A strict two-month budget fits: (1) a preprint plus a workshop submission after about 8 weeks, then (2) TMLR or an applied journal with the extended paper. No venue promise is implied. A journal decision inside two months is not realistic.

## 5. New issues

| ID | Issue | Severity |
|---|---|---|
| A1-01 | Spiegelhalter et al. (Qwen3-4B, LoRA, cost, crossover) not read beyond the abstract. May pre-empt RQ1/RQ3. Get the PDF through a signed-in browser or the authors | HIGH |
| A1-02 | Earlier notes say Pecher has "no API LLMs" and Mansuri "no cost figure". Both wrong: Pecher includes GPT-4o-mini, Mansuri's body has a cost and break-even model. Correct `00-summary.md` and `03` | HIGH |
| A1-03 | Bertsch numbers are not comparable to ours: test-set checkpointing, 250-item test, constrained decoding, head-based FT. Use them qualitatively, never as a reproduction target | MEDIUM |
| A1-04 | "To our knowledge" must be backed by an updated search of ACL Anthology, Semantic Scholar, OpenReview directly, run again just before submission | MEDIUM |
| A1-05 | Several fetched sources were model-summarised in part (Valdes, Trad, Bohnet, Mosbach beyond §4); verify against PDFs before citing numbers | MEDIUM |
| A1-06 | Two-month journal target conflicts with review times; plan preprint plus workshop first | MEDIUM |
| A1-07 | Mansuri's cost method (3-year amortisation, 4,000 h/year, vendor throughput) is an assumption set, not a measurement. Our break-even must show sensitivity ranges | LOW |

## 6. Register issues

- **I-10 (CRITICAL).** Keep OPEN. Overlap is real (Bertsch, Pecher), plus new A1-01. It can drop to HIGH once Spiegelhalter is read and the claim is cut to the conjunction below.
- **I-16 (MEDIUM, agree).** Bertsch avoids invalid labels by design (head for FT, logit constraint for ICL), and without constraint Banking-77 collapses (Fig. 14). Our generated labels, with an invalid-output rate, are a different system. [R] State this, use the same label list and a shared normaliser in all arms, report invalid-output rate separately, and add a sensitivity analysis by scoring over the closed label set.
- **I-17 (MEDIUM, agree).** Pecher itself flags that datasets may have been in pretraining (§3.1, Limitations). BANKING77 and CLINC150 have been public since 2019-20. [R] Disclose, and treat the absolute scores as benchmark-specific.

## 7. Gap statement and contribution wording ([R], to be re-checked after A1-01 and A1-04)

*Gap.* Prior work finds fine-tuned small models overtaking prompted or in-context models after tens to hundreds of labels on classification (Pecher et al. 2025; Spiegelhalter et al. 2026, abstract), and that long-context ICL competes with LoRA on open models (Bertsch et al. 2025). Dollar-based self-hosted-vs-API break-evens exist for single tasks (Mansuri et al. 2026). In the sources I read, no study combines, on one equal labelled pool, retrieval-augmented ICL on cheap commercial APIs, PEFT-tuned small open models, and a long-document summarisation task with a label-budget curve and a break-even that charges the tuning cost.

*Contribution wording.* "We measure, for BANKING77, a second classification set and BillSum, how many labelled examples a LoRA-tuned 4B model needs to match retrieval-augmented, cheap commercial APIs trained or prompted from the same pool, and at what query volume self-hosting pays back when tuning is charged. Results are paired, replicated and priced from measured tokens and GPU time, and we report where fine-tuning does not win."

Do not use "first", "novel combination", or any NER or finance claim.
