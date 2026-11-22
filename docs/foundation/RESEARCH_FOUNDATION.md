# Research foundation

Status: assembled 2026-11-22 from `agents/01..06` and `agents/06b` (06b is an integrator consistency check, not an independent review; see `NOVELTY_AND_RED_TEAM.md`). Later rounds supersede earlier ones (A1 Round 2, A3 Round 2b, A4 Round 2, A5 Round 2, `DECISION_LOG.md` L-01..L-32). **Updated 2026-11-22 (second pass)** for the Spiegelhalter full-text reading (L-21), the overlap and terms check (L-31), `PREREG_DECISIONS.md` (D-01..D-21), the API-arm funding answer (L-32) and the pilot runs. **Nothing here is a result.** No confirmatory run exists. The pilots (`results/pilot-2026-11-21/` with the old prompt, fixed pool and under-trained small N; and the exploratory validation-only runs of 2026-11-22 in `results/pilot-stage2-2026-11-22/`, `pilot-stage2b-2026-11-22/`, `pilot-session3-2026-11-22/`, summarised in `docs/PREREGISTRATION.md` s.16) are never pooled with confirmatory data and are not evidence for any hypothesis. Labels: **[V]** verified from a primary source or by computation, **[I]** interpretation, **[R]** recommendation, **[U]** unverified assumption; confidence H/M/L where given. Detailed evidence, URLs, quotes and access dates live in the agent files cited in brackets.

## 1. Research question and positioning

**Question (decision-grade form, L-01).** For a practitioner who holds N labelled examples, when is a LoRA-tuned small open model (Qwen3-4B-Instruct-2507, Phi-4-mini-instruct; OLMo-3-7B only if time remains) worth training, compared with retrieval-augmented in-context learning on a cheap commercial API (Claude Haiku 5.5, GPT-6 Luna, Gemini 3.1 Flash-Lite), local in-context learning, and a kNN baseline? Accuracy and total cost (tuning charged) are compared at the same labelled pool, on BANKING77 and BillSum (US) as confirmatory tasks and LEDGAR as an exploratory task, for N ∈ {200, 1,000, 3,000}.

**What the paper is not.** It is not a "first equal-pool crossover" study. The classification crossover at tens to hundreds of labels is already reported (Pecher et al. 2025; Spiegelhalter, Franke, Hutter 2026 for Qwen3 models up to 200 labels), and retrieval-ICL vs LoRA on BANKING77/CLINC150 was studied by Bertsch et al. 2025 [`01-literature.md` §1–2, R2.1; V, H; Spiegelhalter now read in full, `docs/research/spiegelhalter-2026-notes.md`, L-21]. The closest earlier BANKING77 + retrieved-examples + commercial-API + cost paper is arXiv 2311.06102 (ACM ICAIF 2023; encoder fine-tuning, 2023 models, no summarisation, no replicates), to be cited as the nearest predecessor (L-31, `docs/research/2026-11-22-overlap-and-terms-check.md`). A mixed outcome (kNN or API retrieval-ICL matches the fine-tuned model on BANKING77; fine-tuning wins mainly at larger N or on summarisation) is a plausible and publishable result [I].

## 2. What is known and what the study adds

| Topic | Known (source) | Left open |
|---|---|---|
| Crossover fine-tuning vs ICL | ~50 labels for Qwen3 on 17 narrow tasks, 14 of them synthetic, N up to 200, 300-item tests, two seeds, tuning budgets explicitly not matched, a NeurIPS 2026 workshop poster (Spiegelhalter 2026, full text; they name "LoRA saturation beyond 200 samples" as open); ~100 labels (Pecher 2025); ICL ≥ LoRA at small N (Bertsch 2025) | Crossover against *commercial APIs* with retrieval from the same pool, N up to 3,000, long-document summarisation |
| Cost | Setup + per-query cost model with measured throughput, LoRA setup cost including the HPO GPU-hours, RAG excluded from the cost analysis, no commercial API arm (Spiegelhalter 2026); $ per report and break-even volume incl. a whole-sweep-charged variant (about 10,000 reports/year), one task, GPT-4o (Mansuri 2026); retrieval few-shot Gemini-1.5-Flash vs fine-tuned encoders, single run (Trad and Chehab 2025); FLOPs only (Pecher 2025) | Break-even against *retrieval-augmented* commercial APIs from the same pool, across tasks and N, over a GPU-rate band and utilisation (charging the tuning sweep is **not new**: Mansuri Table 4 does it for one task) |
| Retrieval ICL | Retrieval gain falls from 51.5 to 4.9 points at 1,500-shot on Banking-77 (Bertsch 2025 §3.2) | API arm with retrieval from the equal pool |
| Summarisation | Only an N-sweep with tiny test set, XSum, ROUGE-1 (ELearnFit 2405.02710); Pecher advises prompting for generation but did not test it | A BillSum label-budget curve with a validated metric |
| Prior flaws | Checkpoint chosen on test (Bertsch App. E [V]); 250-item tests; single-dataset tuning; constrained decoding / classification head | Equal, stated tuning effort; frozen test; generated-label ("deployed generative system") protocol |

**Gap statement (supersedes A1 R2.4; N grid corrected to L-02).** Existing work locates the fine-tuning/ICL crossover at tens to hundreds of labels for classification and narrow tasks and shows retrieval ICL competing with LoRA on open models; dollar-based break-evens, including one that charges the whole tuning sweep, exist for single tasks (Mansuri 2026), and retrieval few-shot with a commercial API has been compared with fine-tuned encoders (Trad and Chehab 2025). In the sources I could read, none compares tuned small open models with retrieval-augmented cheap commercial APIs on the same labelled pool for N from 200 to 3,000, none includes a long-document summarisation curve, and none combines that with replicated pools and a kNN baseline. Charging the tuning cost is **not** part of the gap: Mansuri 2026 does it for one task and Spiegelhalter 2026 includes HPO in LoRA's setup cost; Trad and Chehab 2025 and arXiv 2311.06102 are the nearest retrieval-API-vs-fine-tuning precedents. This is absence in a limited search (about 30 queries over arXiv, OpenAlex, WebSearch, GitHub; Semantic Scholar, OpenReview and ACL Anthology were not searchable), not proof; a manual check is required again before submission (A1-04).

**Allowed claims:** the conjunction above; paired, replicated gaps with CIs at the pre-registered cells; the bracket of the crossover; break-even Q\* with sensitivity bands; "where fine-tuning does not win". **Forbidden:** "first", "novel combination", any NER or finance claim, any "equivalent / no difference" wording for an inconclusive contrast, summarisation quality claims beyond ROUGE-L/BERTScore unless the human validation study is done.

**What the study claims with and without the commercial API arm (L-32; the researcher has not chosen; `PREREGISTRATION.md` s.2.1).**

| | With the API arm (variant A) | Without it (variant B) |
|---|---|---|
| Survives | the gap statement above, as "absent in a limited search"; paired replicated gaps against retrieval-augmented commercial APIs at N = 200 / 1,000 / 3,000; a measured price-and-accuracy break-even | paired replicated gaps between tuned LoRA, retrieval ICL and kNN on one open model (Qwen3-4B-Instruct-2507) at N = 200 / 1,000 / 3,000; a BillSum curve once ROUGE-L evaluation exists; GPU-hours and imputed cost of the local arms; the replicated-pool, crossed-model design and the stated tuning effort |
| Lost | nothing | every statement about commercial APIs and the API cost side of the break-even; with them the main novelty gap (L-13, L-32) |
| Novelty | the conjunction only; I-10 stays HIGH under L-13, whose condition is delivery of the API arm | an incremental extension: retrieval ICL vs LoRA on BANKING77 (Bertsch 2025) and the Qwen3 crossover to N = 200 (Spiegelhalter 2026) already exist; what is added is N up to 3,000, summarisation, replication and kNN. The condition of L-13 is not met, so I-10's severity needs the integrator's re-assessment (`PC-29`) |
| Never claimed | "first", "novel combination", any commercial-API statement from the pilot | the same, plus any commercial-API statement at all |

## 3. Design in one page (details in `EXPERIMENT_PROTOCOL.md`)

| Element | Decision |
|---|---|
| Tasks | BANKING77 (3,080 test items) and BillSum US (1,000 seeded test items) confirmatory; LEDGAR (LexGLUE 100 classes, 3,000 seeded test items, k_max 8) exploratory; SAMSum and CLINC150 not run (L-07; CLINC150 licence is fine but it adds nothing the budget supports) |
| N | 200, 1,000, 3,000 confirmatory; N=77 (BANKING77) / 100 (LEDGAR) one-per-class exploratory for ICL and kNN only; no fine-tuning below 200; N=5,000 and N=50 dropped (L-02) |
| Arms | LoRA of Qwen3-4B-Instruct-2507 (best validation config; Phi-4-mini exploratory, D-05); retrieval-ICL and random-k ICL on three APIs (only if the API arm is funded, L-32) and on the local models; zero-shot; kNN (first-class arm) |
| Pools | Random for N ≥ 1,000; class-stratified only at N=200 (named, separate condition); random-vs-stratified table at N=200, R=3 (L-04) |
| Tuning | 5 trials per system on replicate 0, then frozen (L-25, D-06; L-05 said 4): fine-tuning five learning rates {5e-5 ... 8e-4}, ICL five k {4, ..., 64} on BANKING77 (BillSum {1, 2, 3} plus retriever); validation size min(1,000, N) on every arm (D-02); in-run early stopping (done) |
| Replicates | BANKING77 R=8, BillSum R=5, no confirmatory cell below 5, replicate-major order, blinded variance re-estimation after R=5, extension to ≤10 (L-06) |
| Confirmatory family | 2 tasks × 3 N = 6 contrasts, best-validation fine-tuned vs best-validation ICL of the best API, Holm; everything else exploratory (L-03, L-15) |
| Gate | pilot with written pass criteria G1–G7 before any campaign (L-09; planned about 3.7–5.2 GPU-h); stage 2 has run twice (G5 failed, then G4 failed narrowly), stages 1, 3 and A have not, so the pilot is **not passed**; go/no-go with a cut ladder (L-16) is provisional |
| Release | Adapters for BANKING77 and BillSum only; LEDGAR ids, labels and metrics only; for API arms metrics, per-item correctness flags and ids, not raw Gemini text, no API output in any training pipeline, paid tiers only (L-08, L-31); nothing is released before the researcher's terms and licence check (D-21) |

## 4. Evidence status by research question

| RQ | Status | Needed before a claim |
|---|---|---|
| RQ1 data efficiency / bracket | Design fixed; fine-tuned Qwen arm trains and tunes sensibly on BANKING77 in exploratory validation runs, but the pilot is not passed (G4 failed, stages 1, 3, A not run) | pilot decisions (`PC-30`, `PC-31`); confirmatory runs; for the API comparison, funding (L-32) |
| RQ2 task dependence (summarisation) | Design fixed; metric validity limited | ROUGE-L/BERTScore only unless the 60-bill human study (HD-2, an annotator; not done by default, D-13) is done; no ROUGE-L evaluation code exists yet |
| RQ3 economic efficiency | Prices read 2026-11-21; GPU rate is a $3.6–6.0/h scenario band (no NVL price exists; the credits carry no dollar value, L-32); throughput is partly measured now (`FEASIBILITY_AND_BUDGET.md` s.3a), the rest FLOP arithmetic; API cost needs the unfunded API arm | 1-h throughput benchmark; live API usage fields |
| RQ4 break-even | Method fixed (Q\* with Fieller / "never" regions); central values in `STATISTICAL_ANALYSIS_PLAN.md` s.7 are sensitivity, not results; without the API arm only a labelled projection | measured token counts and train times; API accuracy |
| RQ5 generalisation | Two classification sets were cut to one (BANKING77) plus LEDGAR exploratory; BillSum | Scope statement: one classification benchmark, one summarisation benchmark, one GPU type |

## 5. Source index

Primary: Bertsch et al. 2405.00200; Pecher et al. 2402.12819v4; Mansuri et al. 2609.40236; Mosbach et al. 2305.16938; Spiegelhalter, Franke, Hutter (neurips.cc listings 170044/160515, repository `us58/prompting-vs-finetuning-slm`, OpenReview XFSEKRwBKO, full text read 2026-11-22 from the PDF the researcher downloaded: `docs/research/spiegelhalter-2026-notes.md`); ELearnFit 2405.02710 (read from PDF). Datasets, licences, provider terms: `DATASET_AUDIT.md`. Summarisation-metric literature (Goyal 2209.12356, Bhandari 2010.07100, SummaC 2111.09525, AlignScore 2305.16739, LongEval 2301.13298, others): `STATISTICAL_ANALYSIS_PLAN.md` §6. Several sources were read through fetch agents and are partly model-summarised (A1-05): verify numbers against PDFs before citing.

## 6. Venue plan (L-11, HD-5)

Preprint, then a workshop paper about eight weeks in, then TMLR or an applied journal with the extended paper. A journal decision within two months is not realistic [I; review times partly U]. ACL 2027 ARR deadline reported as 4 Jan 2027 comes from a non-official page [V, L]; too early for confirmatory results.
