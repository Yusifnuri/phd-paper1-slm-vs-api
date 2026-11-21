# Research foundation

Status: assembled 2026-11-22 from `agents/01..06` and `agents/06b` (06b is an integrator consistency check, not an independent review; see `NOVELTY_AND_RED_TEAM.md`). Later rounds supersede earlier ones (A1 Round 2, A3 Round 2b, A4 Round 2, A5 Round 2, `DECISION_LOG.md` L-01..L-16 and later). **Nothing here is a result.** No confirmatory run exists; the pilot in `results/pilot-2026-11-21/` is exploratory (old prompt, fixed pool, under-trained at small N) and is never pooled with confirmatory data. Labels: **[V]** verified from a primary source or by computation, **[I]** interpretation, **[R]** recommendation, **[U]** unverified assumption; confidence H/M/L where given. Detailed evidence, URLs, quotes and access dates live in the agent files cited in brackets.

## 1. Research question and positioning

**Question (decision-grade form, L-01).** For a practitioner who holds N labelled examples, when is a LoRA-tuned small open model (Qwen3-4B-Instruct-2507, Phi-4-mini-instruct; OLMo-3-7B only if time remains) worth training, compared with retrieval-augmented in-context learning on a cheap commercial API (Claude Haiku 5.5, GPT-6 Luna, Gemini 3.1 Flash-Lite), local in-context learning, and a kNN baseline? Accuracy and total cost (tuning charged) are compared at the same labelled pool, on BANKING77 and BillSum (US) as confirmatory tasks and LEDGAR as an exploratory task, for N ∈ {200, 1,000, 3,000}.

**What the paper is not.** It is not a "first equal-pool crossover" study. The classification crossover at tens to hundreds of labels is already reported (Pecher et al. 2025; Spiegelhalter, Franke, Hutter 2026 for Qwen3 models up to 200 labels), and retrieval-ICL vs LoRA on BANKING77/CLINC150 was studied by Bertsch et al. 2025 [`01-literature.md` §1–2, R2.1; V, H except Spiegelhalter, abstract and public code only]. A mixed outcome (kNN or API retrieval-ICL matches the fine-tuned model on BANKING77; fine-tuning wins mainly at larger N or on summarisation) is a plausible and publishable result [I].

## 2. What is known and what the study adds

| Topic | Known (source) | Left open |
|---|---|---|
| Crossover fine-tuning vs ICL | ~50 labels for Qwen3 on 17 narrow tasks (Spiegelhalter 2026, abstract); ~100 labels (Pecher 2025); ICL ≥ LoRA at small N (Bertsch 2025) | Crossover against *commercial APIs* with retrieval from the same pool, N up to 3,000, long-document summarisation |
| Cost | Setup + per-query cost model with measured throughput (Spiegelhalter 2026, repository); $ per report and break-even volume, one task, GPT-4o (Mansuri 2026); FLOPs only (Pecher 2025) | Cost per correct output across APIs, with the **tuning sweep charged** in the break-even, over a GPU-rate band and utilisation |
| Retrieval ICL | Retrieval gain falls from 51.5 to 4.9 points at 1,500-shot on Banking-77 (Bertsch 2025 §3.2) | API arm with retrieval from the equal pool |
| Summarisation | Only an N-sweep with tiny test set, XSum, ROUGE-1 (ELearnFit 2405.02710); Pecher advises prompting for generation but did not test it | A BillSum label-budget curve with a validated metric |
| Prior flaws | Checkpoint chosen on test (Bertsch App. E [V]); 250-item tests; single-dataset tuning; constrained decoding / classification head | Equal, stated tuning effort; frozen test; generated-label ("deployed generative system") protocol |

**Gap statement (supersedes A1 R2.4; N grid corrected to L-02).** Existing work locates the fine-tuning/ICL crossover at tens to hundreds of labels for classification and narrow tasks and shows retrieval ICL competing with LoRA on open models; dollar-based break-evens exist for single tasks. In the sources I could read, none compares tuned small open models with retrieval-augmented cheap commercial APIs on the same labelled pool for N from 200 to 3,000, includes a long-document summarisation curve, and charges the tuning cost in the break-even. This is absence in a limited search (about 30 queries over arXiv, OpenAlex, WebSearch, GitHub; Semantic Scholar, OpenReview and ACL Anthology were not searchable), not proof; a manual check is required again before submission (A1-04).

**Allowed claims:** the conjunction above; paired, replicated gaps with CIs at the pre-registered cells; the bracket of the crossover; break-even Q\* with sensitivity bands; "where fine-tuning does not win". **Forbidden:** "first", "novel combination", any NER or finance claim, any claim resting on the unread Spiegelhalter paper text, any "equivalent / no difference" wording for an inconclusive contrast, summarisation quality claims beyond ROUGE-L/BERTScore unless the human validation study is done.

## 3. Design in one page (details in `EXPERIMENT_PROTOCOL.md`)

| Element | Decision |
|---|---|
| Tasks | BANKING77 (3,080 test items) and BillSum US (1,000 seeded test items) confirmatory; LEDGAR (LexGLUE 100 classes, 3,000 seeded test items, k_max 8) exploratory; SAMSum and CLINC150 not run (L-07; CLINC150 licence is fine but it adds nothing the budget supports) |
| N | 200, 1,000, 3,000 confirmatory; N=77 (BANKING77) / 100 (LEDGAR) one-per-class exploratory for ICL and kNN only; no fine-tuning below 200; N=5,000 and N=50 dropped (L-02) |
| Arms | LoRA (best validation config); retrieval-ICL and random-k ICL on three APIs and on the local models; zero-shot; kNN (first-class arm) |
| Pools | Random for N ≥ 1,000; class-stratified only at N=200 (named, separate condition); random-vs-stratified table at N=200, R=3 (L-04) |
| Tuning | 4 trials per system on replicate 0, then frozen (L-05); validation size min(1,000, N); in-run early stopping (code change A5-12); ICL: 4 values of k plus retriever |
| Replicates | BANKING77 R=8, BillSum R=5, no confirmatory cell below 5, replicate-major order, blinded variance re-estimation after R=5, extension to ≤10 (L-06) |
| Confirmatory family | 2 tasks × 3 N = 6 contrasts, best-validation fine-tuned vs best-validation ICL of the best API, Holm; everything else exploratory (L-03, L-15) |
| Gate | ≈3.7–5.2 GPU-h pilot with written pass criteria G1–G7 before any campaign (L-09); go/no-go at the end of week 2 with a cut ladder (L-16) |
| Release | Adapters for BANKING77 and BillSum only; LEDGAR ids, labels and metrics only; API predictions with a no-training notice; paid API tiers only (L-08) |

## 4. Evidence status by research question

| RQ | Status | Needed before a claim |
|---|---|---|
| RQ1 data efficiency / bracket | Design fixed; fine-tuned arm unproven under the corrected prompt | F3 pilot gates G1–G7; confirmatory runs |
| RQ2 task dependence (summarisation) | Design fixed; metric validity limited | ROUGE-L/BERTScore only unless the 60-bill human study (H2) is done |
| RQ3 economic efficiency | Prices read 2026-11-21; GPU rate is a $3.6–6.0/h scenario band (no NVL price exists); throughput is FLOP arithmetic | 1-h throughput benchmark; live API usage fields |
| RQ4 break-even | Method fixed (Q\* with Fieller / "never" regions); central values in `FEASIBILITY_AND_BUDGET.md` are sensitivity, not results | measured token counts and train times |
| RQ5 generalisation | Two classification sets were cut to one (BANKING77) plus LEDGAR exploratory; BillSum | Scope statement: one classification benchmark, one summarisation benchmark, one GPU type |

## 5. Source index

Primary: Bertsch et al. 2405.00200; Pecher et al. 2402.12819v4; Mansuri et al. 2609.40236; Mosbach et al. 2305.16938; Spiegelhalter, Franke, Hutter (neurips.cc listings 170044/160515, repository `us58/prompting-vs-finetuning-slm`, OpenReview XFSEKRwBKO unread); ELearnFit 2405.02710 (read from PDF). Datasets, licences, provider terms: `DATASET_AUDIT.md`. Summarisation-metric literature (Goyal 2209.12356, Bhandari 2010.07100, SummaC 2111.09525, AlignScore 2305.16739, LongEval 2301.13298, others): `STATISTICAL_ANALYSIS_PLAN.md` §6. Several sources were read through fetch agents and are partly model-summarised (A1-05): verify numbers against PDFs before citing.

## 6. Venue plan (L-11, H5)

Preprint, then a workshop paper about eight weeks in, then TMLR or an applied journal with the extended paper. A journal decision within two months is not realistic [I; review times partly U]. ACL 2027 ARR deadline reported as 4 Jan 2027 comes from a non-official page [V, L]; too early for confirmatory results.
