# Pre-registration (DRAFT v2, not frozen, not registered anywhere)

**Status.** Rewritten 2026-11-22 from the foundation documents (`docs/foundation/*.md`, the independent review `agents/06c-independent-redteam.md`, `docs/research/spiegelhalter-2026-notes.md`) and the exploratory pilot runs of 2026-11-22 (stage 2, stage-2 rerun, GPU session 3; section 16). **Second pass, 2026-11-22:** brought into line with `docs/foundation/PREREG_DECISIONS.md` (D-01..D-21, `PD`), `DECISION_LOG.md` L-22..L-32 and the pilot results; section 2.1 (new) states the study with and without the commercial API arm; section 16 is rewritten; section 17 now shows the decision for every item. Contradictions that no decision covers are listed in `PREREG_CONSISTENCY.md` section F (cited as `PC-29` and up). It replaces the draft of 2026-11-21, whose N levels (50, 5,000), 5 replicates, McNemar test, 2 contrasts per task and conditional LEDGAR role are superseded (list in Appendix B).

- **No confirmatory result exists.** No test split has been read by any experiment of this study. The pilots (`results/pilot-2026-11-21/`, `results/pilot-stage2-2026-11-22/`, `results/pilot-stage2b-2026-11-22/`, `results/pilot-session3-2026-11-22/`) used only the 1,000-row validation split carved from the training set. They are exploratory, mostly replicate 0 or two replicates, partly evaluated on the same rows that selected the configuration (I-25, open), and are never pooled with confirmatory data. **The stage-2 gates are not a pass of the full pilot** (section 16).
- **Nothing here has been submitted.** This file is a working draft for the researcher. It becomes a pre-registration only when the researcher (a) confirms or overrules each decision in section 17 and settles the five items that stay with the researcher plus the open contradictions of `PREREG_CONSISTENCY.md` section F, (b) commits it and creates the protocol tag (section 7), and (c) optionally copies it to a registry. Until then every number may change, and changes before the tag need no deviation note.
- **Conventions.** Numbers are copied from the foundation documents with a pointer in square brackets (`STAT` = `STATISTICAL_ANALYSIS_PLAN.md`, `PROT` = `EXPERIMENT_PROTOCOL.md`, `FEAS` = `FEASIBILITY_AND_BUDGET.md`, `DATA` = `DATASET_AUDIT.md`, `FOUND` = `RESEARCH_FOUNDATION.md`, `PD` = `PREREG_DECISIONS.md`, `L-nn` = `DECISION_LOG.md`, `I-nn` = `ISSUE_REGISTER.md`). Where two documents disagree, the choice made here is explained in `docs/foundation/PREREG_CONSISTENCY.md` (cited as `PC-nn`). **Pilot** numbers are marked `[PILOT, validation only]` and carry their replicate count and whether the rows overlap with selection rows. Model ids, prices and API facts were read from provider pages on 2026-11-21 through fetch agents (`FEAS` section 1) and are not independently re-verified here.
- **Naming.** The hypotheses are called H1-H4 here. `HUMAN_DECISIONS.md` also uses H1-H5 for researcher decisions (API keys, annotator, credits, paper access, venue); in this file those are written `HD-1..HD-5` (PC-24).

---

## 1. Study information

| Field | Value |
|---|---|
| Working title | When does a tuned 4B model beat a cheap API with retrieval? Equal-pool accuracy and cost on BANKING77 and BillSum (`PAPER_OUTLINE.md`) |
| Author, affiliation, funding | researcher (D-18, stays open) |
| Draft date / protocol-tag date / arms-tag dates | draft 2026-11-22 / no date set / no dates set. The calendar of `FEAS` s.4 (protocol tag Dec 6-12, arms tag Dec 13-19) predates the two-stage freeze (tag A for fine-tuned, kNN and local ICL arms; tag B for the API arm, D-08, L-30) and the funding answer (L-32: tag B cannot be created now). Dates are re-planned when the researcher decides the API route (`PC-14`, `PC-29`) |
| Code commit and tag | Not yet set (researcher). Pilot code tags: `pilot-stage2-code` (`f533926`, stage 2), `pilot-stage2b-code` (`ce0dd2a`, rerun and session 2), `session3-code` (`12ec11f`, session 3) [`PROGRESS.md`, `HANDOFF.md`]. The protocol tag must point to a commit containing the frozen analysis scripts |
| Registry | None. Whether to copy the frozen file to OSF is the researcher's choice (D-18, stays open) |

## 2. Research questions

Source: `FOUND` section 1 (decision-grade form, L-01) and section 4. The study is a decision-grade comparison for a practitioner who holds N labelled examples. It is **not** a "first crossover" study: the classification crossover at tens to hundreds of labels is already reported (Pecher et al. 2025; Spiegelhalter, Franke, Hutter 2026; Bertsch et al. 2025) [`FOUND` s.1, s.2; `spiegelhalter-2026-notes.md`].

| RQ | Question | Status here |
|---|---|---|
| RQ1 | Data efficiency: at which N in {200, 1,000, 3,000} does a LoRA-tuned small model (Qwen3-4B-Instruct-2507, Phi-4-mini-instruct) overtake retrieval-augmented in-context learning on the best of three cheap commercial APIs, given the same labelled pool? Qwen3-4B-Instruct-2507 is the confirmatory model (D-05); Phi-4-mini-instruct is exploratory. | Confirmatory (H1, H2), variant A only (section 2.1) |
| RQ2 | Task dependence: does the answer differ between intent classification (BANKING77) and long-document summarisation (BillSum, US)? | Confirmatory (H3) |
| RQ3 | Economic efficiency: accuracy and total cost, tuning charged, for each arm at the same pool | Confirmatory for H4 (variant A), otherwise descriptive |
| RQ4 | Break-even query volume Q* against the API, with sensitivity to GPU rate band, utilisation and sync vs batch pricing | H4 (variant A); in variant B only a labelled projection |
| RQ5 | Generalisation | **Not tested.** Scope is one classification benchmark, one summarisation benchmark, two small models, three APIs, one GPU type [`FOUND` s.4] |

Positioning limits (what the paper may and may not claim) are in `FOUND` section 2 and are not repeated here. In particular: no "first" or "novel" claim; no "equivalent", "no difference" or "ties" wording for an inconclusive contrast; no summarisation-quality claim beyond ROUGE-L and BERTScore unless the human validation study is run (section 11).

### 2.1 Two variants of the study: with and without the commercial API arm (L-32)

On 2026-11-22 the researcher answered that there are no API keys and no money (HD-1; `L-32`). The API arm (default 274,704 requests, `L-22`, `analysis/count_requests.py`) is therefore **unfunded**, no live API test and no tag B (D-08) can exist now, and I-09 stays open. The researcher has not chosen a route (provider or student-credit programmes; a minimal single-provider BANKING77 arm, which `HANDOFF.md` sizes at about 28k requests, a figure not reproduced here; or dropping the commercial API arm). The paper is therefore described in two variants. **Which variant the paper becomes is the researcher's decision** (`PC-29`); this section changes no decision.

| | Variant A: commercial API arm funded and run | Variant B: no commercial API arm |
|---|---|---|
| Comparator in the six contrasts | best-validation retrieval-ICL configuration of the selected API (section 6) | not decided. A drafting proposal, **not a decision**: the best-validation retrieval-ICL configuration of the same open model (Qwen3-4B-Instruct-2507). Its local arm exists in code and was tuned on validation in session 3 (section 16), but is listed as exploratory in section 6 and sits in tag A (L-30) |
| H1, H2, H3 | as written (section 3) | not testable as written: all three are stated against the API. They would have to be re-worded against the local comparator before the protocol tag, which needs the researcher's confirmation (D-01 accepted the current wording) |
| H4, RQ3, RQ4 (cost, break-even against the API) | measured accuracy and billed cost per query | cost per correct for the local arms in GPU-hours and imputed dollars only. A break-even against list API prices could be projected from token counts, but without a measured API accuracy the premise of H4 (FT not worse than API-ICL by more than m) cannot be checked: any such figure is an **exploratory projection**, labelled as such |
| What the paper may claim | the conjunction of `FOUND` s.2, still only as "absent in a limited search" and never "first"; paired replicated gaps against a commercial API at N = 200 / 1,000 / 3,000; a price-and-accuracy break-even | paired, replicated gaps between a tuned LoRA model, retrieval ICL and kNN on one open model at N = 200 / 1,000 / 3,000, on BANKING77 and (once a ROUGE-L evaluation exists) BillSum; a bracket of the crossover on that comparison; GPU-hours and imputed cost of the local arms |
| What is lost | nothing relative to the plan | every statement about commercial APIs, and with it the main novelty gap (L-13, L-32). Retrieval ICL vs LoRA on BANKING77 is already studied (Bertsch et al. 2025) and the crossover for Qwen3 up to N = 200 is already reported (Spiegelhalter et al. 2026); the paper would then be an **incremental extension** (N up to 3,000, one more model, a BillSum curve, kNN, stated tuning effort, replicated pools), not a new crossover finding. The condition under which I-10 was lowered from CRITICAL to HIGH (L-13: deliver the commercial-API arm) is not met (`PC-29`) |
| Open items that disappear | none | tag B, API validation runs, D-16, I-09, the provider-selection rule, the 72-hour window, the API budget |

A third, hybrid possibility (a small API arm on a subset of cells) would need its own list of confirmatory cells and is not specified here. In every variant: no "first" or "novel" claim; the pilot numbers of section 16 are not evidence for any hypothesis.

## 3. Hypotheses

Each hypothesis is tied to confirmatory contrasts (section 9) and states what outcome would **refute** it. Directions are the drafting proposal derived from the cited sentences and are not results; D-01 accepts H1-H4 as drafted (they state outcomes that can fail, and a null or reversed result is publishable, L-01); the researcher can still overrule them before the tag. **All four are stated against the commercial API arm and exist only in variant A (section 2.1).** "Contrast" means delta = accuracy(best-validation fine-tuned configuration) minus accuracy(best-validation retrieval-ICL configuration of the selected API) on the test items, conditional on the frozen arms (section 9). Outcome labels (i)/(ii)/(iii) are defined in section 9.4.

| ID | Statement | Cells | Supported if | Fails if |
|---|---|---|---|---|
| H1 | On BANKING77 the tuned small model beats API retrieval-ICL at the larger label budgets. Motivation: Spiegelhalter et al. find LoRA ahead of local ICL up to N=200 and leave N>200 open; Bertsch et al. find retrieval ICL competitive with LoRA on BANKING77 [`FOUND` s.2; notes consequence 4] | B77 N=1,000 and N=3,000 | both contrasts have outcome (i) "FT better" (Holm p < 0.05, delta > 0) | either contrast is (iii) inconclusive, (ii) practically equal, or (i) "ICL better". The words "no difference" are not used for (iii) |
| H2 | At the smallest budget the API with retrieval is not worse than the tuned model on BANKING77 (the crossover lies above N=200 or inside the bracket) | B77 N=200 (class-stratified pools) | outcome is not "FT better": (iii), (ii) or "ICL better" | outcome (i) "FT better" at N=200. N=200 is the weak cell: realised minimal detectable gap 2.7-5.0 points at R=8 [`STAT` s.3], so (iii) is the expected outcome and is reported as a bracket, not as support for H2 beyond "FT not shown better" |
| H3 | Task dependence: on BillSum (ROUGE-L F1) the tuned model beats API retrieval-ICL at N=1,000 and N=3,000. Motivation: "fine-tuning wins mainly at larger N or on summarisation" is a plausible mixed outcome [`FOUND` s.1]; Pecher advises prompting for generation but did not test it [`FOUND` s.2] | BillSum N=1,000 and N=3,000 | both contrasts (i) "FT better" | either contrast is (iii) or (i) "ICL better". BillSum has no equivalence outcome (section 9.4). Valid on ROUGE-L only; a "better quality" claim needs the human study (section 11) |
| H4 | Cost: for BANKING77 at N=1,000, at GPU utilisation u = 1 and synchronous API pricing, the tuning-charged break-even volume Q* exists and lies inside the pre-specified volume range 10^3..10^7 queries [`STAT` s.7], provided the tuned model is not worse than API retrieval-ICL by more than m = 3 points | B77 N=1,000, cost scenario u=1, sync | the 95% Fieller interval for Q* has a finite upper end at or below 10^7 | the interval is unbounded above, includes "never" (denominator not positive), or exceeds 10^7; or FT is worse than API-ICL by more than m. The sensitivity table in `STAT` s.7 (9.7k / 40k / 265k items at u=1 sync; "never" at some u or batch settings) is an **inputs-based projection, not a result** |

Secondary (outside the Holm family, no hypothesis label): FT vs kNN and API-ICL vs kNN per cell; if kNN is within m of FT the headline says so [`STAT` s.4, 06c R6].

## 4. Design overview

Paired, replicated, nested-pool design. For each task and each label budget N, a pool of N labelled examples is drawn from the training data, and **every arm of the same replicate uses the same pool** (the API arm receives the same pool file) and the same test items. Arms differ only in how they use the pool (section 6). Replicates re-draw the pool and the training seed. Test sets are frozen and hashed; tuning happens on validation data only; confirmatory runs start after two tags (section 7). Gated exploratory pilot stages precede the campaign (section 16).

Grid [`PROT` s.3, corrected on the second pass to the `count_requests.py` output, `PC-01`]:

| | BANKING77 (confirmatory) | BillSum US (confirmatory) | LEDGAR (exploratory) |
|---|---|---|---|
| Test items | 3,080 (full test split) | 1,000 seeded from 3,269 | 3,000 seeded |
| N (all arms) | 200 (class-stratified), 1,000, 3,000 (random) | 200, 1,000, 3,000 (random) | 200, 1,000, 3,000 |
| Replicates R | 8 (+3 random-pool at N=200) | 5 | 3 |
| Fine-tuned models | Qwen3-4B-Instruct-2507 (confirmatory, D-05); Phi-4-mini-instruct exploratory | same | Qwen only |
| k grid (ICL) | {4, 8, 16, 32, 64} (L-25, D-06; `PC-33`) | {1, 2, 3} plus retriever choice (`PC-34`) | {1, 2, 4, 8} (number of trials not decided, `PC-34`) |

- Extra one-pool-per-class conditions N=77 (BANKING77) and N=100 (LEDGAR): ICL and kNN only, exploratory; no fine-tuning below N=200 [`PROT` s.3, L-02; `DATA` s.3 coverage].
- OLMo-3-7B only if time remains; LEDGAR and OLMo are the first cuts [`PROT` s.9, `FEAS` cut ladder]. A cut changes only exploratory cells unless it touches section 9 (see section 15).
- Replicate numbering: replicate 0 is the tuning replicate (`pool_seed=0`, `train_seed=100`) and is **never analysed**; analysed replicates are 1..R with `pool_seed=r`, `train_seed=100+r` [`PROT` s.3, s.4]. R counts analysed replicates (`PROT` s.3 counts 54 B77 finals = 2 models x 3 N x 8 + 6 random-pool finals and 30 BillSum finals = 2 x 3 x 5; with Phi exploratory (D-05) the Qwen-only confirmatory counts are 27 and 15 [arithmetic from those formulas]) [`PROT` s.3; `PC-09`].
- Run order is **replicate-major**, so that a stopped campaign leaves every cell with the same number of replicates.

## 5. Data and splits

### 5.1 Datasets and roles

| Dataset | Role | Size facts | Licence statement (what the documents say, not legal advice) |
|---|---|---|---|
| BANKING77 | Confirmatory | 10,003 train, 3,080 test, 77 classes; 1,000 train rows removed as validation, 9,003 usable pool rows [`DATA` s.3] | CC BY 4.0 (card and repository LICENSE; the README does not scope it) [`DATA` s.1] |
| BillSum, **US split only** | Confirmatory | 18,949 train, 3,269 test; about 17,950 usable pool rows; California split (1,237 test) not used; bill mean 2,111 tokens (max 7,393), summary mean 229 [`DATA` s.2-3] | **No licence from the dataset authors.** CC0 appears only in HF card metadata; cite as "US federal text, public domain by statute", not "CC0 from the authors" [`DATA` s.1-2] |
| LEDGAR (LexGLUE, 100 classes, single label) | Exploratory | 60,000/10,000/10,000 chronological; 338 exact test/train duplicates | MIT (LREC list) and CC BY 4.0 (LexGLUE) disagree with the archive page; release ids, labels and metrics only [`DATA` s.1, L-08] |
| CLINC150, SAMSum | Not run / excluded | SAMSum is CC BY-NC-ND 4.0 | [L-07] |

### 5.2 Splits and pools [`PROT` s.2, L-04]

- **Validation** = 1,000 fixed train rows per task (`VAL_SEED` 42), never inside a pool and never used for training or retrieval; the pool is the remainder. Pools are prefixes of a per-replicate permutation, hence nested across N within a replicate, with a fresh order per `pool_seed`.
- **Selection set** at label budget N is val(N) = the first min(1,000, N) validation rows [`PROT` s.4, L-05]. BillSum validation size for tuning: **1,000 (cap) on every arm, val(N) = min(1,000, N)** (D-02, the researcher's preference, L-26; `--val-cap 1000`; the 300-item alternative would save $22-39 of API cost [`FEAS` s.2] and is not used). The extra BillSum validation GPU time (about 95 s per 1,000-bill pass, about 40 GPU-min per N) is an assumed estimate, not measured at 1,000 bills; the API validation requests are inside the 67,200 BillSum total of `count_requests.py` (L-22). The retriever choice for BillSum uses 300 validation items scored by ROUGE-L of retrieved against gold summaries before any test read [`PROT` s.2].
- **Pool sampling:** random for N >= 1,000; class-stratified (round-robin, at least one example per class, remainder random) **only at N=200** for BANKING77, a separate named condition never pooled with random cells; BillSum is random throughout. A random-pool N=200 condition (R=3) is run for the random-vs-stratified table (exploratory) [L-04].
- **Overlap.** Two N=3,000 pools from 9,003 rows share about 33% of rows (finite-population factor 0.67); BillSum about 17% [`PROT` s.2, 06c section 3 reproduces]. The estimand is therefore stated as conditional on the fixed training set (section 9.1).
- **Retriever:** `BAAI/bge-small-en-v1.5` (MIT), revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (pinned 2026-11-22, `PROGRESS.md`). kNN uses the same embedder. BillSum retriever candidates {bge-head, BM25, TF-IDF}, chosen on validation (above).
- **Near-duplicates are kept.** For every test item the maximum cosine to the pool (and 5-gram Jaccard for BillSum) is stored; results are also reported by similarity bin, and a novel-text sensitivity analysis excludes flagged items [`DATA` s.3]. Indicative counts (own check, validation not involved): BANKING77 417 test items (13.5%) with cosine >= 0.9 to train; BillSum 551 (16.9%) at >= 0.9 and 297 (9.1%) at >= 0.95 [`DATA` s.3].

### 5.3 Test isolation and the test-read ledger

1. **Before the protocol tag, no test row is read** by any experiment code. Tuning, gates and pilots use validation only (pilots to date: test never read, `results/pilot-*/README` and `PROGRESS.md`).
2. **Fingerprint.** `scripts/freeze_test_split.py` (the only code that touches the BANKING77 test split) writes `results/test_split_fingerprint.json` (size and sha256 over text and label rows, never the rows). **This file does not exist yet** [`PROT` s.2]. Decided (D-03): the BillSum test subset is 1,000 items drawn with seed 42 (already in `src/train/billsum.py`) and the LEDGAR subset 3,000 items with seed 42, with fingerprints written by one freeze script before the tag; the script as written covers BANKING77 only, so extending it is open engineering work (assistant).
3. **Gate.** A single `final_eval` entry point refuses to run unless (a) the pre-registration tag exists, (b) the configuration hash is listed in a committed `final_configs.json`, and (c) the test fingerprint matches. Every call appends (config hash, test hash, UTC time, git commit) to the append-only `test_reads.jsonl` [`PROT` s.2, s.9]. This code **does not exist yet** (deferrable until stage 2 passes, `PROT` s.9; it must exist before tag 2).
4. **What counts as a read.** One ledger line per (frozen configuration, replicate, test set) evaluation. The ledger is committed with the results. Rule (decided, D-04): a confirmatory cell is re-run after its first test read only for infrastructure failure (crash, out-of-memory, interrupted job, API outage) with the identical configuration and seeds, never after looking at a test metric; every repeat is logged with its reason; a changed configuration after a test read is a recorded deviation (section 15) and both results are reported.
5. The earlier draft's phrase "each read once per final configuration" is replaced by the ledger above (PC-12).
6. **Contamination.** No "no contamination" claim. Datasets are public for three to seven years; zero-shot base accuracy per model, first-public dates and a descriptive exact-prefix continuation probe (300 test texts per dataset, GPU) are reported [`DATA` s.4, I-17]. The probe reads test texts as prompts: it runs after tag 2 and is logged in the ledger.

## 6. Arms

All arms share one `task_spec` per task (system text = task sentence plus the full label list; user turn `Input: ...\nLabel:`; output cap; decoding; parser). The zero-shot SLM and API prompts are byte-identical (scratch merge; asserted by a test, and re-checked on 20 items per task and technique) [`PROT` s.1, s.7].

| Arm | What it does with the N-pool | Systems |
|---|---|---|
| Zero-shot | nothing (spec only) | 2 SLMs, 3 APIs |
| Retrieval-ICL (k nearest pool items in the user turn, nearest last) | k in grid (section 4); BillSum also retriever choice | local ICL (Qwen exploratory in variant A, tag A; its tuning on validation ran in session 3); 3 APIs (variant A only) |
| Random-k ICL | k random pool items | exploratory: APIs at N=1,000, R=3 (which providers: PC-08); SLMs |
| kNN | label of the most similar pool item(s), similarity-weighted, no LLM; k in {1, 5, 16} chosen on val(N) | first-class baseline, reported next to every API number |
| LoRA fine-tuning | trained on the pool with the zero-shot spec as prompt; r=16, alpha=32, dropout 0.05, AdamW, cosine schedule, warm-up 3%, weight decay 0 | 2 SLMs; Qwen only on LEDGAR |

- **Confirmatory comparator pair per cell:** (a) the best-validation fine-tuned configuration, and (b) the best-validation retrieval-ICL configuration of the selected API (k and, for BillSum, retriever; selection in section 7). Zero-shot, random-k, local ICL and kNN are not candidates for (b) and are reported separately. The fine-tuned model in (a) is **fixed in advance: Qwen3-4B-Instruct-2507** (D-05, L-29); Phi-4-mini is exploratory (a stage-3 pilot, then only if budget remains) and OLMo only if time remains, so no model-selection step exists (`PC-10` closed).
- **Never train on API outputs.** All fine-tuning labels are gold; paid API tiers only (Google's unpaid tier trains on inputs) [`DATA` s.2, L-08].
- **No API fine-tuning** (stated limitation). **Many-shot probe** (k=128) at N=1,000 and 3,000 on the cheapest API and one SLM to bound the k cap (exploratory) [`PROT` s.1].
- Fine-tuned models are evaluated with free generation like every other arm ("deployed generative system"); constrained decoding is a labelled robustness check for SLM arms only; one head-based fine-tune (Qwen, B77, N=1,000, R=3) is an exploratory bridge to Bertsch et al. [`PROT` s.6].
- Phi-4-mini is **not** in the confirmatory set (D-05). `PROT` s.8 stage 3 (Phi, gates G1-G5, G7, one fix cycle) would only qualify it for exploratory cells; stage 3 has **not** been run [section 16].

## 7. Tuning protocol and what is frozen when

### 7.1 Equal stated effort [`PROT` s.4, L-05, L-19, L-25, D-06]

- **Fine-tuning:** 5 trials per (task, N) for Qwen on replicate 0 (D-06, L-25: the researcher's preference of five trials per system supersedes the four of L-05 and L-24), one per learning rate; each trial is one cosine schedule to E_max epochs (N=200: 10; N=1,000: 10; N=3,000: 5; S_min floor 200 steps) with validation evaluation at five checkpoints, early stopping patience 2; selection by validation accuracy (loss tie-break; BillSum: loss); chosen (lr, schedule length, stop step) is frozen. Validation size val(N) = min(1,000, N).
- **Learning-rate grid (decided, D-06).** Five values on a ladder with factor 2: {5e-5, 1e-4, 2e-4, 4e-4, 8e-4}. History: the first grid {5e-5 ... 4e-4} chose 4e-4, its top value, at both N (gate G5 failed; section 16.2); the grid was shifted to {1e-4 ... 8e-4} for the stage-2 rerun (L-24), which again chose 4e-4 and was not at an edge (G5 passed; section 16.3); because 8e-4 was not chosen the fifth value is 5e-5 (D-06; it is in `BASE_LR_GRID`, `PROGRESS.md` 16:30 entry). **At most one extension per grid edge; an arm still at the edge afterwards is reported and kept. E_max at the edge is not extended; it is reported.** Session 3 (section 16.4) selected 8e-4, the top of the five-value grid, at BANKING77 N=3,000, so the edge rule calls for one extension (1.6e-3) in the campaign tuning, a sixth trial for that cell; how that fits "equal effort" is open (`PC-32`).
- **ICL:** 5 values of k, {4, 8, 16, 32, 64}, on validation for BANKING77 (D-06, L-25; equal effort 5 and 5); BillSum has k in {1, 2, 3} (bill length) plus a retriever choice, so its trial count is not 5 (`PC-34`); the prompt text is not tuned (limitation); one paraphrase sensitivity at N=1,000. The API grid is not stated separately in the decisions; whether it also includes k=64 is open (`PC-33`).
- **Edge rule (both arms):** if the chosen value lies on an edge of its grid, extend by one value on that side and repeat the tuning, at most once per edge (D-06). A schedule-length edge (chosen E* equal to E_max: N=1,000 in both stage-2 runs, N=3,000 in session 3) is flagged and reported, **not extended** (D-06; `PC-05` closed). The kNN grid {1, 5, 16} has no high-edge problem; k=1 chosen at N=200 is the lowest possible value.
- **Equality is in trial count, not compute** (a fine-tuning trial is a full training with five validation passes, about 1.5-2x a final run [`PROT` s.4, 06c s.4]). Tuning GPU-hours and dollars are charged in the cost model (section 12).
- **Pre-registered tuning-bar sensitivity** (not exploratory; L-19): on BANKING77, Qwen, replicate-0 pool, N=200 and N=1,000, a 12-configuration fine-tuning grid (4 learning rates x 3 rank/alpha or dropout/weight-decay pairs) and a matched ICL grid (k x retriever x one paraphrased instruction). Decided (D-07): fine-tuning (rank, alpha) in {(8,16), (16,32), (32,64)} x 4 learning rates = 12 configurations at N=200 and 1,000, BANKING77, Qwen; ICL: k x retriever x one paraphrased instruction; held-out slice = the last 500 rows of the 1,000-row validation carve-out, "not used for selection". **At N=1,000, val(N) is all 1,000 validation rows, so the last 500 rows are inside the selection set of the primary arm; the held-out slice only exists if the sensitivity re-selects on the first 500 rows (`PC-31`).** If best-of-12 beats best-of-4 by less than m = 3 points on the held-out slice, the 4-trial rule is reported as adequate; if by more, the campaign grids are widened at N <= 1,000 only (BillSum N=3,000 stays at cut 1) and the flip is reported. Budget: 0.7-2.0 GPU-h (estimate; not re-derived for five learning rates), under $1 API [`PROT` s.4, `FEAS` s.4]. Spiegelhalter et al. use 27 configurations x 5 folds; their App. E.2 finds a fixed configuration within 1-4 points of the per-task oracle at N >= 50 and under 2 points at N=200 [`spiegelhalter-2026-notes.md`]; this is cited in support, not as proof for our tasks.
- **Code prerequisite:** in-run early stopping replaces retraining per (lr, epochs) (A5-12). Tuning code is written and has run on a GPU (stage 2, rerun, session 3, for BANKING77 and, loss-based, for BillSum); BillSum gradient checkpointing and the loss-based branch exist and ran in sessions 2-3 [`PROGRESS.md`]. Not built: BillSum summary generation and ROUGE-L evaluation, the BillSum ICL runner, gates on non-selection rows [`HANDOFF.md`].

### 7.2 Tags (decided: protocol tag, then tag A, then tag B; D-08, L-30) [`PROT` s.4, 06c C6]

| Tag | Planned | Contents frozen |
|---|---|---|
| **Protocol tag** | no date set (`PC-14`) | this file (all sections except the arm choices below), analysis scripts, commit hash, `final_configs.json` schema, test fingerprints, pinned `pip freeze`, grids, margins, R targets and the blinded-re-estimation script |
| **Tag A** | after the protocol tag, before any test read | `frozen_arms.json` and its sha256 for the fine-tuned Qwen arms, the kNN arms and the local ICL arms: per (task, N) the best fine-tuned configuration, kNN k, local-ICL k; all chosen on **replicate-0 validation**, never on test and never re-selected for replicates 1..R |
| **Tag B** (variant A only) | only when an API budget exists | provider, k, retriever and the model id strings of the API arm, chosen on API **validation** data only. **Cannot be created now** (no keys, no money, L-32) |

No test row is read before tag A and no API request on the test set is made before tag B; `final_eval` refuses to run before the tags it needs exist (section 5.3). The risk of seeing fine-tuned test results before tag B is limited by the pre-written selection rule and is disclosed; the order of tags is stated in the paper (D-08).

**Selected provider rule** [`PROT` s.4, L-20]: the provider with the highest replicate-0 validation accuracy for its best retrieval-ICL configuration; ties within 1 standard error go to the cheaper provider. All providers' cost per correct are reported descriptively. The confirmatory CI is conditional on the frozen arms; selection uncertainty (noisy at N=200: 200 items, binomial SE about 3 points) lies outside it and is stated (L-18). In session 3 the fine-tuned and local-ICL arms at N=200 differ by 3.5 points on the 200 selection rows (section 16.4), i.e. about one such standard error.

## 8. Replicates and blinded variance re-estimation

- **Targets:** BANKING77 R=8; BillSum R=5; no confirmatory cell with fewer than 5 replicates (a cell that ends below 5 is demoted to exploratory); replicate-major order [`STAT` s.4, L-06]. Because no API validation runs can be made now (L-32), R is **not** frozen from observed FT-vs-API discordance; 8 and 5 come from the power simulation with assumed inputs (section 9.5) and the blinded re-estimation below is the only data-driven adjustment.
- **Blinded re-estimation.** After replicate 5 is complete for all cells, the variance components of the crossed model (item, replicate, item-by-replicate) are re-estimated **without any arm-effect estimate and without arm labels**, and the minimal detectable gap for m = 3 points is recomputed. For BANKING77 only, R may be extended from 8 to at most 10 if power for m is below 80% [`STAT` s.4]. BillSum is capped at R=5 (its extension would cost +12.5-33 GPU-h [`STAT` s.4, 06c C8]); its stated minimal detectable gaps are 0.6 / 1.0 / 1.9 ROUGE-L points for a replicate SD of the paired mean difference of 0.2 / 0.4 / 0.8 (all assumed, no data) [`STAT` s.3].
- **Who computes it, and what is hidden** (decided, D-09): a script reads the paired difference matrix and returns only the variance components and the implied power, never the mean difference or its sign; the assistant writes and runs it and the supervisor may check it. A one-researcher study cannot blind by person, so the blinding is by code and is auditable. The script does not exist yet (open engineering work).
- **Trigger (decided, D-10).** BANKING77 only: extend from 8 to at most 10 replicates only if the power to detect a 3-point gap is below 80% at N=1,000 or N=3,000. N=200 is excluded from the trigger because it is underpowered a priori and reported as a bracket (L-15); otherwise the rule would always fire there (MDG 4.4 points at R=10 in the pessimistic regime). The trigger is global for the task, not per cell (this reading of D-10 is mine, `PC-11` note).
- Outputs of the re-estimation are descriptive and never change the contrasts, arms or the test data.

## 9. Confirmatory analysis

### 9.1 Estimand [`STAT` s.1]

- N >= 1,000 (random pools): delta_N = E[acc_FT - acc_ICL] over random size-N pools drawn without replacement from the fixed training set (9,003 BANKING77 rows; about 17,950 BillSum rows), over training seeds and test items. It is **conditional on this training set**, labelled so in abstract and tables; no super-population claim is made in the confirmatory family.
- N = 200 (BANKING77, class-stratified pools): conditional on the stratified draw; a separately named condition, never pooled with random cells.
- Both arms are scored on the same test items and the same pools; "arm" means the frozen replicate-0 configuration (estimand: frozen tuned configuration, L-18).

### 9.2 Model

Paired, **crossed items x replicates** linear mixed model of d_ir = c^FT_ir - c^ICL_ir (c = 1 if correct for classification; ROUGE-L F1 per item for BillSum), equivalently variance components with Satterthwaite degrees of freedom; sensitivity: logistic GLMM with random item, replicate and item-by-arm terms. Replicate = (pool seed, training seed). Primary version: uncorrected (conditional label); sensitivity: variance corrected for overlapping pools, sigma^2_rep(1/R + f/(1-f)) with f = N/M [`STAT` s.2]. Type I error for continuous (ROUGE) scores was not simulated; checked once at R=5 when real variance exists [`STAT` s.2, A4-13].

Not used (unsound): majority vote over seeds then McNemar; bootstrap over 3 seeds; pooled McNemar over n*R pairs (type I error 18-45% in simulation); item-mean paired t ignoring replicates (8-22%) [`STAT` s.1].

### 9.3 Family and multiplicity

**Six contrasts:** {BANKING77, BillSum} x N in {200, 1,000, 3,000}; Holm correction at family-wise alpha = 0.05, two-sided; both directions reported (first-step alpha 0.05/6 = 0.0083 is used in the power calculations) [`STAT` s.4, L-03, L-15]. For each contrast report delta-hat, Holm-adjusted p, the 95% crossed-model CI and the 99.2% CI. Everything else is exploratory (section 10). The four-contrast family without N=200 is reported as a sensitivity analysis (L-15).

### 9.4 Outcomes and the inconclusive rule

- **(i) FT better / ICL better** iff the Holm-adjusted p < 0.05 (direction by sign of delta-hat).
- **(ii) Practically equal** (descriptive, **BANKING77 only**) iff the 95% CI lies inside (-m, m) with m = 3 accuracy points; no equivalence p-value; no TOST in the family. Margin justification: smaller than the gap at which the cost ranking could change [`STAT` s.4, R-labelled].
- **(iii) Inconclusive** otherwise. The words "no difference", "equivalent", "ties" are not used for (iii).
- **BillSum has no equivalence outcome**: contrasts are (i) or (iii) only (a 1-point margin needs a CI half-width under 1 point, unreachable at the likely replicate SD) [`STAT` s.4, 06c C10].
- **Precedence (decided, D-11).** The Holm-adjusted test decides "FT better" / "ICL better" / "inconclusive"; the descriptive flags ("practically equal within m", "underpowered a priori") never override it; a contrast that is Holm-significant with a gap below m is reported as "(i), gap below m"; unadjusted intervals are reported beside, labelled as such; the word "equivalent" is never used.
- **Underpowered a priori** [`STAT` s.3, 06c R5; decided, D-12]: for BANKING77, a contrast is declared underpowered a priori, and is then reported as a bracket, iff the minimal detectable gap estimated once from replicate-0 **validation** estimates, fixed **before any test read**, exceeds m = 3 points. Those estimates need FT-vs-comparator discordance and replicate SD that do not exist yet (for the API comparator they need API validation runs, which are unfunded, L-32; `STAT` s.3 inputs are unobserved, A4-02). **Until they exist, N=200 on BANKING77 is declared a bracket by rule**, without a numeric trigger. **The 2-point BillSum threshold is dropped** (vacuous: the stated MDGs 0.6 / 1.0 / 1.9 are all below 2); BillSum reports the realised interval half-width instead. MDGs re-estimated after R=5 are descriptive only.
- **Crossover statement.** Primary: the **bracket** from the contrasts ("ICL better at N_j, FT not worse at N_{j+1}"); secondary N* = root of the piecewise-linear delta(log N) with Monte Carlo draws from a multivariate-t using the crossed-model covariance (report P(no sign change)); power-law fits only as sensitivity. Grid steps are x5 and x3, so no sharper-than-grid claim [`STAT` s.5].

### 9.5 Minimal detectable gaps, BANKING77 (3,080 test items, 80% power, first-step alpha 0.0083; optimistic / pessimistic regime, points of accuracy) [`STAT` s.3]

| N | R=5 | R=8 | R=10 |
|---|---|---|---|
| 200 | 3.8 / 7.6 | **2.7 / 5.0** | 2.5 / 4.4 |
| 1,000 | 2.6 / 4.7 | **1.9 / 3.3** | 1.9 / 2.9 |
| 3,000 | 1.8 / 2.9 | **1.6 / 2.3** | 1.6 / 2.2 |

These rest on assumed ICL accuracy (0.70 / 0.80 / 0.87 at N=200 / 1,000 / 3,000) and on discordance and replicate SD taken from model-vs-model proxies; they are **unobserved for FT-vs-API** and are projections, not results [`STAT` s.3]. A super-population claim would need 0.6-1.6 more points. More than 3,080 test items buys nothing. The design **cannot** support 1-2 point differences at N=200 or N=1,000, equivalence at +-1 point, a point estimate of N* sharper than the grid, or "FT >= API" quality claims for summarisation on ROUGE alone [`STAT` s.8].

## 10. Secondary and exploratory analyses (explicit list)

**Secondary, outside the Holm family** (estimates with CIs; unadjusted p-values labelled): FT vs kNN and API-ICL vs kNN per cell; the decision table reporting the best of {kNN, API-ICL, FT} per cell with its cost [`STAT` s.4, 06c R6].

**Pre-registered sensitivity analyses** (reported beside the primary result): four-contrast family without N=200; overlap-corrected variance; logistic GLMM; the 12-configuration tuning-bar sensitivity (section 7.1); full 1,000-row validation instead of val(N); novel-text subset excluding near-duplicates; for "stochastic" API arms the mean of two runs vs one run (section 14).

**Exploratory** (estimates and CIs; any p-value labelled unadjusted; failed runs reported, not dropped):

1. N=77 (BANKING77) and N=100 (LEDGAR) one-per-class conditions for ICL and kNN.
2. Model comparison (Qwen vs Phi vs OLMo-3-7B, if run); local ICL arms; zero-shot arms; random-k arms; many-shot probe.
3. Technique comparison (QLoRA, DoRA, rsLoRA, each lr-tuned): estimation only; the pilot's earlier rsLoRA gain was confounded with learning rate [`STAT` s.4, I-06].
4. LEDGAR (all arms, Qwen only for fine-tuning).
5. Random-vs-stratified table at N=200 (R=3); per-class results (descriptive; about 40 test items per BANKING77 class, Wilson interval about +-13 points).
6. Near-duplicate similarity bins, length strata, contamination strata and the prefix probe.
7. Head-based fine-tune bridge; constrained-decoding robustness check; paraphrase sensitivity; flip-rate repeats.
8. Calibration (ECE, Brier) only where logprobs exist.
9. BillSum human faithfulness validation, AlignScore / SummaC-ZS, LLM judge (section 11).
10. All-provider cost per correct; sync vs batch agreement on 100 items; GPU-rate, utilisation and labelling-cost sensitivity grids.
11. Any N between grid points (optional N=500 or 2,000 where the bracket spans a factor of 5).

## 11. Metrics

- **Classification:** accuracy over all items, unparseable or out-of-label = wrong (**invalid = wrong**); always report invalid rate, accuracy among valid outputs, macro-F1 over the full fixed label set. Parser: first non-empty line, case- and separator-insensitive exact match to a label, no fuzzy matching; the normaliser must pass a gold-label self-parse test (all 77 + 100 labels parse to themselves) [`PROT` s.6].
- **Summarisation (primary: ROUGE-L F1; BERTScore F1 secondary):** pinned `rouge-score` (stemming, rougeLsum); BERTScore with `microsoft/deberta-xlarge-mnli`, rescaled, hash reported; length covariates (words, compression) with every score; stratified by near-duplicate bin and length [`STAT` s.6, L-10]. Gaps under about 1 ROUGE-L point are not interpretable (Goyal et al.: about 5-point gaps informative) [`STAT` s.3]. BillSum output cap 1,024 tokens for every arm.
- **Human validation (not done unless the researcher names an annotator, HD-2; default decided in D-13: no human study, BillSum quality = ROUGE-L and BERTScore, exploratory, limitation stated):** 60 bills x 3 systems = 180 summaries, blind and shuffled, stratified by length tercile and near-duplicate bin; claim-level faithfulness plus 1-5 acceptability; rater 1 (author) scores all, rater 2 a random 90; Krippendorff alpha with bill-level bootstrap CI; the third system and rater 2 are undecided (D-13 stays with the researcher). A metric is secondary evidence only if Spearman rho >= 0.5 with lower bound >= 0.3. **Without the study:** every BillSum statement is "on ROUGE-L/BERTScore, exploratory", no faithfulness or "as good as" claim, no cost per successful summary (report cost per 1,000 outputs beside ROUGE-L) [`STAT` s.6, L-10].
- **Intervals:** primary crossed-model t interval; per-arm descriptive two-stage percentile bootstrap (replicates then items), 10,000 resamples, fixed seed; with R < 8 report the replicate range plus item bootstrap [`STAT` s.5].

## 12. Cost model

All cost statements are projections until measured. Prices below were read 2026-11-21 and **must be re-read on the day of the API runs**; the price date is stored with every result [`FEAS` s.1, `PROGRESS.md` 05:50 entry].

- **API:** cost per query from provider-billed input (uncached and cached) and output tokens, **including reasoning tokens**, whole prompt including retrieved shots, retries billed; never character counts. Listed rates (USD per 1M tokens, input / output, standard text, prompt-size limits as in `FEAS` s.1): Haiku 5.5 0.10 / 0.50; GPT-6 Luna 0.10 / 0.50; Gemini 3.1 Flash-Lite 0.25 / 1.50 (medium confidence, truncated page); batch 50%. Tokens per request [M, 1,000-row train blocks]: BANKING77 k=32 about 1,220; BillSum k=3 about 9,300 [`FEAS` s.1].
- **Self-hosted:** C_train = all tuning trials plus all finals, GPU-hours x an imputed hourly rate in the band **$3.6-6.0/h** (scenario band, no NVL price exists; $2.3/h spot as a sensitivity row; the university allowance carries no dollar value, HD-3 answered 2026-11-22 (L-32); weekly carry-over is still unknown, D-19) [`STAT` s.7, `FEAS` s.3]; inference per query = measured GPU-seconds / utilisation u in {1, 0.5, 0.2}; measured throughput replaces FLOP-arithmetic ranges (throughput benchmark pending, A5-11).
- **Cost per correct** = (C_train + sum c_q) / sum correct_q at stated volume Q in 10^3..10^7. **Marginal break-even** Q* = C_train / (c_API - c_FT/u), valid only if c_API > c_FT/u; per-correct form Q* = C_train / (a_FT c_API/a_API - c_FT/u). If the denominator's CI contains 0: Fieller interval or P(denominator <= 0) and "no break-even in range"; never a percentile CI of an unbounded ratio. Items are resampled as (cost, correct) pairs, replicates for C_train and accuracy; fixed costs added after resampling [`STAT` s.7].
- **Labelling axis:** total = N c_L + C_train(N) + Q c_q; c_L band 0.10 / 0.50 / 2.00 USD (assumed). For BillSum the labels are sunk (CRS summaries), so this axis is hypothetical [`STAT` s.7].
- **BillSum:** cost per 1,000 outputs beside ROUGE-L; cost per correct is not defined for summarisation (L-10).
- API pricing mode for the base budget (decided, D-14, the researcher's preference L-27): batch for BANKING77 and, where queue limits allow, BillSum; synchronous calls only as the fallback; the 72-hour window runs from the first to the last request of a provider's confirmatory cells, with batch jobs counted by submission time; sync-all is the stress case. This leaves `PC-02` and `PC-35` open: `PROT` s.5 says batch BANKING77 needs 7-13 sequential waves for Luna and Gemini and meets the 72-hour rule only by the sync fallback, and no budget has been re-priced after L-22. The API arm is unfunded (L-32), so none of this can run now. The sensitivity values in `STAT` s.7 and the budget figures in `FEAS` s.2 are inputs for planning, not results.

## 13. Exclusion and failure rules

1. **Nothing is excluded for being inconvenient.** Failed runs are kept in the results and listed; the OLMo-3 DoRA NaN of the old pilot is the model case [old PREREG, `PROT` s.7].
2. **Training health.** Non-finite loss fails the run loudly (no silent continue). A run is **invalid** if loss is non-finite or gold-label parsing fails the self-parse test; it is re-run with the same pool and seeds, the failed attempt is logged.
3. **Examples above `max_length`:** test items are never truncated; training examples above the cap are dropped only if they are under 0.1% of the pool, logged (BillSum; `tokenize_example` raises rather than truncates) [`PROT` s.7].
4. **Output validity:** invalid output counts as wrong; the invalid rate is a reported outcome, never a reason to drop items. API refusals, empty and truncated outputs count as invalid.
5. **API truncation:** truncation rate above 1% in the smoke test stops the run; truncation counts as invalid [`PROT` s.5].
6. **Replicate failure:** a replicate that fails twice for non-infrastructure reasons stays in the record; decided (D-15): it is retried twice with the same seed; if it still fails, the replicate is replaced by the next unused replicate number for that cell, the number of replaced replicates is reported, and replacement never depends on any metric; a cell below 5 analysed replicates is demoted to exploratory (see `PC-40` for the interaction with the R=8 to 10 extension).
7. **Provider drift:** a changed `model_id_returned` mid-run voids that provider's batch; all confirmatory cells of one provider run within 72 hours; one mode (sync or batch) per confirmatory run, agreement checked on 100 items [`PROT` s.5].
8. **Pilot data are never pooled** with confirmatory data and never used in the confirmatory model.
9. **Cell cuts** forced by compute (section 15) are applied to whole cells before tag A where possible; a cut that removes a confirmatory contrast is reported as "not run", not as a result.

## 14. API protocol (set by a live test, not by documentation) [`PROT` s.5]

- **Model ids to be pinned:** `claude-haiku-5-5`, `gpt-6-luna`, `gemini-3.1-flash-lite` as listed in `FEAS` s.1. Haiku 5.5 is described as a "pinned snapshot" by the provider; Luna has only an undated default id. **The exact id string, snapshot date, SDK version and price page version are recorded on the first live call and written into the tag-B addendum** (D-16, stays with the researcher, who holds the keys; no live call exists). Store `model_id_returned`, usage fields, price date and run date for every response. Provider terms version and date are stored in released result files [`DATA` s.2].
- **Sampling:** send `temperature=0` only where a 50-item test shows it is accepted. Otherwise omit it and label the arm **stochastic** (the Haiku 5.5 migration page says to omit `temperature`, `top_p`, `top_k`; the exact failure behaviour is unconfirmed, so the live test decides) [`PROT` s.5, `FEAS` s.1]. For stochastic arms measure the flip rate on a 10% repeat x2; if above 2%, the arm's accuracy is the **mean of two full runs** (pre-priced: about 49,000 extra BANKING77 and about 19,000 extra BillSum requests) [`PROT` s.5]. Decided (D-17): a stochastic arm enters the crossed model as the item-level mean of its two full runs (scores 0, 0.5, 1 for accuracy), replicates unchanged, the item stays the unit, and the flip rate is reported. `TEMPERATURE` is a per-arm field.
- **Reasoning:** lowest accepted setting. *Reasoning-off* is claimed only if reported reasoning tokens are 0 for every test item; otherwise the arm is labelled *reasoning-minimal* with billed reasoning tokens in the cost. Gemini thinking cannot be fully off [`FEAS` s.1]. Arms with different reasoning settings are never compared silently.
- **Output cap:** if reasoning counts against the cap, `max_output_tokens = 32 + max(64, p99.9 observed reasoning tokens)`, else 32; BillSum 1,024 for every arm.
- **Staging:** S0 dry-run cost gate; S1 smoke 20 items; S2 200 validation items; S3 validation tuning of k; S4 test after both tags. Live tests need the researcher's own keys and a spending cap (HD-1 answered 2026-11-22: no keys and no money, L-32). The pilot's stage A (live tests, 50 validation items per provider) **has not run and cannot run now**.
- **All arms are "deployed generative systems"**: free generation, same parser; paid tiers only; no training on API outputs.

## 15. Deviations policy

1. **Before the protocol tag:** this draft may change freely; keep the git history.
2. **After the protocol tag and before tag A:** any change is a deviation entry (what, why, date, whether any test row had been read: must be no) appended to `docs/PREREGISTRATION_DEVIATIONS.md` (to be created at the tag).
3. **After tag A (and, for the API arm, after tag B):** changes to confirmatory contrasts, family, estimand, margins, arm choices or test sets are not allowed without a dated deviation entry that states the reason and whether results had been seen; analyses affected are reported both ways where possible.
4. **Permitted without a deviation entry:** infrastructure re-runs with identical configuration (section 5.3 rule 4); cuts on the cut ladder (`FEAS` s.3) that remove only exploratory cells (LEDGAR, OLMo, random-k); go/no-go outcomes at week 2 (`PROT` and L-16: Qwen BillSum N=3,000 final <= 1.2 h and B77 N=3,000 <= 12 min keeps the design with cut 1; 1.2-2.0 h adds R=3 at BillSum N=3,000, which makes that cell exploratory; > 2.0 h drops BillSum N=3,000 and makes BillSum Qwen-only). The cut ladder changes confirmatory cells at steps 4-6 (BillSum N=3,000 at R=3 or dropped), so those steps **are** deviations from the 6-contrast family and are logged as such, with the family reported as 5 contrasts. **Current provisional state:** the BillSum probe of session 2 projects a one-run time at N=3,000 that falls in the "cuts 1 and 4" class (72.5 min against the 1.2 h = 72 min threshold; an extrapolation, not a measured final run; section 16.4, `PC-36`), which would make BillSum N=3,000 an exploratory R=3 cell; no go/no-go has been formally taken.
5. **Reporting:** the paper has a deviations table; unregistered analyses are labelled exploratory.

## 16. Evidence status and the pilot (exploratory; validation split only; no confirmatory result exists)

### 16.0 Read this first

- **All numbers below are exploratory validation numbers.** The test split was never read. Every run used the 1,000-row validation carve-out of the training set (`VAL_SEED` 42).
- **Replicate counts are tiny:** tuning is replicate 0 (one pool, one seed); finals are replicates 1 and 2 only (one replicate for the random-pool N=200 final and for BillSum). No interval is computed because none would be valid.
- **Selection rows overlap with evaluation rows (I-25, still open).** At N=200 the 200 selection rows are the first 200 of the 1,000 rows on which finals and gates are scored; at N=1,000 and N=3,000 the selection set is all 1,000 rows. The protocol (`PROT` s.8, 06c R1, L-20) asks for non-selection rows; none of the gate numbers below was recomputed that way, and at N=1,000 and N=3,000 no non-selection validation rows exist (`PC-31`). Selection accuracies are also a maximum over five trials, hence optimistic.
- **The stage-2 gates are NOT a pass of the full pilot.** Stage 2 (Qwen, BANKING77, N=200 and 1,000) was run twice and is the only stage with gate criteria that has run; its second run **failed G4**. Stage 1 (loss diagnostics), stage 3 (Phi-4-mini) and stage A (API live tests) have **not been run**; stage 4 (BillSum) ran only as a memory/throughput probe plus loss-based training runs, without generation or ROUGE-L. Phi-4-mini is exploratory (D-05), so stage 3 no longer blocks the confirmatory Qwen arms, but the API protocol (stage A) is **unverified** and cannot be tested now (L-32).
- **A pilot pass or fail is not evidence for H1-H4** [`PROT` s.8]. Nothing here is pooled with confirmatory data.

### 16.1 Inventory

| Folder (`results/`) | Code | Date | What | Replicates and rows | Status |
|---|---|---|---|---|---|
| `pilot-2026-11-21/` | older branches | 2026-11-21 | old prompt (no label list for the fine-tuned arm), fixed pool, under-trained at small N; N in {50, 200, 1,000, 5,000} | 3 seeds, fixed pool | superseded, not used for any number here [`FOUND` header] |
| `pilot-stage2-2026-11-22/` | tag `pilot-stage2-code`, `f533926` | 2026-11-22 03:47-05:00 | stage 2, first run: 4 learning rates {5e-5 ... 4e-4}, Qwen, BANKING77 | tuning replicate 0; finals replicates 1, 2 (+1 random-pool N=200); 1,000 validation rows incl. selection rows | exploratory; **G5 failed** |
| `pilot-stage2b-2026-11-22/` | tag `pilot-stage2b-code`, `ce0dd2a` | 2026-11-22 (GPU session 2) | stage-2 rerun with the grid {1e-4 ... 8e-4}; BillSum probe; local ICL smoke test | same; ICL smoke on 50 validation rows | exploratory; **G4 failed narrowly, G5 passed** |
| `pilot-session3-2026-11-22/` | tag `session3-code`, `12ec11f` | 2026-11-22 (GPU session 3) | BANKING77 N=3,000 tuning (5 lr); local ICL k tuning at three N (5 k); BillSum loss-based tuning N=200 and two BillSum runs | replicate 0 tuning only for BANKING77 and ICL; BillSum finals: 1 run each, loss only | exploratory; no gates defined or computed |

Sources: `PROGRESS.md` entries of 05:02, 16:30 and 18:50 (+03:00); the result files in each folder; `HANDOFF.md` first section. GPU time used on 2026-11-22 so far: about 6 h of the 13.3 h weekly allowance by the assistant's own accounting (billed time may be higher; check the portal) [`PROGRESS.md` 18:50 entry].

### 16.2 Stage 2, first run (`pilot-stage2-2026-11-22/`, tag `pilot-stage2-code`, 4 learning rates {5e-5, 1e-4, 2e-4, 4e-4}) [PILOT, validation only; replicate 0 tuning; 2 final replicates + 1 random-pool final; selection rows overlap]

| Quantity | N=200 (stratified) | N=1,000 (random) |
|---|---|---|
| Replicate-0 selection accuracy at the chosen lr (selection rows, optimistic) | 0.785 (200 val rows) | 0.883 (1,000 val rows) |
| Best-checkpoint val accuracy for lr 5e-5 / 1e-4 / 2e-4 / 4e-4 | 0.73 / 0.735 / 0.76 / 0.785 | 0.831 / 0.849 / 0.862 / 0.883 |
| Chosen lr; stop step | 4e-4 (**top of grid**); 160 of 200 | 4e-4 (**top of grid**); 630 of 630 (E*=10=E_max) |
| Fine-tuned finals, replicates 1 / 2 (1,000 val rows) | 0.776 / 0.769 | 0.869 / 0.872 |
| kNN (bge-small, same pools), replicates 1 / 2 | 0.730 / 0.713 (k=1) | 0.840 / 0.844 (k=5) |
| Random-pool N=200 final, replicate 1 (FT / kNN) | 0.745 / 0.681 | n/a |
| Mean invalid-output rate | 0.4% | 0.1% |
| Fine-tuning GPU time per final / per tuning | 71 s / 457 s | 433 s / 2,534 s |

Gates (script `run_stage2.py`): G1 pass, G2 pass (FT minus kNN +5.1 points), G3 pass (+2.9), G4 pass (+9.8 points from N=200 to 1,000), **G5 FAIL** (chosen lr at the grid top at both N), G6 pass, G7 pass (time-band part not checked); overall FAIL on G5 only. Validation accuracy rose monotonically with lr in this grid, so the frozen lr was an artefact of the grid. [`PROGRESS.md` 05:02 entry, recomputed from `results.jsonl`.] The script's G3 margin was 5 points instead of the protocol's 2 (bug fixed afterwards, `PROGRESS.md` 06:15 entry; `PC-17`).

### 16.3 Stage-2 rerun (`pilot-stage2b-2026-11-22/`, tag `pilot-stage2b-code`, grid {1e-4, 2e-4, 4e-4, 8e-4}) [PILOT, validation only; replicate 0 tuning; 2 final replicates + 1 random-pool final; selection rows overlap]

| Quantity | N=200 (stratified) | N=1,000 (random) |
|---|---|---|
| Replicate-0 selection accuracy at the chosen lr (selection rows) | 0.770 (200 val rows) | 0.871 (1,000 val rows) |
| Best-checkpoint val accuracy for lr 1e-4 / 2e-4 / 4e-4 / 8e-4 [`chain.clean.log` tuning lines] | 0.750 / 0.765 / 0.770 / 0.765 | 0.855 / 0.861 / 0.871 / 0.864 |
| Chosen lr; stop step; grid edge | 4e-4; 160 of 200 (E*=8); none | 4e-4; 630 of 630 (E*=10=E_max); none |
| Fine-tuned finals, replicates 1 / 2 (1,000 val rows) | 0.786 / 0.789 | 0.869 / 0.854 |
| kNN, same pools, replicates 1 / 2 | 0.730 / 0.713 | 0.840 / 0.844 |
| Random-pool N=200 final, replicate 1 (FT / kNN) | 0.753 / 0.681 | n/a |
| Fine-tuning GPU time per final / per tuning; peak memory | 71 s / 509 s; 53 GB | 432 s / 2,537 s; 81 GB |

Gates (automatic report in `chain.clean.log`; same rows as selection, `I-25`): G1 pass (mean invalid 0.45% / 0.2%, max 0.6%); G2 pass (FT minus kNN +6.6 points: +5.6 and +7.6); G3 pass at the protocol's 2-point margin (+1.95 points: +2.9 and +1.0); **G4 FAIL**: accuracy gain from N=200 to N=1,000 is 7.4 points (0.7875 to 0.8615) against the threshold of 8; G5 pass (no edge at either N); G6 pass (replicate spread 0.3 and 1.5 points); G7 pass (time-band part not checked). Overall: FAIL (G4).

**Why G4 failed, from the files.** (1) The threshold of 8 points was set in advance (`PROT` s.8) and the miss is 0.6 points. (2) The lr chosen and the stop step are the same as in the first run (4e-4; step 160 and step 630), so the failure is not due to a different tuned configuration. (3) The finals moved between the two runs with the same lr and the same pool and training seeds: N=200 finals rose by 1.0 and 2.0 points (0.776 to 0.786, 0.769 to 0.789) and N=1,000 finals changed by 0.0 and -1.8 points (0.869 to 0.869, 0.872 to 0.854); the replicate-0 selection accuracy of the same configuration also moved (N=200 0.785 to 0.770, N=1,000 0.883 to 0.871). The code commit also differs (`f533926` vs `ce0dd2a`). The cause of this run-to-run movement of 1-2 points was not investigated [open; candidates, not verified: GPU nondeterminism, code changes between the commits]. The explanation in `PROGRESS.md` (the better-tuned N=200 improved) is therefore only partly supported by the files. (4) G4 is not in the protocol's fix-and-repeat list (`PROT` s.8 lists G1, G2, G3, G5, G6), so the protocol has no rule for this outcome; it is neither waived nor repeated here (`PC-30`). (5) The size of the movement, 1-2 points, is of the order of the replicate spread (G6) and of the minimal detectable gaps of section 9.5 at N=200.

### 16.4 GPU session 3 (`pilot-session3-2026-11-22/`, tag `session3-code`) [PILOT, validation only; replicate 0 only for BANKING77 and ICL; selection rows = evaluation rows]

**BANKING77 N=3,000 fine-tuning tuning** (five learning rates, 4,241 s = 70.7 GPU-min; schedule 940 steps = E_max 5 epochs; 1,000 validation rows). Best-checkpoint val accuracy for lr 5e-5 / 1e-4 / 2e-4 / 4e-4 / 8e-4: 0.889 / 0.904 / 0.912 / 0.921 / 0.921 [`chain.clean.log`; `PROGRESS.md` quotes last-checkpoint values 0.903 and 0.910 for two of them]. The two best values tie at 0.921; the loss tie-break (0.0629 at 8e-4, 0.0675 at 4e-4) selected **8e-4, the top edge of the grid**, with the stop step at the schedule end (E*=5=E_max). Edge rule (D-06): one extension (1.6e-3) in the campaign tuning, about +12 GPU-min [`PROGRESS.md`; estimate], not yet run; E_max is not extended. Consequence: the grid is still at its top edge for N=3,000, and this cell would have six trials (`PC-32`).

**Local ICL (Qwen3-4B-Instruct-2507, retrieved examples, k in {4, 8, 16, 32, 64}) on validation, replicate 0** (`icl_arms.json`):

| N | val rows | val accuracy at k = 4 / 8 / 16 / 32 / 64 | chosen k | fine-tuning selection accuracy (same rows) | difference FT minus ICL |
|---|---|---|---|---|---|
| 200 (stratified) | 200 | 0.790 / 0.805 / 0.790 / 0.780 / 0.790 | 8 | 0.770 | -3.5 points |
| 1,000 (random) | 1,000 | 0.844 / 0.853 / 0.850 / 0.848 / 0.850 | 8 | 0.871 | +1.8 |
| 3,000 (random) | 1,000 | 0.896 / 0.901 / 0.894 / 0.900 / 0.888 | 8 | 0.921 | +2.0 |

One replicate; selection rows; both columns are a maximum over five trials; the binomial SE of one accuracy at 200 rows is about 3 points. So local ICL ahead at N=200 and fine-tuning ahead at N=1,000 and 3,000 is **a descriptive pattern only**, not a test, not a crossover estimate, and not evidence for H1-H3 (which are stated against the commercial API). k hardly matters (spread 1-2 points). Each ICL tuning took 1-2 min GPU. The earlier session-2 smoke test (4 k values, 50 validation rows, k=8 best at 0.82; replicate-1 final 0.78 on 50 rows) only shows that the vLLM path works (after the `hide_broken_flash_attn` workaround) [`PROGRESS.md` 16:30 entry].

**BillSum** (Qwen3-4B-Instruct-2507; loss-based selection; **no summary generation and no ROUGE-L exist yet**, so no summarisation quality number exists):
- *Probe* (session 2, `probe.jsonl`): 20 steps on the longest 1% of bills (longest 7,970 tokens), micro-batch 1 with gradient checkpointing: peak memory 24.4 GB, 8,248 tokens/s, 49 s. Projected one full-schedule run (upper bound by assumption): 925.9 / 2,916.6 / 4,351.7 s = 15.4 / 48.6 / 72.5 min for N = 200 / 1,000 / 3,000; the pre-set class for N=3,000 is "cuts 1 and 4" because 72.5 min exceeds 1.2 h = 72 min by 0.5 min.
- *Loss-based tuning N=200* (five lr, 200 validation bills, 42.8 GPU-min): chosen lr 4e-4, stop step 40 of 200 (epoch 2), validation loss 0.7464 on the 200 selection bills; higher learning rates overfit sooner (final losses 0.755 / 0.789 / 0.826 / 0.978 / 1.124 from 5e-5 to 8e-4, `PROGRESS.md`).
- *Frozen-arm final, N=200, replicate 1:* validation loss 0.7877 on 1,000 bills, 120.8 s, 19.2 GB peak.
- *Untuned run, N=1,000* (lr 4e-4, 5 epochs, 315 steps, batch 16): 1,512 s, 19.2 GB, validation loss 0.7842 (train loss 0.41, i.e. overfit). The measured time per step is about 4.8 s [arithmetic from 1,512 s / 315 steps]; extrapolated to the 630-step and 940-step schedules this is 50.4 and 75.2 min [arithmetic], slightly above the probe's projections, so "upper bound" holds only if loss-based stopping ends runs early (as at N=200, where it stopped at step 40 of 200) [I].
- **Warning for the summarisation claim:** validation loss is flat between N=200 (0.788 on 1,000 bills) and an untuned N=1,000 run (0.784). Loss on this kind of run is not the confirmatory metric (ROUGE-L), the N=1,000 run was not tuned, and nothing yet says whether more labels improve the summaries, but it is a reason to expect a flat BillSum curve.
- The go/no-go rule of L-16 is defined on a measured final run (<= 1.2 h) while only a projection exists (`PC-36`).

### 16.5 What the pilot has not shown

Stage 1 loss diagnostics (Phi first-step loss 18.88, OLMo DoRA NaN; I-14) not run; stage 3 Phi-4-mini not run (exploratory only, D-05); stage 4 not complete (no BillSum generation or ROUGE-L, no BillSum gates); stage A (API live tests: temperature and thinking acceptance, reasoning tokens, truncation, flip rate) not run and unfundable now (L-32); no ICL evaluation beyond validation tuning and a 50-row smoke test; no retrieval-ICL or fine-tuning number on a test split; no gate recomputed on non-selection rows; no pilot number comparing against any commercial API. Hence I-01 and I-02 (fine-tuned arm proven under the corrected prompt) have positive but exploratory evidence for Qwen on BANKING77 only, not independent sign-off.

### 16.6 What changed in this pre-registration because of the pilots

1. **Learning-rate ladder fixed** at five values {5e-5 ... 8e-4} (section 7.1), one extension per edge; the N=3,000 edge hit of session 3 is an open consequence (`PC-32`).
2. **Schedule-length edge** (E* = E_max at N=1,000 and N=3,000) is flagged and reported, not extended (D-06).
3. **kNN:** k=1 at N=200 (lowest possible value) and k=5 at N=1,000; kNN stays a first-class comparator because it is within about 2-7 points of fine-tuning on the validation rows (secondary contrasts, `PROT` s.2).
4. **Gate implementation:** the gate script scores all 1,000 validation rows including the selection rows, against a protocol that asks for non-selection rows (`PC-17`, `PC-31`); G3 now uses the protocol's 2-point margin.
5. **Measured timings** (BANKING77 N=200, 1,000, 3,000 tuning; BillSum probe and runs) are recorded in section 16 and in `FEAS` s.3a; they replace no committed budget figure yet. The week-2 go/no-go is provisional (`PC-36`).
6. **BillSum selection uses validation loss** because no ROUGE-L evaluation exists; whether the confirmatory BillSum arms keep loss-based selection is as in `PROT` s.4 (loss) and is not changed here.
7. **No change** to N levels, R, family, test sizes or the estimand.

### 16.7 Open pilot items before the protocol tag

Decide what the G4 miss means (`PC-30`); gates on non-selection rows or a decision that they cannot exist at N >= 1,000 (`PC-31`); run the N=3,000 extension trial and a BillSum N=3,000 final or accept the provisional go/no-go class (`PC-32`, `PC-36`); BillSum generation and ROUGE-L evaluation and the BillSum ICL runner; Stage-0 exit items (tagged commit, `pip freeze`, test fingerprints for BANKING77, BillSum and LEDGAR, `test_reads.jsonl`, `final_eval`); the blinded variance script (D-09); the independent re-review (D-20); stage A only if an API budget appears. If G1-G7 cannot be passed after two fix cycles of about 2 GPU-h each, the outcome is *re-scope* (head-based fine-tuning, fine-tuning only at N >= 1,000, or a kNN/ICL/API-cost paper) and this pre-registration is rewritten, not tagged [`PROT` s.8].

## 17. Decisions on the former `[TO DECIDE BEFORE FREEZE]` items

All 21 items now carry a decision. Sixteen were decided by the assistant under the researcher's explicit delegation on 2026-11-22 (`PD`, L-28), with three stated researcher preferences (five tuning trials per system, 1,000 validation examples, a batch-based API budget); the researcher can overrule any of them before the protocol tag. **Five stay open with the researcher** (D-13, D-16, D-18, D-19, D-21). "Open" in the last column means an issue is flagged in `PREREG_CONSISTENCY.md` section F, not that the decision was changed.

| ID | Item | Decision (`PD`) | Status / owner | Flag |
|---|---|---|---|---|
| D-01 | Direction and wording of H1-H4 | accepted as drafted (section 3) | decided | H1-H4 exist only in variant A (`PC-29`) |
| D-02 | BillSum validation size | 1,000 (cap) on every arm, `val(N) = min(1,000, N)` (researcher's preference, L-26) | decided | API request total 67,200 (L-22); see `PC-33` |
| D-03 | BillSum and LEDGAR test subsets and fingerprints | BillSum 1,000 items, seed 42; LEDGAR 3,000 items, seed 42; one freeze script before the tag | decided; code open (assistant) | |
| D-04 | Re-run rule after a first test read | infrastructure failure only, same configuration, never after a test metric; every repeat logged | decided | |
| D-05 | Fine-tuned model in the confirmatory contrasts | Qwen3-4B-Instruct-2507 fixed in advance; Phi-4-mini exploratory; OLMo only if time remains (L-29) | decided | |
| D-06 | Learning-rate grid, trials, edge rule | five trials per system (fine-tuning five lr, ICL five k {4, 8, 16, 32, 64}, BillSum ICL {1, 2, 3}); ladder factor 2; fifth value 5e-5 because 8e-4 was not chosen in the rerun; at most one extension per edge; E_max at the edge not extended | decided; fifth value fixed after the rerun | N=3,000 hit 8e-4 in session 3 (`PC-32`); BillSum trial count (`PC-34`); API k grid (`PC-33`) |
| D-07 | Tuning-bar sensitivity | (rank, alpha) in {(8,16), (16,32), (32,64)} x 4 lr = 12 configurations at N=200 and 1,000, BANKING77, Qwen; ICL k x retriever x one paraphrase; held-out slice = last 500 of the 1,000 validation rows | decided | held-out slice does not exist at N=1,000 (`PC-31`) |
| D-08 | Arm freeze order | protocol tag, tag A (fine-tuned, kNN, local ICL), tag B (API provider, k, retriever, ids); no API test request before B, no test read of any arm before A (L-30) | decided | tag B cannot exist now (L-32, `PC-29`) |
| D-09 | Blinded variance re-estimation | a script returns only variance components and implied power, never the mean difference or its sign; assistant writes and runs it, supervisor may check | decided; script open | |
| D-10 | R-extension trigger | BANKING77 only: extend 8 to at most 10 if power for 3 points is below 80% at N=1,000 or N=3,000; N=200 excluded | decided | |
| D-11 | Outcome labels | Holm-adjusted test decides; descriptive flags never override; unadjusted intervals labelled; "equivalent" never used | decided | |
| D-12 | "Underpowered a priori" | computed once from replicate-0 validation estimates after API validation runs exist; until then N=200 is a bracket by rule; BillSum 2-point threshold dropped, realised half-width reported | decided | needs API validation runs, unfunded (`PC-29`) |
| D-13 | Human faithfulness study | not done unless the researcher names an annotator; default ROUGE-L and BERTScore, exploratory, limitation stated | **researcher** (annotator, HD-2) | |
| D-14 | API budget mode | batch-based (researcher's preference, L-27): batch for BANKING77 and, where queue limits allow, BillSum; sync as the fallback; 72-hour window from first to last request of a provider's confirmatory cells, batch counted by submission time; sync-all is the stress case | decided | `PC-35`; unfunded |
| D-15 | Replicate failure | two retries with the same seed; then the next unused replicate number; the number replaced is reported; never depends on a metric | decided | `PC-40` |
| D-16 | API model ids, snapshot dates, SDK and price-page versions | recorded at the first live call | **researcher** (keys) | no live call exists |
| D-17 | Stochastic API arm in the crossed model | item-level mean of its two full runs (0, 0.5, 1), flip rate reported | decided | |
| D-18 | Authors, affiliation, funding, data-availability statement, registry | personal information | **researcher** | |
| D-19 | Credit-to-dollar value and carry-over of the GPU allowance | imputed rate band $3.6-6.0/h stays a scenario | **researcher**: the value part is answered (the credits carry no dollar value, L-32); carry-over is still unknown | |
| D-20 | Independent re-review before the tag | a fresh reviewer re-reads the final protocol text and the pilot results; the assistant starts it | decided; review open | gate 8 not passed |
| D-21 | Provider terms and dataset licences | confirmation after the runs, before any release of predictions or adapters; no release before that | **researcher** | |

## Appendix A. Documents each section copies or follows

| Section | Source |
|---|---|
| 2-3 | `FOUND` s.1-4, `spiegelhalter-2026-notes.md` |
| 4-5 | `PROT` s.1-3, s.7, s.9; `DATA` s.1-4; L-02, L-04, L-07, L-08 |
| 6-7 | `PROT` s.1, s.4, s.6, s.8; L-05, L-09, L-17, L-19, L-20 |
| 8-9 | `STAT` s.1-5; L-03, L-06, L-15, L-18, L-20; 06c R5, R6 |
| 11-12 | `STAT` s.5-7; `FEAS` s.1-3; L-10 |
| 13-14 | `PROT` s.5-7; `FEAS` s.5 |
| 16 | `results/pilot-stage2-2026-11-22/`, `pilot-stage2b-2026-11-22/`, `pilot-session3-2026-11-22/`; `PROGRESS.md`; `PROT` s.8 |
| 17 | `PREREG_DECISIONS.md` (D-01..D-21); `DECISION_LOG.md` L-25..L-32 |

## Appendix B. Superseded content of the draft of 2026-11-21

N = 50 and N = 5,000 (L-02); five replicates for all cells (L-06); McNemar and paired bootstrap over items, two contrasts per task, Holm over tasks x 2 (replaced by the crossed model and six contrasts, L-03, L-15); "pool sampling pending" (decided, L-04); LEDGAR conditional on a licence check with CLINC150 fallback (LEDGAR is exploratory, ids/labels/metrics only, L-07, L-08); Gemini "optional" (all three APIs run; the selected provider carries R=8/5, the others R=3); k_max 5 for summarisation (now 3); "reasoning off" for Luna as fixed (now set by a live test); BillSum "CC0 per card" (replaced by the statutory wording); test sets "at least 1,500 items where available" (B77 3,080, BillSum 1,000 seeded, fixed by power analysis).
