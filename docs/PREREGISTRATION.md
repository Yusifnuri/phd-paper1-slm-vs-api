# Pre-registration (DRAFT v2, not frozen, not registered anywhere)

**Status.** Rewritten 2026-11-22 from the foundation documents (`docs/foundation/*.md`, the independent review `agents/06c-independent-redteam.md`, `docs/research/spiegelhalter-2026-notes.md`) and the exploratory stage-2 pilot (`results/pilot-stage2-2026-11-22/`). It replaces the draft of 2026-11-21, whose N levels (50, 5,000), 5 replicates, McNemar test, 2 contrasts per task and conditional LEDGAR role are superseded (list in Appendix B).

- **No confirmatory result exists.** No test split has been read by any experiment of this study. The pilots (`results/pilot-2026-11-21/`, `results/pilot-stage2-2026-11-22/`) used only a 1,000-row validation split carved from the training set. They are exploratory and are never pooled with confirmatory data.
- **Nothing here has been submitted.** This file is a working draft for the researcher. It becomes a pre-registration only when the researcher (a) resolves every `[TO DECIDE BEFORE FREEZE]` item in section 17, (b) commits it and creates the protocol tag (section 7), and (c) optionally copies it to a registry. Until then every number may change, and changes before the tag need no deviation note.
- **Conventions.** Numbers are copied from the foundation documents with a pointer in square brackets (`STAT` = `STATISTICAL_ANALYSIS_PLAN.md`, `PROT` = `EXPERIMENT_PROTOCOL.md`, `FEAS` = `FEASIBILITY_AND_BUDGET.md`, `DATA` = `DATASET_AUDIT.md`, `FOUND` = `RESEARCH_FOUNDATION.md`, `L-nn` = `DECISION_LOG.md`, `I-nn` = `ISSUE_REGISTER.md`). Where two documents disagree, the choice made here is explained in `docs/foundation/PREREG_CONSISTENCY.md` (cited as `PC-nn`). **Pilot** numbers are marked `[PILOT, validation only]`. Model ids, prices and API facts were read from provider pages on 2026-11-21 through fetch agents (`FEAS` section 1) and are not independently re-verified here.
- **Naming.** The hypotheses are called H1-H4 here. `HUMAN_DECISIONS.md` also uses H1-H5 for researcher decisions (API keys, annotator, credits, paper access, venue); in this file those are written `HD-1..HD-5` (PC-24).

---

## 1. Study information

| Field | Value |
|---|---|
| Working title | When does a tuned 4B model beat a cheap API with retrieval? Equal-pool accuracy and cost on BANKING77 and BillSum (`PAPER_OUTLINE.md`) |
| Author, affiliation, funding | [TO DECIDE BEFORE FREEZE] owner: researcher |
| Draft date / protocol-tag date / arms-tag date | draft 2026-11-22 / planned week 3 (Dec 6-12) / planned end of week 4 (Dec 13-19) [FEAS s.4; `PC-14`: dates are not a promise] |
| Code commit and tag | [TO DECIDE BEFORE FREEZE] owner: researcher. The pilot code is at tag `pilot-stage2-code` (`f533926`, `docs/foundation/PROGRESS.md`); the protocol tag must point to a commit containing the frozen analysis scripts |
| Registry | None. Whether to copy the frozen file to OSF is the researcher's choice (HD, section 17) |

## 2. Research questions

Source: `FOUND` section 1 (decision-grade form, L-01) and section 4. The study is a decision-grade comparison for a practitioner who holds N labelled examples. It is **not** a "first crossover" study: the classification crossover at tens to hundreds of labels is already reported (Pecher et al. 2025; Spiegelhalter, Franke, Hutter 2026; Bertsch et al. 2025) [`FOUND` s.1, s.2; `spiegelhalter-2026-notes.md`].

| RQ | Question | Status here |
|---|---|---|
| RQ1 | Data efficiency: at which N in {200, 1,000, 3,000} does a LoRA-tuned small model (Qwen3-4B-Instruct-2507, Phi-4-mini-instruct) overtake retrieval-augmented in-context learning on the best of three cheap commercial APIs, given the same labelled pool? | Confirmatory (H1, H2) |
| RQ2 | Task dependence: does the answer differ between intent classification (BANKING77) and long-document summarisation (BillSum, US)? | Confirmatory (H3) |
| RQ3 | Economic efficiency: accuracy and total cost, tuning charged, for each arm at the same pool | Confirmatory for H4, otherwise descriptive |
| RQ4 | Break-even query volume Q* against the API, with sensitivity to GPU rate band, utilisation and sync vs batch pricing | H4 |
| RQ5 | Generalisation | **Not tested.** Scope is one classification benchmark, one summarisation benchmark, two small models, three APIs, one GPU type [`FOUND` s.4] |

Positioning limits (what the paper may and may not claim) are in `FOUND` section 2 and are not repeated here. In particular: no "first" or "novel" claim; no "equivalent", "no difference" or "ties" wording for an inconclusive contrast; no summarisation-quality claim beyond ROUGE-L and BERTScore unless the human validation study is run (section 11).

## 3. Hypotheses

Each hypothesis is tied to confirmatory contrasts (section 9) and states what outcome would **refute** it. Directions are the drafting proposal derived from the cited sentences and are not results; the researcher confirms each direction before the freeze (`[TO DECIDE BEFORE FREEZE]` D-01). "Contrast" means delta = accuracy(best-validation fine-tuned configuration) minus accuracy(best-validation retrieval-ICL configuration of the selected API) on the test items, conditional on the frozen arms (section 9). Outcome labels (i)/(ii)/(iii) are defined in section 9.4.

| ID | Statement | Cells | Supported if | Fails if |
|---|---|---|---|---|
| H1 | On BANKING77 the tuned small model beats API retrieval-ICL at the larger label budgets. Motivation: Spiegelhalter et al. find LoRA ahead of local ICL up to N=200 and leave N>200 open; Bertsch et al. find retrieval ICL competitive with LoRA on BANKING77 [`FOUND` s.2; notes consequence 4] | B77 N=1,000 and N=3,000 | both contrasts have outcome (i) "FT better" (Holm p < 0.05, delta > 0) | either contrast is (iii) inconclusive, (ii) practically equal, or (i) "ICL better". The words "no difference" are not used for (iii) |
| H2 | At the smallest budget the API with retrieval is not worse than the tuned model on BANKING77 (the crossover lies above N=200 or inside the bracket) | B77 N=200 (class-stratified pools) | outcome is not "FT better": (iii), (ii) or "ICL better" | outcome (i) "FT better" at N=200. N=200 is the weak cell: realised minimal detectable gap 2.7-5.0 points at R=8 [`STAT` s.3], so (iii) is the expected outcome and is reported as a bracket, not as support for H2 beyond "FT not shown better" |
| H3 | Task dependence: on BillSum (ROUGE-L F1) the tuned model beats API retrieval-ICL at N=1,000 and N=3,000. Motivation: "fine-tuning wins mainly at larger N or on summarisation" is a plausible mixed outcome [`FOUND` s.1]; Pecher advises prompting for generation but did not test it [`FOUND` s.2] | BillSum N=1,000 and N=3,000 | both contrasts (i) "FT better" | either contrast is (iii) or (i) "ICL better". BillSum has no equivalence outcome (section 9.4). Valid on ROUGE-L only; a "better quality" claim needs the human study (section 11) |
| H4 | Cost: for BANKING77 at N=1,000, at GPU utilisation u = 1 and synchronous API pricing, the tuning-charged break-even volume Q* exists and lies inside the pre-specified volume range 10^3..10^7 queries [`STAT` s.7], provided the tuned model is not worse than API retrieval-ICL by more than m = 3 points | B77 N=1,000, cost scenario u=1, sync | the 95% Fieller interval for Q* has a finite upper end at or below 10^7 | the interval is unbounded above, includes "never" (denominator not positive), or exceeds 10^7; or FT is worse than API-ICL by more than m. The sensitivity table in `STAT` s.7 (9.7k / 40k / 265k items at u=1 sync; "never" at some u or batch settings) is an **inputs-based projection, not a result** |

Secondary (outside the Holm family, no hypothesis label): FT vs kNN and API-ICL vs kNN per cell; if kNN is within m of FT the headline says so [`STAT` s.4, 06c R6].

## 4. Design overview

Paired, replicated, nested-pool design. For each task and each label budget N, a pool of N labelled examples is drawn from the training data, and **every arm of the same replicate uses the same pool** (the API arm receives the same pool file) and the same test items. Arms differ only in how they use the pool (section 6). Replicates re-draw the pool and the training seed. Test sets are frozen and hashed; tuning happens on validation data only; confirmatory runs start after two tags (section 7). Gated exploratory pilot stages precede the campaign (section 16).

Grid [`PROT` s.3; the BillSum API request counts there are inconsistent, `PC-01`]:

| | BANKING77 (confirmatory) | BillSum US (confirmatory) | LEDGAR (exploratory) |
|---|---|---|---|
| Test items | 3,080 (full test split) | 1,000 seeded from 3,269 | 3,000 seeded |
| N (all arms) | 200 (class-stratified), 1,000, 3,000 (random) | 200, 1,000, 3,000 (random) | 200, 1,000, 3,000 |
| Replicates R | 8 (+3 random-pool at N=200) | 5 | 3 |
| Fine-tuned models | Qwen3-4B-Instruct-2507, Phi-4-mini-instruct | same | Qwen only |
| k grid (ICL) | {4, 8, 16, 32} | {1, 2, 3} | {1, 2, 4, 8} |

- Extra one-pool-per-class conditions N=77 (BANKING77) and N=100 (LEDGAR): ICL and kNN only, exploratory; no fine-tuning below N=200 [`PROT` s.3, L-02; `DATA` s.3 coverage].
- OLMo-3-7B only if time remains; LEDGAR and OLMo are the first cuts [`PROT` s.9, `FEAS` cut ladder]. A cut changes only exploratory cells unless it touches section 9 (see section 15).
- Replicate numbering: replicate 0 is the tuning replicate (`pool_seed=0`, `train_seed=100`) and is **never analysed**; analysed replicates are 1..R with `pool_seed=r`, `train_seed=100+r` [`PROT` s.3, s.4]. R counts analysed replicates (the 54 B77 finals = 2 models x 3 N x 8 + 6 random-pool finals; BillSum 30 finals = 2 x 3 x 5) [`PROT` s.3; `PC-09`].
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
- **Selection set** at label budget N is val(N) = the first min(1,000, N) validation rows [`PROT` s.4, L-05]. BillSum validation size for API tuning: [TO DECIDE BEFORE FREEZE] D-02, owner: researcher (1,000-capped, as `count_requests.py` default, vs 300 items, a saving of $22-39 [`FEAS` s.2]; `PC-01`). The retriever choice for BillSum uses 300 validation items scored by ROUGE-L of retrieved against gold summaries before any test read [`PROT` s.2].
- **Pool sampling:** random for N >= 1,000; class-stratified (round-robin, at least one example per class, remainder random) **only at N=200** for BANKING77, a separate named condition never pooled with random cells; BillSum is random throughout. A random-pool N=200 condition (R=3) is run for the random-vs-stratified table (exploratory) [L-04].
- **Overlap.** Two N=3,000 pools from 9,003 rows share about 33% of rows (finite-population factor 0.67); BillSum about 17% [`PROT` s.2, 06c section 3 reproduces]. The estimand is therefore stated as conditional on the fixed training set (section 9.1).
- **Retriever:** `BAAI/bge-small-en-v1.5` (MIT), revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (pinned 2026-11-22, `PROGRESS.md`). kNN uses the same embedder. BillSum retriever candidates {bge-head, BM25, TF-IDF}, chosen on validation (above).
- **Near-duplicates are kept.** For every test item the maximum cosine to the pool (and 5-gram Jaccard for BillSum) is stored; results are also reported by similarity bin, and a novel-text sensitivity analysis excludes flagged items [`DATA` s.3]. Indicative counts (own check, validation not involved): BANKING77 417 test items (13.5%) with cosine >= 0.9 to train; BillSum 551 (16.9%) at >= 0.9 and 297 (9.1%) at >= 0.95 [`DATA` s.3].

### 5.3 Test isolation and the test-read ledger

1. **Before the protocol tag, no test row is read** by any experiment code. Tuning, gates and pilots use validation only (pilots to date: test never read, `results/pilot-*/README` and `PROGRESS.md`).
2. **Fingerprint.** `scripts/freeze_test_split.py` (the only code that touches the BANKING77 test split) writes `results/test_split_fingerprint.json` (size and sha256 over text and label rows, never the rows). **This file does not exist yet** [`PROT` s.2]. An equivalent fingerprint for the 1,000 seeded BillSum items (and LEDGAR items) and the seed that draws them are [TO DECIDE BEFORE FREEZE] D-03, owner: researcher/code; the script as written covers BANKING77 only.
3. **Gate.** A single `final_eval` entry point refuses to run unless (a) the pre-registration tag exists, (b) the configuration hash is listed in a committed `final_configs.json`, and (c) the test fingerprint matches. Every call appends (config hash, test hash, UTC time, git commit) to the append-only `test_reads.jsonl` [`PROT` s.2, s.9]. This code **does not exist yet** (deferrable until stage 2 passes, `PROT` s.9; it must exist before tag 2).
4. **What counts as a read.** One ledger line per (frozen configuration, replicate, test set) evaluation. The ledger is committed with the results. Rule (drafting proposal, D-04): a confirmatory cell is re-run after its first test read only for infrastructure failure (crash, out-of-memory, API outage) with the identical configuration and seeds; any other re-run is a deviation (section 15) and both results are reported.
5. The earlier draft's phrase "each read once per final configuration" is replaced by the ledger above (PC-12).
6. **Contamination.** No "no contamination" claim. Datasets are public for three to seven years; zero-shot base accuracy per model, first-public dates and a descriptive exact-prefix continuation probe (300 test texts per dataset, GPU) are reported [`DATA` s.4, I-17]. The probe reads test texts as prompts: it runs after tag 2 and is logged in the ledger.

## 6. Arms

All arms share one `task_spec` per task (system text = task sentence plus the full label list; user turn `Input: ...\nLabel:`; output cap; decoding; parser). The zero-shot SLM and API prompts are byte-identical (scratch merge; asserted by a test, and re-checked on 20 items per task and technique) [`PROT` s.1, s.7].

| Arm | What it does with the N-pool | Systems |
|---|---|---|
| Zero-shot | nothing (spec only) | 2 SLMs, 3 APIs |
| Retrieval-ICL (k nearest pool items in the user turn, nearest last) | k in grid (section 4); BillSum also retriever choice | 2 SLMs (local ICL, exploratory), 3 APIs |
| Random-k ICL | k random pool items | exploratory: APIs at N=1,000, R=3 (which providers: PC-08); SLMs |
| kNN | label of the most similar pool item(s), similarity-weighted, no LLM; k in {1, 5, 16} chosen on val(N) | first-class baseline, reported next to every API number |
| LoRA fine-tuning | trained on the pool with the zero-shot spec as prompt; r=16, alpha=32, dropout 0.05, AdamW, cosine schedule, warm-up 3%, weight decay 0 | 2 SLMs; Qwen only on LEDGAR |

- **Confirmatory comparator pair per cell:** (a) the best-validation fine-tuned configuration, and (b) the best-validation retrieval-ICL configuration of the selected API (k and, for BillSum, retriever; selection in section 7). Zero-shot, random-k, local ICL and kNN are not candidates for (b) and are reported separately. Which fine-tuned model enters (a) (Qwen vs Phi) is decided on replicate-0 validation and frozen: [TO DECIDE BEFORE FREEZE] D-05, owner: researcher (`PC-10`).
- **Never train on API outputs.** All fine-tuning labels are gold; paid API tiers only (Google's unpaid tier trains on inputs) [`DATA` s.2, L-08].
- **No API fine-tuning** (stated limitation). **Many-shot probe** (k=128) at N=1,000 and 3,000 on the cheapest API and one SLM to bound the k cap (exploratory) [`PROT` s.1].
- Fine-tuned models are evaluated with free generation like every other arm ("deployed generative system"); constrained decoding is a labelled robustness check for SLM arms only; one head-based fine-tune (Qwen, B77, N=1,000, R=3) is an exploratory bridge to Bertsch et al. [`PROT` s.6].
- Phi-4-mini enters the confirmatory candidate set only if it passes pilot stage 3 (gates G1-G5, G7; one fix cycle, otherwise dropped, `PROT` s.8). Stage 3 has **not** been run [PILOT status, section 16].

## 7. Tuning protocol and what is frozen when

### 7.1 Equal stated effort [`PROT` s.4, L-05, L-19]

- **Fine-tuning:** 4 trials per (model, task, N) on replicate 0, one per learning rate; each trial is one cosine schedule to E_max epochs (N=200: 10; N=1,000: 10; N=3,000: 5; S_min floor 200 steps) with validation evaluation at five checkpoints, early stopping patience 2; selection by validation accuracy (loss tie-break; BillSum: loss); chosen (lr, schedule length, stop step) is frozen. Validation size val(N) = min(1,000, N).
- **Learning-rate grid.** The planned grid was {5e-5, 1e-4, 2e-4, 4e-4}. The pilot selected 4e-4, the top of the grid, at both N (gate G5 failed; section 16), so the edge rule below extends the grid on the high side by one value (8e-4). Whether the campaign grid is then five values {5e-5 .. 8e-4} (5 trials) or four values {1e-4 .. 8e-4} (keeping "4 trials per system") and what cap applies if 8e-4 is again the edge: [TO DECIDE BEFORE FREEZE] D-06, owner: researcher after the repeat tuning run (`PC-04`).
- **ICL:** 4 values of k (BillSum: also the better of two retrievers) on validation; the prompt text is not tuned (limitation); one paraphrase sensitivity at N=1,000.
- **Edge rule (both arms):** if the chosen value lies on an edge of its grid, extend by one value on that side and repeat the tuning. A schedule-length edge (chosen E* equal to E_max, which happened at N=1,000 in the pilot) is flagged and reported; whether it triggers an extension of E_max: D-06 (`PC-05`).
- **Equality is in trial count, not compute** (a fine-tuning trial is a full training with five validation passes, about 1.5-2x a final run [`PROT` s.4, 06c s.4]). Tuning GPU-hours and dollars are charged in the cost model (section 12).
- **Pre-registered tuning-bar sensitivity** (not exploratory; L-19): on BANKING77, Qwen, replicate-0 pool, N=200 and N=1,000, a 12-configuration fine-tuning grid (4 learning rates x 3 rank/alpha or dropout/weight-decay pairs) and a matched ICL grid (k x retriever x one paraphrased instruction). Held-out slice and exact pairs: D-07. If best-of-12 beats best-of-4 by less than m = 3 points on the held-out slice, the 4-trial rule is reported as adequate; if by more, the campaign grids are widened at N <= 1,000 only (BillSum N=3,000 stays at cut 1) and the flip is reported. Budget: 0.7-2.0 GPU-h, under $1 API [`PROT` s.4, `FEAS` s.4]. Spiegelhalter et al. use 27 configurations x 5 folds; their App. E.2 finds a fixed configuration within 1-4 points of the per-task oracle at N >= 50 and under 2 points at N=200 [`spiegelhalter-2026-notes.md`]; this is cited in support, not as proof for our tasks.
- **Code prerequisite:** in-run early stopping replaces retraining per (lr, epochs) (A5-12). Tuning code is written and was exercised once on a GPU (pilot stage 2); BillSum gradient checkpointing is not yet written [`PROGRESS.md`, `PROT` s.9].

### 7.2 Two tags [`PROT` s.4, 06c C6]

| Tag | Planned | Contents frozen |
|---|---|---|
| **Protocol tag** | week 3 (Dec 6-12) | this file (all sections except the arm choices below), analysis scripts, commit hash, `final_configs.json` schema, test fingerprints, pinned `pip freeze`, grids, margins, R targets and the blinded-re-estimation procedure |
| **Arms tag** | end of week 4 (Dec 13-19) | `frozen_arms.json` and its sha256: per (task, N) the best fine-tuned configuration; per provider the best (k, retriever); the selected provider; kNN k; all chosen on **replicate-0 validation**, never on test and never re-selected for replicates 1..R |

No test row is read before the arms tag. `final_eval` refuses to run before both tags exist (section 5.3). Whether the API validation runs that choose the provider can be done by the arms tag, given that no API keys exist yet and API runs were scheduled last: [TO DECIDE BEFORE FREEZE] D-08, owner: researcher (`PC-06`).

**Selected provider rule** [`PROT` s.4, L-20]: the provider with the highest replicate-0 validation accuracy for its best retrieval-ICL configuration; ties within 1 standard error go to the cheaper provider. All providers' cost per correct are reported descriptively. The confirmatory CI is conditional on the frozen arms; selection uncertainty (noisy at N=200: 200 items, binomial SE about 3 points) lies outside it and is stated (L-18).

## 8. Replicates and blinded variance re-estimation

- **Targets:** BANKING77 R=8; BillSum R=5; no confirmatory cell with fewer than 5 replicates (a cell that ends below 5 is demoted to exploratory); replicate-major order [`STAT` s.4, L-06].
- **Blinded re-estimation.** After replicate 5 is complete for all cells, the variance components of the crossed model (item, replicate, item-by-replicate) are re-estimated **without any arm-effect estimate and without arm labels**, and the minimal detectable gap for m = 3 points is recomputed. For BANKING77 only, R may be extended from 8 to at most 10 if power for m is below 80% [`STAT` s.4]. BillSum is capped at R=5 (its extension would cost +12.5-33 GPU-h [`STAT` s.4, 06c C8]); its stated minimal detectable gaps are 0.6 / 1.0 / 1.9 ROUGE-L points for a replicate SD of the paired mean difference of 0.2 / 0.4 / 0.8 (all assumed, no data) [`STAT` s.3].
- **Who computes it, and what is hidden** (06c): [TO DECIDE BEFORE FREEZE] D-09, owner: researcher. The study has one researcher; the plan needs a script that outputs variance components with arm labels masked, run by a person (supervisor or colleague) who does not see the effect, or an explicit statement that this is done by the author with the effect estimate computed in a separate, later script.
- **Trigger ambiguity.** At N=200 the MDG under the pessimistic regime stays above m = 3 even at R=10 (4.4 points), so a literal "power below 80%" trigger always fires there; the rule applied to cells where R=10 could reach 80% power only: D-10 (`PC-11`).
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
- Precedence when a contrast is both Holm-significant and has its 95% CI inside (-m, m) on BANKING77, and when the unadjusted 95% CI excludes 0 but Holm p >= 0.05: D-11 (`PC-13`). Drafting proposal: Holm p decides (i) vs (iii); "(i) with a gap smaller than m" is reported as such.
- **Underpowered a priori** [`STAT` s.3, 06c R5]: a contrast is declared underpowered a priori, and is then reported as a bracket, iff the minimal detectable gap estimated from the replicate **validation-split** runs, fixed **before any test read**, exceeds m = 3 points (BANKING77) or 2 ROUGE-L points (BillSum). MDGs re-estimated after R=5 are descriptive only. Those validation-split runs need FT-vs-API discordance and replicate SD that do not exist yet (no API keys; `STAT` s.3 inputs are unobserved, A4-02): D-12, owner: researcher (`PC-07`). Note that under the stated BillSum SD range (0.2-0.8) the MDG never exceeds 2, so the BillSum arm of this rule is vacuous unless the observed SD is larger.
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
- **Human validation (conditional on HD-2, an annotator):** 60 bills x 3 systems = 180 summaries, blind and shuffled, stratified by length tercile and near-duplicate bin; claim-level faithfulness plus 1-5 acceptability; rater 1 (author) scores all, rater 2 a random 90; Krippendorff alpha with bill-level bootstrap CI; the third system and rater 2: D-13. A metric is secondary evidence only if Spearman rho >= 0.5 with lower bound >= 0.3. **Without the study:** every BillSum statement is "on ROUGE-L/BERTScore, exploratory", no faithfulness or "as good as" claim, no cost per successful summary (report cost per 1,000 outputs beside ROUGE-L) [`STAT` s.6, L-10].
- **Intervals:** primary crossed-model t interval; per-arm descriptive two-stage percentile bootstrap (replicates then items), 10,000 resamples, fixed seed; with R < 8 report the replicate range plus item bootstrap [`STAT` s.5].

## 12. Cost model

All cost statements are projections until measured. Prices below were read 2026-11-21 and **must be re-read on the day of the API runs**; the price date is stored with every result [`FEAS` s.1, `PROGRESS.md` 05:50 entry].

- **API:** cost per query from provider-billed input (uncached and cached) and output tokens, **including reasoning tokens**, whole prompt including retrieved shots, retries billed; never character counts. Listed rates (USD per 1M tokens, input / output, standard text, prompt-size limits as in `FEAS` s.1): Haiku 5.5 0.10 / 0.50; GPT-6 Luna 0.10 / 0.50; Gemini 3.1 Flash-Lite 0.25 / 1.50 (medium confidence, truncated page); batch 50%. Tokens per request [M, 1,000-row train blocks]: BANKING77 k=32 about 1,220; BillSum k=3 about 9,300 [`FEAS` s.1].
- **Self-hosted:** C_train = all tuning trials plus all finals, GPU-hours x an imputed hourly rate in the band **$3.6-6.0/h** (scenario band, no NVL price exists; $2.3/h spot as a sensitivity row; the university allowance is not billed in dollars, HD-3 open) [`STAT` s.7, `FEAS` s.3]; inference per query = measured GPU-seconds / utilisation u in {1, 0.5, 0.2}; measured throughput replaces FLOP-arithmetic ranges (throughput benchmark pending, A5-11).
- **Cost per correct** = (C_train + sum c_q) / sum correct_q at stated volume Q in 10^3..10^7. **Marginal break-even** Q* = C_train / (c_API - c_FT/u), valid only if c_API > c_FT/u; per-correct form Q* = C_train / (a_FT c_API/a_API - c_FT/u). If the denominator's CI contains 0: Fieller interval or P(denominator <= 0) and "no break-even in range"; never a percentile CI of an unbounded ratio. Items are resampled as (cost, correct) pairs, replicates for C_train and accuracy; fixed costs added after resampling [`STAT` s.7].
- **Labelling axis:** total = N c_L + C_train(N) + Q c_q; c_L band 0.10 / 0.50 / 2.00 USD (assumed). For BillSum the labels are sunk (CRS summaries), so this axis is hypothetical [`STAT` s.7].
- **BillSum:** cost per 1,000 outputs beside ROUGE-L; cost per correct is not defined for summarisation (L-10).
- API pricing mode for the base budget (sync vs batch): D-14 (`PC-02`). The sensitivity values in `STAT` s.7 and the budget figures in `FEAS` s.2 are inputs for planning, not results.

## 13. Exclusion and failure rules

1. **Nothing is excluded for being inconvenient.** Failed runs are kept in the results and listed; the OLMo-3 DoRA NaN of the old pilot is the model case [old PREREG, `PROT` s.7].
2. **Training health.** Non-finite loss fails the run loudly (no silent continue). A run is **invalid** if loss is non-finite or gold-label parsing fails the self-parse test; it is re-run with the same pool and seeds, the failed attempt is logged.
3. **Examples above `max_length`:** test items are never truncated; training examples above the cap are dropped only if they are under 0.1% of the pool, logged (BillSum; `tokenize_example` raises rather than truncates) [`PROT` s.7].
4. **Output validity:** invalid output counts as wrong; the invalid rate is a reported outcome, never a reason to drop items. API refusals, empty and truncated outputs count as invalid.
5. **API truncation:** truncation rate above 1% in the smoke test stops the run; truncation counts as invalid [`PROT` s.5].
6. **Replicate failure:** a replicate that fails twice for non-infrastructure reasons stays in the record; whether it is replaced by the next seed or the cell is analysed with fewer replicates (never below 5, otherwise demoted to exploratory): D-15 (`PC-15`).
7. **Provider drift:** a changed `model_id_returned` mid-run voids that provider's batch; all confirmatory cells of one provider run within 72 hours; one mode (sync or batch) per confirmatory run, agreement checked on 100 items [`PROT` s.5].
8. **Pilot data are never pooled** with confirmatory data and never used in the confirmatory model.
9. **Cell cuts** forced by compute (section 15) are applied to whole cells before the arms tag where possible; a cut that removes a confirmatory contrast is reported as "not run", not as a result.

## 14. API protocol (set by a live test, not by documentation) [`PROT` s.5]

- **Model ids to be pinned:** `claude-haiku-5-5`, `gpt-6-luna`, `gemini-3.1-flash-lite` as listed in `FEAS` s.1. Haiku 5.5 is described as a "pinned snapshot" by the provider; Luna has only an undated default id. **The exact id string, snapshot date, SDK version and price page version are recorded on the first live call and written into the protocol tag or its addendum** (D-16, owner: researcher). Store `model_id_returned`, usage fields, price date and run date for every response. Provider terms version and date are stored in released result files [`DATA` s.2].
- **Sampling:** send `temperature=0` only where a 50-item test shows it is accepted. Otherwise omit it and label the arm **stochastic** (the Haiku 5.5 migration page says to omit `temperature`, `top_p`, `top_k`; the exact failure behaviour is unconfirmed, so the live test decides) [`PROT` s.5, `FEAS` s.1]. For stochastic arms measure the flip rate on a 10% repeat x2; if above 2%, the arm's accuracy is the **mean of two full runs** (pre-priced: about 49,000 extra BANKING77 and about 19,000 extra BillSum requests) [`PROT` s.5]. How a two-run mean enters the crossed model (fractional correctness, replicates unchanged): D-17 (`PC-16`). `TEMPERATURE` is a per-arm field.
- **Reasoning:** lowest accepted setting. *Reasoning-off* is claimed only if reported reasoning tokens are 0 for every test item; otherwise the arm is labelled *reasoning-minimal* with billed reasoning tokens in the cost. Gemini thinking cannot be fully off [`FEAS` s.1]. Arms with different reasoning settings are never compared silently.
- **Output cap:** if reasoning counts against the cap, `max_output_tokens = 32 + max(64, p99.9 observed reasoning tokens)`, else 32; BillSum 1,024 for every arm.
- **Staging:** S0 dry-run cost gate; S1 smoke 20 items; S2 200 validation items; S3 validation tuning of k; S4 test after both tags. Live tests need the researcher's own keys and a spending cap (HD-1, not yet answered; no keys exist, `PROGRESS.md`). The pilot's stage A (live tests, 50 validation items per provider) **has not run**.
- **All arms are "deployed generative systems"**: free generation, same parser; paid tiers only; no training on API outputs.

## 15. Deviations policy

1. **Before the protocol tag:** this draft may change freely; keep the git history.
2. **After the protocol tag and before the arms tag:** any change is a deviation entry (what, why, date, whether any test row had been read: must be no) appended to `docs/PREREGISTRATION_DEVIATIONS.md` (to be created at the tag).
3. **After the arms tag:** changes to confirmatory contrasts, family, estimand, margins, arm choices or test sets are not allowed without a dated deviation entry that states the reason and whether results had been seen; analyses affected are reported both ways where possible.
4. **Permitted without a deviation entry:** infrastructure re-runs with identical configuration (section 5.3 rule 4); cuts on the cut ladder (`FEAS` s.3) that remove only exploratory cells (LEDGAR, OLMo, random-k); go/no-go outcomes at week 2 (`PROT` and L-16: Qwen BillSum N=3,000 final <= 1.2 h and B77 N=3,000 <= 12 min keeps the design with cut 1; 1.2-2.0 h adds R=3 at BillSum N=3,000, which makes that cell exploratory; > 2.0 h drops BillSum N=3,000 and makes BillSum Qwen-only). The cut ladder changes confirmatory cells at steps 4-6 (BillSum N=3,000 at R=3 or dropped), so those steps **are** deviations from the 6-contrast family and are logged as such, with the family reported as 5 contrasts.
5. **Reporting:** the paper has a deviations table; unregistered analyses are labelled exploratory.

## 16. Evidence status and the pilot (exploratory; validation split only; no confirmatory result exists)

**What exists.**
- **Pilot 1** (`results/pilot-2026-11-21/`): old prompt (no label list for the fine-tuned arm), fixed pool, under-trained at small N, N in {50, 200, 1,000, 5,000}; exploratory, superseded, not used for any number in this file [`FOUND` header].
- **Pilot stage 2** (`results/pilot-stage2-2026-11-22/`, code tag `pilot-stage2-code`, commit `f533926`): Qwen3-4B-Instruct-2507, BANKING77, LoRA, corrected shared prompt, tuned procedure, 1,000-row validation split, test never read, one H200 NVL. Only the pieces below were run: stage 2. Stages 1 (loss diagnostics), 3 (Phi-4-mini), 4 (BillSum memory/throughput) and stage A (API live tests) have **not** been run; the fine-tuning arm for Phi and BillSum, and the whole API protocol, are therefore **unverified**.

**Pilot numbers [PILOT, validation only, 2 replicates plus 1 random-pool final; not evidence for any hypothesis]** (`results.jsonl`, `frozen_arms.json`; 4 learning rates {5e-5, 1e-4, 2e-4, 4e-4} on replicate 0):

| Quantity | N=200 (stratified) | N=1,000 (random) |
|---|---|---|
| Replicate-0 selection accuracy at chosen lr (selection rows, optimistic) | 0.785 (200 val rows) | 0.883 (1,000 val rows) |
| Best-checkpoint val accuracy for lr 5e-5 / 1e-4 / 2e-4 / 4e-4 | 0.73 / 0.735 / 0.76 / 0.785 | 0.831 / 0.849 / 0.862 / 0.883 |
| Chosen lr; stop step of schedule | 4e-4 (**top of grid**); step 160 of 200 (E*=8) | 4e-4 (**top of grid**); step 630 of 630 (E*=10 = E_max) |
| Fine-tuned finals, replicates 1 / 2 (1,000 val rows) | 0.776 / 0.769 | 0.869 / 0.872 |
| kNN (bge-small, same pools), replicates 1 / 2 | 0.730 / 0.713 (k=1) | 0.840 / 0.844 (k=5) |
| Random-pool N=200 final, replicate 1 (FT / kNN) | 0.745 / 0.681 | n/a |
| Invalid-output rate (mean) | 0.4% | 0.1% |
| Training / tuning GPU time per final / per tuning | 71 s / 457 s | 433 s / 2,534 s |

Gate report (script `run_stage2.py`): G1 pass, G2 pass (FT minus kNN +5.1 points at N=200), G3 pass (+2.9 at N=1,000), G4 pass (+9.8 points from N=200 to 1,000), **G5 FAIL**, G6 pass (replicate spread 0.7 and 0.3 points), G7 pass (time-band part not checked); overall **FAIL on G5 only** [`PROGRESS.md`, recomputed from the files]. A failed or passed pilot is not evidence for H1-H4 [`PROT` s.8].

**What changed in this pre-registration because of the pilot.**
1. **Learning-rate grid extended on the high side** (8e-4) by the edge rule, for both arms, and the tuning repeated (rerun and five finals planned, about 35-45 GPU-min; `PROGRESS.md` decision of 05:10). The grid size question is D-06. Validation accuracy rose monotonically with lr at both N, so the frozen lr of the pilot is an artefact of the grid, not a tuned optimum.
2. **The edge rule now also names the schedule length:** at N=1,000 the selected number of epochs equals E_max; this is flagged and D-06 decides whether E_max is extended.
3. **kNN:** k=1 was selected at N=200 (the lowest possible value, so the edge rule cannot extend it) and k=5 at N=1,000; k=16 was never at the edge. kNN is as strong as fine-tuning within 3-5 points on validation, which is why it stays a first-class comparator (secondary contrasts) [`PROT` s.2].
4. **Gate implementation deviates from the protocol text:** the gate script uses all 1,000 validation rows (the same rows partly used for selection at N=1,000, and including the 200 selection rows at N=200) and one margin of 5 points for both G2 and G3; the protocol (06c R1, L-20) requires non-selection rows (rows 201-1,000 at N=200; a fresh 500-row slice at N=1,000) and a 2-point tolerance for G3. The pilot result is unchanged in direction (G3 margin +2.9 passes either tolerance) but has **not been recomputed on non-selection rows** (`PC-17`). The repeat must use the protocol's definition.
5. **Measured timings** replace nothing in `FEAS` yet; the pilot supplies measured numbers for B77 at N=200 and 1,000 only. The week-2 go/no-go still needs BillSum and B77 N=3,000 measurements.
6. **No change** to N levels, R, family, test sizes or the estimand.

**Open pilot items before the protocol tag:** repeat of stage 2 with the extended grid; stage 3 (Phi); stage 4 (BillSum memory, needs gradient-checkpointing code); stage A (needs keys); Stage-0 exit items (tagged commit, `pip freeze`, test fingerprint, `test_reads.jsonl`, `final_eval`) [`PROT` s.9]. If G1-G7 cannot be passed after two fix cycles of about 2 GPU-h each, the outcome is *re-scope* (head-based fine-tuning, fine-tuning only at N >= 1,000, or a kNN/ICL/API-cost paper) and this pre-registration is rewritten, not tagged [`PROT` s.8].

## 17. Not yet decided: `[TO DECIDE BEFORE FREEZE]`

**Update 2026-11-22: the decisions for these items are written in `docs/foundation/PREREG_DECISIONS.md` (taken by the assistant under the researcher's delegation; the researcher can overrule any of them; D-13, D-16, D-18, D-19, D-21 stay with the researcher). The table below is the original list of open questions.**

All items need a written decision before the protocol tag unless marked "arms tag". Owner is the person who must decide; "researcher" is the PhD student.

| ID | Item | Why open | Owner | Needed by |
|---|---|---|---|---|
| D-01 | Confirm the direction and wording of H1-H4 (section 3) | Drafted from the documents' motivation, not from data | researcher | protocol tag |
| D-02 | BillSum validation size for API tuning (1,000-capped vs 300); affects request counts and $22-39 [`FEAS` s.2, `PC-01`] | documents disagree | researcher | protocol tag |
| D-03 | Seed and fingerprint for the 1,000 seeded BillSum test items and 3,000 LEDGAR items; the freeze script covers BANKING77 only | not written | researcher (code) | protocol tag |
| D-04 | Re-run rule after a first test read (infrastructure-only proposal, section 5.3) | not in the documents | researcher | protocol tag |
| D-05 | Which fine-tuned model enters the confirmatory contrast (Qwen vs Phi, chosen on replicate-0 validation, tie rule) and whether Phi is in the family at all | documents say "best-validation fine-tuned configuration" without naming the candidate set (`PC-10`) | researcher | arms tag |
| D-06 | LR grid after the pilot's G5 failure: 4 values (shifted) or 5 values; cap on repeated edge extension; whether E_max at the edge is extended [`PC-04`, `PC-05`] | G5 failed; repeat not run | researcher, after the repeat | protocol tag |
| D-07 | Tuning-bar sensitivity details: pairs of (rank/alpha or dropout/weight-decay), held-out slice definition | L-19 gives the shape only | researcher | protocol tag |
| D-08 | Can API validation (provider choice, k) run before the arms tag when no keys exist and API runs were scheduled last? If not, move the arms tag or freeze provider choice differently [`PC-06`] | HD-1 unanswered; schedule vs progress-log decision | researcher | protocol tag |
| D-09 | Who computes the blinded variance re-estimation and what is masked | single-researcher study | researcher (supervisor?) | protocol tag |
| D-10 | Trigger for the R extension (B77) given that N=200 can never reach 80% power for m [`PC-11`] | rule would always fire at N=200 | researcher | protocol tag |
| D-11 | Precedence of outcome labels (i) vs (ii), and (i) vs unadjusted CI [`PC-13`] | unspecified | researcher | protocol tag |
| D-12 | "Underpowered a priori" validation-split MDG: needs FT-vs-API discordance and replicate SD (no API keys, HD-1); and the 2-point BillSum threshold is vacuous under the assumed SD [`PC-07`] | inputs unobserved | researcher | protocol tag (or declare the rule unused) |
| D-13 | Human faithfulness study: annotator available (HD-2), third system, second rater, fee | no annotator | researcher | protocol tag |
| D-14 | Base API budget mode: sync (meets the 72 h rule) or batch (the cheaper mixed mode in `FEAS` s.2) [`PC-02`] | documents conflict | researcher | protocol tag |
| D-15 | Replicate-failure replacement rule | not specified | researcher | protocol tag |
| D-16 | Exact API model id strings, snapshot dates, SDK versions, price-page versions | need live calls | researcher (keys) | arms tag |
| D-17 | How a "stochastic" arm's two-run mean enters the crossed model [`PC-16`] | not specified | researcher (statistics) | protocol tag |
| D-18 | Author, affiliation, funding, data-availability statement, registry (OSF or none) | personal | researcher | protocol tag |
| D-19 | Credit-to-dollar value and carry-over of the GPU allowance (HD-3); imputed GPU rate | only visible in the portal | researcher | before cost analysis |
| D-20 | Whether the independent re-review after the pilot and after the section-9 text has happened before tagging (`NOVELTY_AND_RED_TEAM.md` gate 8 not passed) | no re-review yet | researcher / reviewer | protocol tag |
| D-21 | Legal/terms: incorporated provider policies not fully read (I-18); written confirmation from dataset authors optional | partly unread | researcher | before release, not before tag |

## Appendix A. Documents each section copies or follows

| Section | Source |
|---|---|
| 2-3 | `FOUND` s.1-4, `spiegelhalter-2026-notes.md` |
| 4-5 | `PROT` s.1-3, s.7, s.9; `DATA` s.1-4; L-02, L-04, L-07, L-08 |
| 6-7 | `PROT` s.1, s.4, s.6, s.8; L-05, L-09, L-17, L-19, L-20 |
| 8-9 | `STAT` s.1-5; L-03, L-06, L-15, L-18, L-20; 06c R5, R6 |
| 11-12 | `STAT` s.5-7; `FEAS` s.1-3; L-10 |
| 13-14 | `PROT` s.5-7; `FEAS` s.5 |
| 16 | `results/pilot-stage2-2026-11-22/`; `PROGRESS.md`; `PROT` s.8 |

## Appendix B. Superseded content of the draft of 2026-11-21

N = 50 and N = 5,000 (L-02); five replicates for all cells (L-06); McNemar and paired bootstrap over items, two contrasts per task, Holm over tasks x 2 (replaced by the crossed model and six contrasts, L-03, L-15); "pool sampling pending" (decided, L-04); LEDGAR conditional on a licence check with CLINC150 fallback (LEDGAR is exploratory, ids/labels/metrics only, L-07, L-08); Gemini "optional" (all three APIs run; the selected provider carries R=8/5, the others R=3); k_max 5 for summarisation (now 3); "reasoning off" for Luna as fixed (now set by a live test); BillSum "CC0 per card" (replaced by the statutory wording); test sets "at least 1,500 items where available" (B77 3,080, BillSum 1,000 seeded, fixed by power analysis).
