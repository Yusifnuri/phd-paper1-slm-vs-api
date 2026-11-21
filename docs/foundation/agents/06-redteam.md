# A6: Red-team review (2026-11-21)

Labels: **[V]** I checked it myself today (primary source or arithmetic), **[I]** my inference, **[U]** unverified. Cost figures marked "my arithmetic" use A5's prices and A3/A5 token counts, not new measurements. A1 and A3 Round 2 sections did not exist when I finished (see end).

Verdict: **reject as designed.** The pieces are individually careful, but the five drafts describe different experiments, the confirmatory contrasts sit where earlier work says the answer is already known, and nothing about the fine-tuned arm has been shown to work under the corrected prompt.

## 1. Cross-draft contradictions (reconcile before anything runs)

| # | Item | A3 | A4 | A5 | Recommended value | Consequence |
|---|---|---|---|---|---|---|
| 1 | B77 test items | 3,080 | simulates 1,000 and 3,080 | 1,000 | **3,080** (A4: 3 pt at N=1000, R=8, power .84/.73 at 1,000 vs 1.0/.99 at 3,080) | FT/SLM cost is trivial; API B77 volume x3.08 versus A5 (b). Offset by item 5 |
| 2 | BillSum eval items | 1,000 | analytic only | 500 | **1,000** | Haiku/Luna k=3: about $1.1-1.4 per 1,000 requests, Gemini $2.5-3.5; doubling from 500 adds roughly $20-35 per all-provider replicate set (my arithmetic) |
| 3 | LEDGAR test | 3,000 seeded | family of 6 contrasts | 1,000, only in (c) | Drop LEDGAR from the confirmatory family; if kept, 3,000 and exploratory | A4's Holm family is 6 vs 4 depending on this |
| 4 | LEDGAR k | k_max 32 (about 5k tokens) | - | k=8 (1.77k tokens) | **k_max 8** | A3's 100k requests per provider at 5k tokens is 500M tokens: about $50 each for Haiku/Luna and $125 Gemini, so about $225 standard. A5's pricing is 2.8x lower. A3's matrix cannot be afforded |
| 5 | Replicates, FT | 5 min / 8 rec.; BillSum 3 / 5 rec. | 8 B77; at least 5 others; type I liberal at R=3 | 5; 3 at N=5000 | **B77 R=8; BillSum and LEDGAR R=5; no confirmatory cell at R=3** | A5 (b) "R=3 at N=5000" falls in A4's invalid regime. BillSum N=5000 at R=5, 2 models is 12-35 GPU-h (A5's per-run times) |
| 6 | N levels | 200, 1000, 5000; drops N<C; 77 and 9,003 exploratory | N=50 in ICL only; confirmatory 1000 and 5000 | 50, 200, 1000, 5000 on B77 | Confirmatory 200, 1000, 3000 (see F4, F6); N=77 exploratory; delete N=50 FT | N=50 FT is under-trained by construction and uncosted by A3 |
| 7 | Where tuning happens | rep 0, once per (model, technique, N) incl. N=5000 (<=10 epochs) | "best on validation" | g LR trials at N<=1000 only, none at N=5000 | A3's rule, but costed | If FT is untuned at 5000 while ICL k is tuned, the comparison is asymmetric (F5) |
| 8 | Number of trainings | 78 finals + 216 tuning trials (min); 126 + 216 (rec.) | - | 60 / 112 / 237 including tuning | One table, one source | A5 is about 2.6x lower than A3's min at the same scope. A3 and A5 disagree on whether `--tune`/`lr_grid` exist (A3 says verified in code; A5-02 says no `train.tune` module). Different branches [I] |
| 9 | Task set, "minimum viable" | 3 tasks (LEDGAR conditional) | 6 or 4 contrasts | 2 tasks | **2 tasks (B77, BillSum) + LEDGAR as exploratory** | Align A3 matrix and A4 family |
| 10 | GPU $/h band | - | $2.3-5.4 | $3.6-6.0 (no NVL price) | A5's band; A4 to cite it | Break-even Q* changes about 1.7x between the two |
| 11 | API random-k arm | all N x R | not in the family | priced | **Cut to one N, R=3** | Largest single saving, about 30-40% of API calls (item 12) |
| 12 | API requests per provider (B77) | about 110k; 330k total | needs R=8 | 52k incl. BillSum (b) | Confirmatory provider chosen on validation: R=8, N in {1000, 3000}, 3,080 items = about 49k requests. Other providers: R=3 | My arithmetic at about 1.3k tokens: 64M tokens, under $10 for Haiku/Luna. Full A3 grid at R=8 would be about 170k per provider and about $100-110 standard for all three, before the Haiku tokenizer's +0-35% [U] |
| 13 | BillSum output cap | 1,024 all arms | - | API template 2,048 | 1,024; check Gemini thinking inside cap | Truncation asymmetry |
| 14 | Pool sampling | class-stratified primary | estimand "random labelled pool" | - | **Decide** (F6) | The estimand and the design disagree |
| 15 | Wall-clock engineering | long unbuilt-code list (A3-04) | needs crossed-model code | 2.5 weeks [U] for (b) | Treat as >=4 weeks (F9) | Eats half of the two months |

GPU consequence of A3's own tuning rule (my arithmetic from A5 §2.2): BillSum N=5000 tuning at 2 lr x 3 epoch settings = 6 trials x (1.2-3.3 h Phi + 1.3-3.5 h Qwen) = about 15-41 GPU-h, plus finals. BillSum alone is then about 35-85 GPU-h against A5's 60-75 productive hours for the whole study. A5's (b) "17-47 h" omits this.

## 2. Findings (problem / threat / evidence / fix / feasible? / owner)

**F1 (CRITICAL). Novelty: the confirmatory grid lies beyond the known crossover.**
Problem: Spiegelhalter et al. (Qwen3 0.6/1.7/4B, LoRA+HPO, three ICL strategies, cost analysis, budgets to 200) report crossover near 50 labels [V, abstract]; Pecher about 100; Bertsch covers B77/CLINC retrieval vs LoRA. A4's confirmatory family tests N=1000 and 5000 only; N=200 and below are "exploratory, no p-values". Threat: the paper's headline (a crossover) is not confirmatory, and the confirmatory points are where LoRA is expected to win, so a reviewer calls it a replication with an API arm. Evidence: A4 §2 Multiplicity; A1 §2. A search result I saw says the crossover came earlier for 0.6B/1.7B than 4B and that ICL saturates by 25-50 examples [V, search snippet]; so for 4B it is later than 50, near our N=200. Fix: reposition the primary claim as decision-grade, not crossover: at a fixed labelled pool, how do LoRA-4B, retrieval-ICL on cheap APIs, and kNN compare in accuracy and total cost with tuning charged. Make N=200 confirmatory (7-contrast Holm, or 3 N x 2 tasks). Feasible: yes, design change only. Owner: A1 (rewrite gap statement; read full Spiegelhalter, which A1 could not), A4 (move N=200 into the family, re-run power at the 200 point).

**F2 (CRITICAL). The honest contribution is narrower than the drafts say.**
Strongest defensible: (i) equal-pool comparison against API retrieval-ICL and kNN, (ii) a break-even with the tuning sweep charged, (iii) a BillSum label-budget curve, (iv) fixing prior flaws (test-set checkpointing: confirmed [V] in Bertsch App. E; 250-item tests). Likely headline is mixed: kNN already equals the pilot FT at N=5,000 (A3 §2), so on B77 the LLM may add little. Say so; do not promise "FT beats API". "First" is not defensible (A1 ran about 9 queries). Fix: A1 reruns search on ACL Anthology, Semantic Scholar, OpenReview directly, and again before submission. Feasible: yes. Owner: A1.

**F3 (CRITICAL). The FT arm is unproven under the corrected prompt.**
Evidence: I-01/I-02 are FIXED-UNVERIFIED; pilot N=200 was 0.17 accuracy with 74% invalid; Phi first-step loss 18.88 and OLMo DoRA NaN are unexplained (I-14, A3 §11 items 1-3). Threat: every FT-vs-API gap may be a recipe artefact, the exact B2 failure. Fix: a gated 3-hour GPU pilot (N=200 and 1000, B77, Qwen, new prompt, tuned) with pass criteria written first (invalid <5%, accuracy above kNN at N=200 minus 5 pts); no campaign before it. Feasible: yes, about 3 GPU-h. Owner: A3 (criteria, run), A5 (include in budget).

**F4 (HIGH). Estimand mismatch: stratified pools vs "random labelled pool".**
A3 makes class-stratified primary; A4's estimand is expectation over random pools. Stratified at LEDGAR N=5000 is impossible (min train class 23). Threat: reviewers cannot tell what N means; a uniform pool prior changes accuracy and favours kNN/retrieval differently from FT. Fix: pick one. Recommended: random pools primary at N>=1000 (natural prior, practitioner reality), stratified only for N=200 where coverage fails (50.5% absent at N=50, 7.5% at 200 for B77 [A3]), flagged as a separate stated condition; the A3 random-vs-stratified check at R=3 becomes a reported table. Owner: A3 decides, A4 rewrites the estimand to match.

**F5 (HIGH). Tuning symmetry and validation-set information.**
Problem: 1,000 validation labels vs N=50/200 pool; FT gets about 12 trials, ICL 4 (A3 §5); A5 skips N=5000 FT tuning. The validation labels would be an enormous help to retrieval at N=200 (1,000 extra neighbours) but they are not in the pool. Threat: "true few-shot" critique, and unequal selection effort inflates one arm. Fix: validation size = min(1,000, N) at N<=200 as primary, not a sensitivity; equal trial counts (4 vs 4, or FT lr x 3 and ICL k x 4 with the FT grid shortened to lr only at N=5000); report tuning dollars in the cost. Feasible: yes, reduces cost. Owner: A3.

**F6 (HIGH). Pool overlap invalidates "independent replicates" at N=5000.**
Two N=5000 pools from 9,003 rows share about 55% (A3-07); A4's crossed model and all power numbers assume independent replicates. Threat: R=8 behaves like fewer effective replicates; CIs too narrow or, if the estimand is the 9,003-row population, the variance estimator is biased. Fix: replace 5000 with 3000 (overlap about 33%; still a 4x step from 1000), or use B77 train+val minus a 500-item validation and say the estimand is conditional on that finite set; A4 re-simulates with overlapping pools. Feasible: yes. Owner: A4 (simulation with overlap), A3 (N).

**F7 (HIGH). A4's power rests on two unobserved quantities, then is frozen into R.**
FT-vs-API discordance and pool variance are unobserved (A4 §1.1). At R=5 pessimistic, 3 pts, first-step alpha .0083: power .41 (A4 table). Threat: likely a non-significant, uninformative confirmatory result for LEDGAR/BillSum; null misread as no difference. Fix: do the 5-replicate validation-split API runs A4 itself proposes (A4-02) before fixing R; add a pre-registered rule "if the interval is wider than the margin, report inconclusive". BillSum power is only analytic (0.8 pt assumes SD 0.4 ROUGE-L, unknown) and A3's R=3 gives t(2)=9.9 critical values: BillSum cannot be a confirmatory contrast below R=5. Feasible: about $5-15 API plus 1-2 GPU-h. Owner: A4.

**F8 (HIGH). Fairness details that a reviewer will attack.**
(a) Haiku 5.5 takes no `temperature`: migration page says "omit temperature, top_p, and top_k" [V]. I could not find the words "returns a 400" for sampling parameters on that page (it says 400 for thinking budgets and prefill); A5's "settled" is slightly overstated. Nothing on that page shows `thinking: {"type":"disabled"}`; it describes adaptive thinking on by default, with lower effort that "can skip thinking" [V, partial]. If thinking cannot be fully disabled for Haiku, reasoning tokens count against `max_tokens` (page: thinking counts toward `max_tokens`) and the 32-token class cap can truncate. (b) Gemini thinking cannot be off; hidden cost and nondeterminism. (c) The 2% flip-rate rule "average two runs" doubles API cost only after the fact; pre-price it. (d) Free-generation vs head (Bertsch): report constrained decoding for SLM only, so API and SLM are not on the same footing at N<=200; say the comparison is "deployed generative system". (e) kNN is a first-class arm, not an appendix. Fix: a 50-item live test per provider (tool-call results recorded) before any plan is frozen. Feasible: yes, needs the user's keys. Owner: A5 (live test, cost with thinking), A3 (protocol text).

**F9 (HIGH). Engineering is not two weeks and some results silently depend on unbuilt code.**
A3-04 lists missing: SLM ICL harness, shared few-shot rendering, kNN, BillSum and LEDGAR loaders and training, test-read ledger, API repeats, replicate id, embedder revision pin. A4 needs a crossed-model/Fieller package (A4-04). A5 needs a vLLM evaluation harness with LoRA serving not re-tested (A5 §2.3), BillSum gradient checkpointing, and a 1-hour throughput benchmark (A5-11) because throughput is FLOP arithmetic, 12-30% MFU assumed. Human-eval tooling is also unbuilt. Threat: weeks 1-4 are code; 13.3 GPU-h per week leaves about 6 compute weeks, and writing has no slot. Fix: a dependency-ordered build plan with owners and dates; a go/no-go at the end of week 2 (F3 pilot + throughput benchmark). Cut OLMo and LEDGAR first. Feasible: only if scope is cut as in section 1. Owner: A5 (schedule, with A3's code list); A3 (which components are blocking).

**F10 (HIGH). Summarisation validity and budget.**
ROUGE/BERTScore alone are not defensible (A4 §4); no factual metric is validated on legislative text. A4's human evaluation (400 ratings, about $2-5k) is 10-25x the $50-200 budget and its "8 min per summary" is optimistic for a 2k-token bill with claim-level checks [I]. BillSum retrieval also copies: 297 test bills have cosine >=0.95 to a train bill (A2), so retrieval-ICL may reproduce a prior CRS summary. Output cap 1,024 vs 200-token median references makes ROUGE F1 depend on verbosity. Fix: (1) report all BillSum results stratified by near-duplicate bin and with length-matched ROUGE; (2) replace the 400-rating study with 60 bills x 3 systems rated by the author and one volunteer annotator on faithfulness only (about 20-30 h of unpaid time, otherwise a small paid pilot under $300), used to validate AlignScore/an LLM judge, not to rank; (3) if that is not available, state the summarisation claim as ROUGE-only and exploratory. Cost per successful summary dropped; report cost per 1,000 summaries beside quality. Feasible: yes at reduced scope. Owner: A4 (protocol, power for a validation study), A5 (budget line).

**F11 (MEDIUM). Cost model omits labelling and uses imputed prices.**
GPU rate is imputed (no NVL price; A4 and A5 bands differ); no utilisation or serving overhead; batch API (24 h) compared with an always-on GPU. Labelled-pool cost (50 to 5,000 labels) is excluded: it is common to both arms at a given N but dominates the N-axis economics. Fix: report break-even Q* over the rate band, utilisation 1/0.5/0.2, batch vs sync, and an extra axis "cost per label x N". Feasible: yes. Owner: A5, A4.

**F12 (MEDIUM). Release and legal plan.**
Confirmed: BillSum card says cc0-1.0 in metadata only, licensing section empty, CC0 attached to GovInfo [V]; LREC lists MIT for LEDGAR [V]. A2's reading is right. Gaps: LEDGAR contracts are third-party exhibits (low but nonzero risk); an adapter trained on 50k clauses can reproduce text; provider texts read are three pages, incorporated policies (usage policies, Google additional terms) not read, confidence "M"; A2-06 (Google paid tier) is not reflected in A5's budget. Fix: release LEDGAR nothing but ids, labels, metrics; release adapters for B77 and BillSum only; release API-generated predictions with a no-training notice; A5 states paid tier and billing enabled; A2 reads the incorporated policies. SAMSum stays out. Feasible: yes. Owner: A2.

**F13 (MEDIUM). Reproducibility.** Haiku has no sampling control; pilot logged `git_commit: unknown`; pinned requirements never installed on the server (A5 §5); three branches unmerged. Fix: merge to one tagged commit, lock `pip freeze`, run all confirmatory API cells per provider inside 72 h. Owner: A3/A5.

**F14 (LOW). Minor.** Max-token inconsistency across drafts for BillSum (A2 4.9k test, A5 5.6k train, A3 7.4k mixed); `?` normaliser; per-class results; contamination probe on GPU. Owner: A3.

## 3. Independent spot-checks

| # | Claim | Result | My quote (<=15 words) and URL |
|---|---|---|---|
| a | Bertsch App. E picks checkpoint on test | **Confirmed** (quote relayed by the fetch tool from the HTML text) | "choosing the checkpoint with the highest test accuracy" https://arxiv.org/html/2405.00200 |
| b | Haiku 5.5 rejects `temperature` | **Partly confirmed**: guidance to omit; "400" for sampling not seen; `thinking: disabled` not seen | "omit temperature, top_p, and top_k" https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide |
| c | LREC gives MIT for LEDGAR | **Confirmed** | "MIT License" (LEDGAR entry, License field) https://lrec2020.lrec-conf.org/en/shared-lrs/ |
| d | BillSum CC0 wording | **Confirmed**: metadata cc0-1.0; licensing section empty; GovInfo line | "[More Information Needed]"; "under CC0-1.0 license" https://huggingface.co/datasets/FiscalNote/billsum |
| e | Pecher includes GPT-4o-mini | **Confirmed** (prompting/ICL only) | "GPT-4 (4o-mini-2024-07-18 version)" https://arxiv.org/html/2402.12819v4 |
| f | Spiegelhalter abstract | **Confirmed**, authors Spiegelhalter, Franke, Hutter | "Approximately 50 labeled samples suffice for fine-tuning to overtake all ICL strategies" https://neurips.cc/virtual/2026/160515 |

## 4. Quality gates

| Gate | Result | Missing evidence |
|---|---|---|
| 1 Literature validity | PARTIAL | Spiegelhalter full text; search of ACL Anthology/S2/OpenReview; several sources model-summarised (A1-05) |
| 2 Dataset validity | PARTIAL | Licences adequately evidenced; leakage stratification not implemented; B77 label noise unaudited; BillSum near-duplicate effect on retrieval |
| 3 Novelty | FAIL | Claim restricted to a conjunction; Spiegelhalter unread; N grid beyond known crossover |
| 4 Methodological validity | FAIL | No GPU evidence the FT arm works (F3); estimand mismatch (F4); tuning asymmetry (F5); unreconciled design (section 1) |
| 5 Statistical validity | PARTIAL | Sound method, but unobserved parameters, N=5000 overlap, crossover not in family, BillSum power unsimulated |
| 6 Economic feasibility | FAIL | A3's design unaffordable (LEDGAR k=32, R=8, random-k); BillSum GPU-hours undercounted; human eval unfunded; throughput not measured |
| 7 Reproducibility | PARTIAL | Code unmerged, versions unpinned, API nondeterminism unquantified |
| 8 Adversarial review | PARTIAL | This report; cannot pass itself; needs re-review after fixes and Round 2 |

## 5. Top ten, ranked

1. CRITICAL: F1 confirmatory grid beyond the crossover (A1, A4).
2. CRITICAL: F3 FT arm unproven under the fixed prompt (A3, A5).
3. CRITICAL: F2 contribution narrower than claimed; Spiegelhalter unread (A1).
4. HIGH: Section 1 contradictions; A3's design costs far more than A5's (A3, A4, A5).
5. HIGH: F9 unbuilt code and a two-week engineering estimate (A5, A3).
6. HIGH: F6 N=5000 pool overlap breaks replicate independence (A4, A3).
7. HIGH: F4 stratified vs random estimand (A3, A4).
8. HIGH: F7 power depends on unobserved discordance; BillSum R=3 invalid (A4).
9. HIGH: F10 summarisation validity and human-eval budget (A4, A5).
10. HIGH: F5/F8 tuning and API fairness (thinking, temperature, caps) (A3, A5).

Reject today for: novelty not established, no evidence the central arm functions, a design that is internally inconsistent and unaffordable as written, summarisation claims that cannot be validated within budget.

## 6. Register changes

- I-10: keep CRITICAL (do not downgrade until Spiegelhalter is read).
- I-01, I-02: stay FIXED-UNVERIFIED; gate on F3. I-03: RESOLVED for the case bug is acceptable after independent re-score.
- I-04: stay OPEN (A3's IN-REVIEW is premature: R and N=5000 conflicts). I-05: IN-REVIEW only once code exists. I-07: not resolved (family excludes N=200). I-08: OPEN.
- I-09: keep HIGH; A5's "settled" overstates (thinking-off not confirmed). I-11: MEDIUM acceptable if no LEDGAR adapter is released. I-13: HIGH. I-15: keep HIGH (tuning cost omitted). I-18: do not close until incorporated policies are read.
- New: I-19 estimand vs stratified pools (HIGH); I-20 cross-draft contradictions (HIGH); I-21 confirmatory family excludes crossover region (HIGH); I-22 labelling cost not in the cost model (MEDIUM); I-23 tuning asymmetry at N=5000 and validation size (HIGH).

## 7. A3 Round 2 (read at the end; A1 still has none)

Resolves: test sizes (3,080 / 3,000 / 1,000), R rule (BillSum N=5000 exploratory at R=3), tuning counts (112 min / 252 rec.), `tune()` exists (A5-02 was wrong), Haiku API rule, frozen-arm file. Does not resolve:
- F1 stands: A3 keeps N=200 outside the confirmatory family and calls N=200 "the end of Spiegelhalter's range". The confirmatory points remain where LoRA is expected to win.
- F4 stands: stratified-vs-random estimand not settled.
- F6 stands: A3 keeps N=5000 and relabels the estimand as conditional on 9,003 rows. Defensible if stated, but replicates are still positively correlated; the "crossed-model CI stays valid" claim is unproved without A4's overlap simulation.
- F5 partly: validation size still 1,000 at N<=200; FT 10 trials vs ICL 4.
- Cost: A3's recommended grid is about $165 standard (about $120 with batch) for B77+BillSum only, my section 1 row 12 estimate still lower because it drops random-k; it exceeds A5 (b) and sits at the top of the $50-200 budget with no margin for repeats, Gemini thinking or Haiku tokenizer inflation. Minimum viable ($85) is the only affordable plan. 462k requests is not compatible with a 72-hour-per-provider window on Luna/Gemini batch limits (A5-06).
- GPU: A3's 112 / 252 trainings are not costed by A5 (still 60 / 112 / 237); BillSum hours remain unreconciled (section 1).
- A3's `tokenize_example` correction of A5-01 and the `max_length` rule are accepted.
- F3, F7-F12 unaffected. Integrator: send both Round 2 files to A6 for a second pass after A1 writes its section.
