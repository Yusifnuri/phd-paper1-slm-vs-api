# Session record, 2026-11-22 to 2026-11-27

Written 2026-11-27 because the chat context is full. It records what was asked, what was done and what is open in the long interactive session. Technical detail lives in the files named here; this is the map. Current state and rules: `docs/HANDOFF.md` (first section).

## 1. Chronology

| When | What happened | Where it is recorded |
|---|---|---|
| 22 Nov, night | Resumed from handoff; verified the research foundation deliverables; added second-pass check, quality gates | `docs/foundation/`, `agents/06b`, `agents/06c` |
| 22 Nov | Honest assessment of the topic: sound but incremental applied paper; Q2 applied journal realistic; preprint early | this file, section 3 |
| 22 Nov | Merged eval, API and training branches; read Spiegelhalter et al. 2026 in full (PDF supplied by the researcher) | `docs/research/spiegelhalter-2026-notes.md` |
| 22 Nov | Built tuned-procedure code, kNN, GPU pilot stage 2 (G5 failed), lr grid changes | `results/pilot-stage2-2026-11-22/` |
| 22 Nov | Parallel local workers: BillSum data layer, local ICL, analysis code, pre-registration draft v2 | commits `fda3625`..`bfc16b4` |
| 22 Nov | GPU sessions 2 and 3: stage-2 rerun (G4 narrow fail, G5 pass), BillSum probe, local ICL, N=3,000 tuning, first BillSum runs | `results/pilot-stage2b-2026-11-22/`, `results/pilot-session3-2026-11-22/` |
| 22 Nov | Overlap and provider-terms check; ARS skills installed and tested once | `docs/research/2026-11-22-overlap-and-terms-check.md`, `ars_review_test_2026-11-22.md` |
| 22 Nov | Four more local workers: BillSum generation + ROUGE-L, test-read gate and fingerprints, blinded variance and contrast tools, document alignment; decisions L-33..L-38 | commits `37ce468`..`cafc966` |
| 23 Nov | Independent review 06d: **not safe to freeze yet**; must-fix list adopted (L-39) | `docs/foundation/agents/06d-independent-redteam.md` (commit `0cecc45`) |
| 23 Nov | Progress presentation for the teacher (11 slides, Turkish) | https://claude.ai/artifact/MwUuwwyMXeN3dRdxwrkPuo (private) |
| 23 Nov | Research proposal as a Claude Doc (Turkish, 10 sections, chart and timeline) | https://claude.ai/artifact/JMEtaKJsjH6Z3ycAssxkgr (private) |
| 24 Nov | Doctoral article proposal filled into the institute's Word form (ENT_FR31), for course BYL625, 9 pages | `docs/proposal/Makale_Onerisi_BYL625.docx` and `.pdf` (**not committed**) |
| 27 Nov | $15 API budget question; Gemini Flash-tier prices checked | `docs/research/2026-11-27-gemini-prices.md`, section 4 |

## 2. Study in one paragraph

LoRA-tuned Qwen3-4B-Instruct-2507 versus local in-context learning (retrieved examples), kNN, and cheap commercial APIs with retrieval from the same labelled pool of N = 200 / 1,000 / 3,000, on BANKING77 (3,080 test items, 8 replicates) and BillSum US (1,000 seeded test items, 5 replicates); LEDGAR exploratory (never run, no loader). Crossed items-by-replicates model, six Holm-corrected contrasts, cost per correct output with tuning charged. Never claim "first".

## 3. Answers given to the researcher (plain-language facts they may reuse)

- **Pilot** = small cheap trial on the validation split only, to test design and code; its numbers show direction, not results. About 6 GPU-hours in three sessions (own accounting).
- **ICL is not fine-tuning:** fine-tuning changes model weights with the labelled examples; ICL leaves weights unchanged and puts retrieved examples in the prompt; kNN votes among the most similar labels; the API arm is ICL on a provider's model.
- **Pilot sizes:** pools N = 200 / 1,000 / 3,000; tuning validation = min(1,000, N) rows; final pilot runs scored on 1,000 validation rows; 2 replicates for finals (replicate 0 for tuning); the test split was never read.
- **Full study sizes:** test 3,080 (BANKING77) and 1,000 (BillSum); replicates 8 and 5; every replicate re-scores the same test items. Steps: N 200 -> 1,000 -> 3,000; learning rates double from 5e-5 to 8e-4; k = 4, 8, 16, 32, 64 (BillSum 1, 2, 3); five checkpoints per training schedule.
- **Pilot validation accuracy (replicate 0, same rows as selection, exploratory):** N=200 kNN 0.670, local ICL 0.805, LoRA 0.770; N=1,000 0.833 / 0.853 / 0.871; N=3,000 ICL 0.901, LoRA 0.921.
- **LEDGAR metrics** are classification metrics: accuracy (invalid = wrong), invalid rate, accuracy among valid outputs, macro-F1; exploratory only; only ids, labels and metrics are released.
- **Datasets and licences:** `docs/foundation/DATASET_AUDIT.md`.

## 4. API budget and model choice (open)

- The researcher said keys will come and asked whether $15 is enough, then: "use the Flash tier".
- First answer (22 Nov price basis, Haiku / GPT mini class): $15 would cover one provider on BANKING77 at about $7-9. **Superseded:** with Gemini prices read on 27 Nov this does not hold.
- Estimate with Gemini prices (batch; token counts are estimates): `gemini-3.1-flash-lite`, BANKING77 only, 5 replicates: about $9-17; 8 replicates: $13-25; full Flash (3.6/3.8): $37-70 for BANKING77, not within $15; BillSum alone at least about $20.
- **Recommendation, not yet approved:** one provider (`gemini-3.1-flash-lite`), BANKING77 only, 5 replicates (the confirmatory floor), batch mode, drop the random-k control and the extra random-pool cells first if thinking tokens are high; prepay billing (minimum $5) plus a project spend cap; never the free tier; start with a $5 live test.
- **Open question to the researcher:** does "Flash tier" mean Flash-Lite (fits $15) or full Flash (needs $40+)?
- **If approved, documents must change before the freeze:** the "best API" selection becomes a pre-chosen single API; H3 and the six-contrast family must be rewritten because BillSum would have no API arm.

## 5. Open items for the researcher

1. API: Flash-Lite or full Flash, budget, approval of the narrowed scope; the key is typed by the researcher into the environment, never into chat.
2. Course proposal form: student number (blank), spelling of the name ("Yusif NURİ" was inferred), department sub-field name on the cover, ethics-board question, defence date; check the LEDGAR reference title (written from memory). Advisor shows "not assigned"; course BYL625, instructor Prof. Dr. Ferdi Sönmez.
3. External timestamp for the pre-registration (for example OSF) and reading/approving the decisions taken on the researcher's behalf (`PREREG_DECISIONS.md`).
4. Q2 requirement in writing; annotator for the optional faithfulness study; author and registry details.

## 6. Open engineering work (no GPU), from review 06d (L-39)

1. One test-evaluation path per arm through `src/final/final_eval`, dry-run on validation, committed before the tag (today nothing writes a test result).
2. One BillSum selection criterion for both arms; rewrite H3/RQ2; pin the ROUGE variant.
3. Recast H2 and H4 so they can fail; state tuning effort honestly and count validation labels.
4. Disclose test-derived audit statistics, log fingerprint reads, pin dataset revisions.
5. Align pre-registration text (API variant, stale passages); G4 waiver.
6. Needs GPU: three repeat runs of one configuration (non-determinism), BillSum validation generation pilot (ROUGE-L spread, timing); first GPU run of BillSum generation and the BillSum ICL runner.

## 7. Operational state

GPU server: off (the researcher stops it in the portal). No background jobs or scheduled tasks are active. Last pushed commit before this record: `0cecc45`.
