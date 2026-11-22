# Overlap and provider-terms check (2026-11-22)

Method: four web searches by the coordinator, then two fetch agents that read the papers' HTML versions and the providers' official terms pages. Both agents are AI readers and can err; pages marked "partly summarised" were not read in full. **This is research on public documents, not legal advice.** Absence of a paper in a limited search is not proof that none exists.

## 1. Is the study already done by someone else?

No paper found that combines our full design (LoRA Qwen3-4B, three current cheap APIs, retrieval-augmented ICL from the same N labelled examples at N = 200/1,000/3,000, BANKING77 and BillSum, replicated pools, kNN baseline, cost-per-correct break-even with tuning charged). Closest partial overlaps (read 2026-11-22; none uses BillSum, Qwen3-4B, Haiku, a current GPT mini or Flash-Lite):

| Paper | Overlap | What it does not cover | Confidence |
|---|---|---|---|
| arXiv 2311.06102, "Making LLMs Worth Every Penny", ACM ICAIF 2023 | **BANKING77**, similarity-retrieved examples (all-mpnet-base-v2, cosine), commercial APIs (GPT-3.5/4, Cohere, Claude 1/2), per-run cost table | encoder (SetFit/MPNet) not LoRA LLM; 2023 models; shots 1-20 per class; no replicates; no summarisation; no tuning-cost break-even | H for sections 1-6.2 (rest partly summarised) |
| arXiv 2609.20218 (single author, preprint, July 2026) | "labelled-data crossover N*", gpt-4.1-nano/mini, replicated design | tabular data only; classical models, not LLM fine-tuning | H (sections 3-6 read) |
| arXiv 2606.05781 (PayPal, 4 pages, preprint) | LoRA of an 8B model on 219 transcripts, cost per evaluation vs frontier APIs | one private task; no measured API accuracy arm; no retrieval | H (full read) |
| arXiv 2610.24260 (Walmart, preprint, "work done in 2024") | fine-tuned small model vs GPT-3.5, cost and routing | private survey data; RoBERTa encoder | H |
| Spiegelhalter et al. 2026 (workshop poster) | crossover ~50 labels, cost model | see `spiegelhalter-2026-notes.md` | H |

Consequence: 2311.06102 must be cited as the closest BANKING77 + retrieval + API-cost predecessor (not in `01-literature.md` before). Our contribution stays the conjunction (modern models, LoRA, N sweep to 3,000, summarisation, replicated pools, kNN, tuning charged), never "first". Search coverage: the agent could not search; the coordinator's four searches are the only discovery attempt, so scoop risk stays open and a re-search just before submission remains required.

## 2. Anything in the providers' official terms that forbids the study?

Read 2026-11-22 (Anthropic Commercial Terms effective 2025-06-17, AUP effective 2026-11-12; OpenAI Services Agreement updated 2025-12-01, usage policies effective 2026-11-20, sharing and publication policy of 2022; Google Gemini API Additional Terms effective 2026-03-23, Google APIs Terms 2021). Not opened: OpenAI Service-Specific Terms, Anthropic Consumer/Service-Specific Terms and retention page, Google Prohibited Use Policy and DPA, any academic programme terms.

| Provider | Benchmark publication | Competing-model / output-training clause | Publishing predictions | Risk for our plan |
|---|---|---|---|---|
| Anthropic | no restriction found | Commercial Terms D.4(a) "competing" limit; AUP bars using outputs to train a model without authorisation | customer owns Outputs (B); no restriction found | low if no Claude output enters any training pipeline |
| OpenAI | usage policies explicitly allow "model evaluation and benchmarking"; Services Agreement 10 bars public statements about the relationship | Services Agreement 3.3(e); usage policies bar outputs as labels or filters for substitute models | customer owns Output (4.1) | low with gold labels only; high if GPT output touches training labels, filtering or validation |
| Google | no clause found | "may not use the Services to develop models that compete"; vague "extract or replicate" wording | APIs Terms 5.e (distribution of "content returned from the APIs") is ambiguous; unpaid tier uses inputs for training and human review | medium: use the paid tier; do not publish raw Gemini outputs |

Decisions taken (L-31): (1) no API output is ever used to create, filter, select or validate training data for any model; (2) paid tiers only, recorded in every run; (3) public release contains metrics, per-item correct/incorrect flags and item ids, not raw Gemini output text; raw outputs of the other two providers are released only after the researcher's check (D-21); (4) model names are used as plain identifiers, with no statement of partnership or endorsement; (5) negotiated order-form prices, if any, are not published.
