# Spiegelhalter, Franke, Hutter 2026: full-text notes

Read 2026-11-22 from the PDF the researcher downloaded from OpenReview (36 pages; the PDF itself is not committed to this repository). Venue: NeurIPS 2026 Workshop on On-Device Intelligence (poster), not the main track [V, page 1 footer and neurips.cc/virtual/2026/170044]. Code: `github.com/automl/prompting-vs-finetuning-slm` [V, §1]. Page numbers are the paper's own.

## What they did [V]

| Element | Their setup | Where |
|---|---|---|
| Models | Qwen3-0.6B, 1.7B (YaRN to 128K), 4B-Instruct-2507; 8B/14B ICL-only references | §2, App. E.3 |
| Tasks | 17 narrow tasks, 7 categories; 14 of 17 synthetic (LLM-generated); real: TREC, E-Commerce, Injury | Table 1, §4 |
| N | 5, 10, 25, 50, 100, 200 | §2 |
| Test | fixed 300 items per task | §2 |
| Replication | two seeds (train/validation draws); error bars are across tasks, not seeds | §2, Fig. 1, §4 |
| ICL | All-samples, CV-greedy (5-fold, up to 30 demos), RAG (bge-m3, k=10) | §2 |
| APO | GEPA (reflection model gpt-oss-120b) | §2 |
| LoRA | Unsloth, r=64, 27 configs (lr x batch x epochs) x 5-fold CV = 135 runs per cell; best fold checkpoint tested | §2 |
| Tuning budgets | explicitly **not matched** across methods | §2, §4 |
| Cost | C_total(Q) = C_setup + Q*c_query; LoRA setup **includes HPO GPU-hours**; A6000 at $1.09/h; merged adapter; concurrency 1; **RAG excluded from the cost analysis** | §2, App. C.6 |
| Commercial APIs | none as an adaptation target; API prices only for GEPA's reflection model | §2 |
| Summarisation / long documents | none | Table 1 |

## What they found [V]

- ICL saturates at 25-50 examples; LoRA keeps improving to N=200 (§3, Takeaway 1).
- About 50 samples for LoRA to overtake all ICL strategies on average; on 4B the gap is significant only from N=100; at 4B, N=50: +3.4 pp [-0.7, 8.8] (§4 Statistics).
- Task category can shift or reverse the ordering; format following and code are exceptions (Takeaway 3).
- Cost-optimal method depends mainly on data budget, not volume, under merged single-task serving (Takeaway 4); with dynamic adapter serving LoRA's per-query cost roughly doubles on 4B (§3, App. E.5).
- HPO sensitivity: a single fixed LoRA config is within 1-4 pp of the per-task oracle at N >= 50, under 2 pp at N=200 (App. E.2).

## Their own limitations that matter for us [V, §4]

- "LoRA saturation beyond 200 samples is open."
- Synthetic, template-like data "may favor fine-tuning".
- One model family; two seeds; unmatched optimisation budgets; RAG not costed.

## Consequences for our paper [I/R]

1. **Novelty (I-10).** No commercial API arm, no N > 200, no long-document summarisation, no real-data majority, RAG not costed, no kNN baseline. Our core gap holds against this paper. Their stated open question (beyond 200 samples) is exactly our N range: cite it as motivation.
2. **Not new any more:** charging the tuning (HPO) cost in the setup cost of LoRA is done here. Our break-even contribution must be worded as "against commercial API prices, with retrieval costed", not "tuning cost charged" (consistent with I-28).
3. **Tuning bar (L-14/L-17).** Their App. E.2 supports a small grid at N >= 200 (regret under 2 pp at N=200 on their suite): evidence for our 4-trial primary protocol, to be cited, with our wider-grid sensitivity kept.
4. **Hypothesis framing.** Their result predicts LoRA beats local ICL at all our N; the open question is whether that survives against stronger commercial models with retrieval on real data. A null or reversed result there is informative, not a failure.
5. **Statistics contrast.** Their uncertainty is across tasks with two seeds and 300-item tests; our item-by-replicate mixed model with 5-8 replicates is a genuine methodological difference to state (without disparaging).
