# Gemini API Flash tier: prices and limits (read 2026-11-27)

Read by a fetch agent from Google's pages (ai.google.dev URLs redirect to aistudio.google.com/docs/...; no page showed a last-updated date). The pricing page was truncated, so rows marked M come from a model-extracted summary. Re-read on the day of any paid run.

## Prices, paid tier, USD per 1M tokens (output includes thinking tokens)

| Model | In | Out | Batch in / out | Status | Conf |
|---|---|---|---|---|---|
| gemini-3.8-flash | 0.75 (1.50 from 2027-01-01) | 3.75 (7.50) | 0.375 / 1.875 (0.75 / 3.75) | stable | H |
| gemini-3.6-flash | same as 3.8 | same | same | stable | H |
| gemini-3.5-flash-lite | 0.30 | 2.50 | 0.15 / 1.25 | stable | H |
| gemini-3.1-flash-lite | 0.25 | 1.50 | 0.125 / 0.75 | stable, shutdown 2027-05-07 | M price, H date |
| gemini-3-flash-preview | 0.50 | 3.00 | 0.25 / 1.50 | preview | M |
| gemini-2.5-flash | 0.30 | 2.50 | 0.15 / 1.25 | deprecated, earliest shutdown 2026-12-01 | M price |
| gemini-2.5-flash-lite | 0.10 | 0.40 | 0.05 / 0.20 | existing users only | M price |

## Thinking
Cannot be switched fully off: `thinking_level` minimal is the lowest for 3.6-flash, 3.5-flash-lite, 3.1-flash-lite; 3.8-flash starts at low. (Inference from the level lists.)

## Limits and billing
- Batch enqueued tokens, Tier 1: 10M (Flash-Lite), 3M (3.6/3.8 Flash). Tier 1 spend limit $10 per rolling 10 minutes (M).
- Prepay: minimum $5, credits expire after 12 months; at zero balance keys fail with HTTP 402, overage possible for about 10 minutes. A project monthly cap exists ("experimental").
- Free tier: content is used to improve Google products (do not use for results, L-08/L-31).

## Cost estimates (assistant's arithmetic; token counts are assumptions, not measured)
Assumptions: BANKING77 input 900-1,220 tokens (k=32 basis), output 8; BillSum input 6,950-9,300, output 250; thinking 0 to 300 tokens; batch mode; request counts from `analysis/count_requests.py` (4 k values).

| Model | B77 R=8 (106,128 req) | B77 R=3 (50,688) | BillSum R=3 (19,400) |
|---|---|---|---|
| 3.1-flash-lite, thinking 0-100 | $13-25 | $6-12 | $21-28 |
| 3.5-flash-lite, thinking 0-100 | $15-34 | $7-16 | $26-36 |
| 3.6/3.8-flash, thinking 0-100 | $37-70 | $18-34 | $60-80 |

The R=5 figure quoted in chat (about $9-17 for 3.1-flash-lite, 62k-81k requests) is a hand calculation with a mid input length; `count_requests.py` has no R=5 scenario yet and still assumes 4 k values.

Pages: https://aistudio.google.com/docs/pricing , /models , /deprecations , /thinking , /rate-limits , /billing
