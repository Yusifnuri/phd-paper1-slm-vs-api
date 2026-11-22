#!/usr/bin/env python3
"""Token statistics of ALL 18,949 BillSum US train bills for the max_length decision.

CPU only; downloads tokenizer files (no model weights) and the train split only. The test
split is never read. Writes results/billsum_token_stats.json.

For each tokenizer: percentiles, mean and max of (a) bill text tokens, (b) gold summary
tokens, (c) the full training example (chat-template prompt + summary + closing token, i.e.
exactly what `billsum.tokenize_example` produces), the share of examples above candidate caps,
and a proposed max_length following the protocol rule (max train example + margin, rounded up).

    python scripts/billsum_token_stats.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train import billsum  # noqa: E402

TOKENIZERS = {"qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507", "phi4-mini": "microsoft/Phi-4-mini-instruct"}
CAPS = (4096, 6144, 8192, 9216, 10240, 12288, 16384)
PCTS = (5, 25, 50, 75, 90, 95, 99, 99.9)
OUT = Path("results/billsum_token_stats.json")


def pct(sorted_vals, q):
    # nearest-rank percentile on a sorted list
    k = max(0, min(len(sorted_vals) - 1, int(-(-q / 100 * len(sorted_vals) // 1)) - 1))
    return sorted_vals[k]


def describe(vals):
    s = sorted(vals)
    return {
        "mean": round(sum(s) / len(s), 1), "min": s[0], "max": s[-1],
        "percentiles": {str(q): pct(s, q) for q in PCTS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    texts, summaries = billsum.load_train_split()
    n = len(texts)
    result = {
        "dataset": billsum.DATASET_ID, "split": "train", "n_bills": n,
        "chars": {"text": describe([len(t) for t in texts]), "summary": describe([len(s) for s in summaries])},
        "rule": "max_length = ceil((max train example tokens + margin) / round_to) * round_to; "
                "test items are never truncated; drop train items above the cap only if under 0.1%",
        "margin": billsum.LENGTH_MARGIN, "round_to": billsum.LENGTH_ROUND,
        "tokenizers": {},
    }
    for key, hf_id in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(hf_id)
        doc = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        summ = [len(x) for x in tok(summaries, add_special_tokens=False)["input_ids"]]
        full = []
        for t, s in zip(texts, summaries):
            # exact training-example length; max_length is effectively unbounded here
            full.append(len(billsum.tokenize_example(tok, t, s, 10 ** 9)["input_ids"]))
        longest = max(range(n), key=lambda i: full[i])
        proposed = billsum.propose_max_length(max(full))
        result["tokenizers"][key] = {
            "hf_id": hf_id,
            "doc_tokens": describe(doc),
            "summary_tokens": describe(summ),
            "example_tokens": describe(full),
            "share_examples_above_cap": {
                str(c): {"count": sum(v > c for v in full), "share": sum(v > c for v in full) / n}
                for c in CAPS
            },
            "summary_tokens_above_1024": sum(v > billsum.MAX_NEW_TOKENS for v in summ),
            "longest_example_train_index": longest,
            "proposed_max_length": proposed,
            "dropped_at_proposed_cap": sum(v > proposed for v in full),
        }
    props = [v["proposed_max_length"] for v in result["tokenizers"].values()]
    result["proposed_max_length_all_models"] = max(props)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tokenizers"}, indent=1)[:600])
    for k, v in result["tokenizers"].items():
        print(k, "example max", v["example_tokens"]["max"], "p99", v["example_tokens"]["percentiles"]["99"],
              "proposed", v["proposed_max_length"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
