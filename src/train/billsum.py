"""BillSum (US) data layer for the training side: splits, pools, prompt, tokenisation, near-duplicates.

Mirrors src/train/data.py (BANKING77) so both tasks obey the same protocol (EXPERIMENT_PROTOCOL
section 2):

    dataset      FiscalNote/billsum (verified 2026-11-22: splits train 18,949 / test 3,269 /
                 ca_test 1,237; fields text, summary, title). Only the US train and US test
                 splits are used. The California split (ca_test) is never loaded.
    val          shuffle(range(18,949), VAL_SEED)[:1000]   FIXED for every replicate, selection only
    remainder    the other 17,949 rows
    pool(N)      shuffle(remainder, pool_seed)[:N]         nested prefixes, fresh order per
                 pool_seed (= replicate r), shared by every method and by the API arm.
                 Random pools throughout (BillSum has no classes, so no stratified variant).
    test subset  1,000 of the 3,269 US test bills, seeded draw (TEST_SUBSET_SEED), kept in
                 original order, fingerprinted with a sha256 over (text, summary) rows.

Validation size: the protocol fixes 1,000 fixed train rows for every task ("Validation = 1,000
fixed train rows ... never in a pool"), so VAL_SIZE = 1000 here too; the per-N rule
val(N) = min(1000, N) is applied by the caller on top of this fixed carve-out.

Test isolation: the US test split is read by exactly ONE function,
`load_us_test_subset_for_final_eval`, which only the final-evaluation entry point may call.
Nothing else in this module (or in training/validation code) touches it. The split name is
kept in one private constant so a test can verify that.

Prompt: the rendered system and user text is byte-identical to what src/api/prompts.py renders
for the BillSum summarisation template with zero examples (tests/train/test_billsum.py imports
the API template to prove it). Output cap for every arm is MAX_NEW_TOKENS = 1024 (protocol
section 5); the API template currently says 2048 and must be changed to match.
"""

from __future__ import annotations

import ast
import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import data

TASK = "billsum"
DATASET_ID = "FiscalNote/billsum"
_US_TRAIN_SPLIT = "train"
_US_TEST_SPLIT = 'test'  # used ONLY in load_us_test_subset_for_final_eval (checked by a test)

N_TRAIN, N_TEST = 18949, 3269
VAL_SEED = data.VAL_SEED
VAL_SIZE = data.VAL_SIZE  # 1,000, same fixed carve-out as BANKING77 (protocol section 2)
POOL_SIZES = (200, 1000, 3000)  # confirmatory N; the pool function accepts any size <= 17,949
TEST_SUBSET_SIZE = 1000
TEST_SUBSET_SEED = 42

MAX_NEW_TOKENS = 1024  # protocol section 5: BillSum output cap, every arm
TEMPERATURE = 0.0

# --------------------------------------------------------------------------- prompt

BILLSUM_DESCRIPTION = (
    "You summarise US legislative bills. Summarise the bill in the style of the example summaries."
)
SYSTEM_TEMPLATE = (
    "{task_description}\n\n"
    "Write only the summary, as plain prose, with no heading or preamble."
)
USER_TEMPLATE = "Document:\n{input}\n\nSummary:"


def system_prompt() -> str:
    return SYSTEM_TEMPLATE.format(task_description=BILLSUM_DESCRIPTION)


def user_prompt(text: str) -> str:
    return USER_TEMPLATE.format(input=text)


def build_messages(text: str) -> List[Dict[str, str]]:
    """Zero-shot chat messages for one bill (what the SLM is trained and evaluated on)."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(text)},
    ]


# --------------------------------------------------------------------------- splits

def make_splits(pool_seed: int = data.DATA_SEED, n_train: int = N_TRAIN) -> data.Splits:
    """Fixed validation carve-out and the nested random pools of replicate `pool_seed`."""
    return data.make_splits(
        n_train, pool_seed=pool_seed, val_size=VAL_SIZE, val_seed=VAL_SEED, sampling="random"
    )


def subset_indices(
    n_test: int = N_TEST, size: int = TEST_SUBSET_SIZE, seed: int = TEST_SUBSET_SEED
) -> List[int]:
    """Seeded draw of `size` test rows, returned in ascending (original) order."""
    if not 1 <= size <= n_test:
        raise ValueError(f"subset size must be in [1, {n_test}], got {size}")
    return sorted(random.Random(seed).sample(range(n_test), size))


def rows_hash(texts: Sequence[str], summaries: Sequence[str]) -> str:
    """sha256 over (text, summary) rows in order; the content hash of the test subset."""
    return data.rows_fingerprint(texts, summaries)


def item_id(split: str, index: int) -> str:
    return f"{TASK}-{split}-{index}"


def load_train_split() -> Tuple[List[str], List[str]]:
    """Load ONLY the US train split. Returns (bill texts, gold summaries). Never reads test."""
    from datasets import load_dataset  # lazy: not needed for offline tests

    ds = load_dataset(DATASET_ID, split=_US_TRAIN_SPLIT)
    texts, summaries = list(ds["text"]), list(ds["summary"])
    if len(texts) != N_TRAIN:
        raise RuntimeError(f"expected {N_TRAIN} US train bills, got {len(texts)}")
    return texts, summaries


def load_us_test_subset_for_final_eval() -> Dict[str, Any]:
    """THE ONLY reader of the US test split. Call it from the final-evaluation entry point only.

    Returns the seeded 1,000-bill subset in original order together with its content hash:
    {"indices", "texts", "summaries", "hash", "n_full_test"}.
    """
    from datasets import load_dataset  # lazy

    ds = load_dataset(DATASET_ID, split=_US_TEST_SPLIT)
    if len(ds) != N_TEST:
        raise RuntimeError(f"expected {N_TEST} US test bills, got {len(ds)}")
    idx = subset_indices(len(ds))
    texts, summaries = [ds[i]["text"] for i in idx], [ds[i]["summary"] for i in idx]
    return {
        "indices": idx, "texts": texts, "summaries": summaries,
        "hash": rows_hash(texts, summaries), "n_full_test": len(ds),
    }


def split_summary(splits: data.Splits, n: int) -> Dict[str, Any]:
    out = data.split_summary(splits, n)
    out["task"] = TASK
    return out


# --------------------------------------------------------------------------- tokenisation

def render_pair(tokenizer: Any, text: str, summary: str) -> Tuple[str, str]:
    """(prompt_text, completion_text): prompt ends with the generation prompt; completion is the
    gold summary plus whatever the template emits to close the assistant turn."""
    msgs = build_messages(text)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": summary}], tokenize=False
    )
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    return prompt, full[len(prompt):]


def tokenize_example(tokenizer: Any, text: str, summary: str, max_length: int) -> Dict[str, Any]:
    """input_ids and labels (-100 on the prompt: completion-only loss).

    Never truncates. If the example is longer than max_length it raises, so an over-long
    test-side item can never be silently cut. Training code that wants to drop over-long
    train bills must do so explicitly through `select_trainable` (logged, under 0.1%).
    """
    prompt, completion = render_pair(tokenizer, text, summary)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_length:
        raise ValueError(
            f"example has {len(ids)} tokens > max_length {max_length} (no truncation): {text[:60]!r}"
        )
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + c_ids, "n_prompt": len(p_ids)}


def prompt_ids(tokenizer: Any, text: str) -> List[int]:
    prompt = tokenizer.apply_chat_template(
        build_messages(text), tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


MAX_DROP_FRACTION = 0.001  # protocol: drop train examples above the cap only if under 0.1%
LENGTH_MARGIN = 256
LENGTH_ROUND = 512


def propose_max_length(max_example_tokens: int, margin: int = LENGTH_MARGIN, round_to: int = LENGTH_ROUND) -> int:
    """Protocol rule (A5-01): the maximum observed train example length plus a margin, rounded
    up to a multiple of `round_to`. Test items are never truncated, so the margin is the
    safety room for test bills longer than any train bill; `tokenize_example` raises if the
    room is not enough."""
    return int(math.ceil((max_example_tokens + margin) / round_to) * round_to)


def select_trainable(
    indices: Sequence[int], lengths: Dict[int, int], max_length: int,
    max_drop_fraction: float = MAX_DROP_FRACTION,
) -> Tuple[List[int], List[int]]:
    """(kept, dropped) train indices; drops pool items above max_length, but only if they are
    below `max_drop_fraction` of the pool, otherwise raises (the cap must be raised instead).
    The caller must log `dropped`."""
    kept = [i for i in indices if lengths[i] <= max_length]
    dropped = [i for i in indices if lengths[i] > max_length]
    if indices and len(dropped) / len(indices) >= max_drop_fraction:
        raise ValueError(
            f"{len(dropped)}/{len(indices)} pool items exceed max_length {max_length} "
            f"(>= {max_drop_fraction:.3%}): raise max_length instead of dropping"
        )
    return kept, dropped


# --------------------------------------------------------------------------- near duplicates

NGRAM = 5
NEAR_DUP_THRESHOLD = 0.8  # protocol: flag 5-gram Jaccard >= 0.8
_WORD = re.compile(r"\w+")


def shingles(text: str, n: int = NGRAM) -> Set[int]:
    """Set of hashed word n-grams of the lower-cased text (punctuation and case ignored).
    A text shorter than n words yields the single shingle of all its words (empty -> empty set)."""
    words = _WORD.findall(text.lower())
    if not words:
        return set()
    if len(words) < n:
        return {hash(tuple(words))}
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


class NearDupIndex:
    """Exact max 5-gram Jaccard of a query bill against a pool of bills (inverted index).

    Hashes are Python hashes of word tuples: stable within one process, which is all that is
    needed (the index is rebuilt per pool and never stored).
    """

    def __init__(self, pool_texts: Iterable[str], n: int = NGRAM):
        self.n = n
        self.sets: List[Set[int]] = [shingles(t, n) for t in pool_texts]
        self.postings: Dict[int, List[int]] = {}
        for j, s in enumerate(self.sets):
            for h in s:
                self.postings.setdefault(h, []).append(j)

    def max_jaccard(self, text: str) -> Tuple[float, Optional[int]]:
        """(max Jaccard, position in the pool of the best match); (0.0, None) if no overlap."""
        q = shingles(text, self.n)
        counts: Counter = Counter()
        for h in q:
            for j in self.postings.get(h, ()):
                counts[j] += 1
        best, best_j = 0.0, None
        for j, inter in counts.items():
            sim = inter / (len(q) + len(self.sets[j]) - inter)
            if sim > best:
                best, best_j = sim, j
        return best, best_j


def max_jaccard_to_pool(test_texts: Sequence[str], pool_texts: Sequence[str]) -> List[Tuple[float, Optional[int]]]:
    """Per test item: (max 5-gram Jaccard to any pool bill, pool position). Flag items with
    value >= NEAR_DUP_THRESHOLD; they stay in the primary analysis (protocol section 2)."""
    index = NearDupIndex(pool_texts)
    return [index.max_jaccard(t) for t in test_texts]


# --------------------------------------------------------------------------- guard helper

def functions_referencing(source: str, name: str) -> Set[str]:
    """Names of top-level functions whose body mentions identifier `name` (used by a test to prove
    that only one function can reach the test split)."""
    out: Set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node)):
                out.add(node.name)
    return out
