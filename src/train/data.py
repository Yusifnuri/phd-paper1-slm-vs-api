"""BANKING77 data: fixed validation carve-out, nested per-replicate training pools, prompts.

The pilot never reads the test split: `load_train_split` loads only "train".

Splits are index lists into the original train split (10,003 rows):
    val       = shuffle(all, VAL_SEED)[:1000]     FIXED for every replicate (model selection only)
    remainder = the other 9,003 rows
    order     = shuffle(remainder, pool_seed)     (or class-interleaved, see below)
    pool(N)   = order[:N]                          so pool(50) < pool(200) < pool(1000) < pool(5000)
A replicate is (pool_seed, seed): each pool_seed gives a fresh nested pool, shared by every
method (and by the API arm) of that replicate; seed only changes training randomness.

sampling="random" (default): plain random order, so at N=50 most of the 77 intents have no
example. sampling="stratified": classes are interleaved round-robin, so every prefix pool(N)
has floor(N/77) or ceil(N/77) examples per class (and is still nested). Which one the paper
uses is still pending, hence the flag.

The test split is frozen: its size and fingerprint (sha256 of texts+labels in original order)
are recorded once by scripts/freeze_test_split.py, which is the only code that reads it.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import task_spec
from .task_spec import BANKING77_LABELS, TASK, build_messages  # noqa: F401

DATASET_ID = "PolyAI/banking77"
# Pinned dataset revision, the one hashed in results/test_split_fingerprint.json (L-42).
REVISION = "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
DATA_SEED = 42  # default pool seed
VAL_SEED = 42
VAL_SIZE = 1000
POOL_SIZES = (50, 200, 1000, 5000)
N_TRAIN, N_TEST = 10003, 3080


@dataclass(frozen=True)
class Splits:
    pool_seed: int
    val_seed: int
    sampling: str
    val: Tuple[int, ...]
    remainder: Tuple[int, ...]  # pool order; pools are prefixes of it

    def pool(self, n: int) -> List[int]:
        if not 1 <= n <= len(self.remainder):
            raise ValueError(f"pool size must be in [1, {len(self.remainder)}], got {n}")
        return list(self.remainder[:n])


def _interleave_by_class(order: Sequence[int], labels: Sequence[str], seed: int) -> List[int]:
    by_class: Dict[str, List[int]] = {}
    for i in order:
        by_class.setdefault(labels[i], []).append(i)
    classes = sorted(by_class)
    random.Random(seed + 1).shuffle(classes)  # which classes come first at small N
    out: List[int] = []
    depth = max(len(v) for v in by_class.values())
    for d in range(depth):
        out += [by_class[c][d] for c in classes if d < len(by_class[c])]
    return out


def make_splits(
    n_train: int, pool_seed: int = DATA_SEED, val_size: int = VAL_SIZE, val_seed: int = VAL_SEED,
    sampling: str = "random", labels: Optional[Sequence[str]] = None,
) -> Splits:
    if val_size >= n_train:
        raise ValueError("val_size must be smaller than the training set")
    perm = list(range(n_train))
    random.Random(val_seed).shuffle(perm)  # random.Random(int) is stable across Python 3.x
    val, rest = perm[:val_size], sorted(perm[val_size:])
    random.Random(pool_seed).shuffle(rest)
    if sampling == "stratified":
        if labels is None or len(labels) != n_train:
            raise ValueError("stratified sampling needs the label of every training row")
        rest = _interleave_by_class(rest, labels, pool_seed)
    elif sampling != "random":
        raise ValueError(f"unknown sampling {sampling!r}")
    return Splits(pool_seed, val_seed, sampling, tuple(val), tuple(rest))


def indices_hash(indices: Sequence[int]) -> str:
    """sha256 of the ordered index list; recorded per pool so runs can be matched later."""
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def rows_fingerprint(texts: Sequence[str], labels: Sequence[str]) -> str:
    """sha256 over (text, label) rows in order; used to freeze and verify the test split."""
    h = hashlib.sha256()
    for t, l in zip(texts, labels):
        h.update(json.dumps([t, l], ensure_ascii=False).encode() + b"\n")
    return h.hexdigest()


def load_train_split() -> Tuple[List[str], List[str]]:
    """Load ONLY the train split from the Hub. Returns (texts, label names per row)."""
    from datasets import load_dataset  # lazy: not needed for offline tests

    ds = load_dataset(DATASET_ID, split="train", revision=REVISION)
    names = ds.features["label"].names
    if list(names) != BANKING77_LABELS:
        raise RuntimeError("dataset label names differ from task_spec.BANKING77_LABELS")
    return list(ds["text"]), [names[i] for i in ds["label"]]


def item_id(train_index: int) -> str:
    return f"{TASK}-train-{train_index}"


# ---------------------------------------------------------------- prompts and tokens

def render_pair(tokenizer: Any, query: str, label: str) -> Tuple[str, str]:
    """(prompt_text, completion_text) using the tokenizer's chat template.

    prompt_text ends with the generation prompt; completion_text is the label plus
    whatever the template emits to close the assistant turn (e.g. <|im_end|>).
    """
    msgs = build_messages(query)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": label}], tokenize=False
    )
    if not full.startswith(prompt):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    return prompt, full[len(prompt):]


def tokenize_example(
    tokenizer: Any, query: str, label: str, max_length: int
) -> Dict[str, Any]:
    """input_ids and labels (-100 on the prompt). Raises if longer than max_length."""
    prompt, completion = render_pair(tokenizer, query, label)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    ids = p_ids + c_ids
    if len(ids) > max_length:
        raise ValueError(f"example has {len(ids)} tokens > max_length {max_length}: {query[:60]!r}")
    return {
        "input_ids": ids,
        "labels": [-100] * len(p_ids) + c_ids,
        "n_prompt": len(p_ids),
    }


def prompt_ids(tokenizer: Any, query: str) -> List[int]:
    prompt = tokenizer.apply_chat_template(
        build_messages(query), tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def pad_batch(examples: Sequence[Dict[str, Any]], pad_id: int) -> Dict[str, List[List[int]]]:
    """Right-pad to the longest example (training)."""
    width = max(len(e["input_ids"]) for e in examples)
    out: Dict[str, List[List[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        gap = width - len(e["input_ids"])
        out["input_ids"].append(e["input_ids"] + [pad_id] * gap)
        out["attention_mask"].append([1] * len(e["input_ids"]) + [0] * gap)
        out["labels"].append(e["labels"] + [-100] * gap)
    return out


def left_pad(seqs: Sequence[Sequence[int]], pad_id: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Left-pad prompts for batched generation. Returns (input_ids, attention_mask)."""
    width = max(len(s) for s in seqs)
    ids, mask = [], []
    for s in seqs:
        gap = width - len(s)
        ids.append([pad_id] * gap + list(s))
        mask.append([0] * gap + [1] * len(s))
    return ids, mask


def split_summary(splits: Splits, n: int) -> Dict[str, Any]:
    pool = splits.pool(n)
    return {
        "pool_seed": splits.pool_seed, "val_seed": splits.val_seed, "sampling": splits.sampling,
        "val_size": len(splits.val),
        "val_hash": indices_hash(splits.val),
        "pool_n": n,
        "pool_hash": indices_hash(pool),
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
