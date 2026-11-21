"""BANKING77 data: fixed validation carve-out, nested training pools, prompt building.

The pilot never reads the test split: `load_train_split` loads only "train", and nothing
in this package references a test split.

Splits are index lists into the original train split (10,003 rows):
    perm      = seeded shuffle of all train indices
    val       = perm[:1000]                 fixed validation set
    remainder = perm[1000:]                 9,003 rows
    pool(N)   = remainder[:N]               so pool(50) < pool(200) < pool(1000) < pool(5000)
Pools are plain random draws, not class-stratified: at N=50 most of the 77 intents have
no example. That is the label-budget question itself, not a bug.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

DATASET_ID = "PolyAI/banking77"
TASK = "banking77"
DATA_SEED = 42
VAL_SIZE = 1000
POOL_SIZES = (50, 200, 1000, 5000)

INSTRUCTION = (
    "Classify the customer's banking query into one intent label. "
    "Answer with the label name only."
)


@dataclass(frozen=True)
class Splits:
    seed: int
    val: Tuple[int, ...]
    remainder: Tuple[int, ...]  # permuted order; pools are prefixes of it

    def pool(self, n: int) -> List[int]:
        if not 1 <= n <= len(self.remainder):
            raise ValueError(f"pool size must be in [1, {len(self.remainder)}], got {n}")
        return list(self.remainder[:n])


def make_splits(n_train: int, seed: int = DATA_SEED, val_size: int = VAL_SIZE) -> Splits:
    if val_size >= n_train:
        raise ValueError("val_size must be smaller than the training set")
    perm = list(range(n_train))
    random.Random(seed).shuffle(perm)  # random.Random(int) is stable across Python 3.x
    return Splits(seed=seed, val=tuple(perm[:val_size]), remainder=tuple(perm[val_size:]))


def indices_hash(indices: Sequence[int]) -> str:
    """sha256 of the ordered index list; recorded per pool so runs can be matched later."""
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()


def load_train_split() -> Tuple[List[str], List[str]]:
    """Load ONLY the train split from the Hub. Returns (texts, label names per row)."""
    from datasets import load_dataset  # lazy: not needed for offline tests

    ds = load_dataset(DATASET_ID, split="train")
    names = ds.features["label"].names
    return list(ds["text"]), [names[i] for i in ds["label"]]


def item_id(train_index: int) -> str:
    return f"{TASK}-train-{train_index}"


# ---------------------------------------------------------------- prompts and tokens

def build_messages(query: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": f"{INSTRUCTION}\n\nQuery: {query}"}]


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
        "data_seed": splits.seed,
        "val_size": len(splits.val),
        "val_hash": indices_hash(splits.val),
        "pool_n": n,
        "pool_hash": indices_hash(pool),
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
