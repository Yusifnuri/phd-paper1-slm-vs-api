"""Offline tests for src/train/billsum.py: synthetic data only, no network, no model downloads."""

import sys
from pathlib import Path

import pytest

from src.api import prompts as api_prompts
from src.train import billsum as bs
from src.train import data

SRC = Path(bs.__file__)


# ------------------------------------------------------------------ splits

def test_pools_nested_val_fixed_and_disjoint():
    s = bs.make_splits(pool_seed=1)
    assert len(s.val) == 1000 and len(s.remainder) == bs.N_TRAIN - 1000 == 17949
    pools = [s.pool(n) for n in bs.POOL_SIZES]
    for small, big in zip(pools, pools[1:]):
        assert big[: len(small)] == small
    assert not set(s.val) & set(s.remainder)
    assert len(set(s.val) | set(s.remainder)) == bs.N_TRAIN
    s2 = bs.make_splits(pool_seed=2)
    assert s2.val == s.val  # validation is the same for every replicate
    assert s2.pool(1000) != s.pool(1000)  # fresh pool per replicate
    assert bs.make_splits(pool_seed=1) == s  # deterministic
    assert s.sampling == "random"


def test_split_summary_has_hashes():
    out = bs.split_summary(bs.make_splits(0), 200)
    assert out["task"] == "billsum" and out["pool_n"] == 200
    assert len(out["pool_hash"]) == 64 and len(out["val_hash"]) == 64


def test_test_subset_seeded_sorted_and_hashed():
    idx = bs.subset_indices()
    assert len(idx) == 1000 == len(set(idx)) and idx == sorted(idx)
    assert min(idx) >= 0 and max(idx) < bs.N_TEST
    assert idx == bs.subset_indices()
    assert idx != bs.subset_indices(seed=7)
    texts = [f"bill {i}" for i in idx]
    sums = [f"sum {i}" for i in idx]
    h = bs.rows_hash(texts, sums)
    assert h == bs.rows_hash(texts, sums) and len(h) == 64
    assert h != bs.rows_hash(texts[::-1], sums[::-1])  # order sensitive
    assert h != bs.rows_hash(texts, sums[:-1] + ["changed"])  # content sensitive
    with pytest.raises(ValueError):
        bs.subset_indices(n_test=10, size=11)


# ------------------------------------------------------------------ test-split isolation

def _fake_datasets(monkeypatch, calls, n_train, n_test):
    class FakeDS(list):
        def __getitem__(self, k):
            return list.__getitem__(self, k) if isinstance(k, int) else [r[k] for r in self]

    def fake_load(name, split=None, **kw):
        calls.append((name, split))
        n = {"train": n_train, "test": n_test}[split]
        return FakeDS({"text": f"{split} bill {i}", "summary": f"{split} sum {i}"} for i in range(n))

    mod = type(sys)("datasets")
    mod.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_train_loader_requests_train_split_only(monkeypatch):
    calls = []
    _fake_datasets(monkeypatch, calls, bs.N_TRAIN, bs.N_TEST)
    texts, sums = bs.load_train_split()
    assert calls == [("FiscalNote/billsum", "train")]
    assert len(texts) == len(sums) == bs.N_TRAIN


def test_train_loader_checks_size(monkeypatch):
    _fake_datasets(monkeypatch, [], 10, 10)
    with pytest.raises(RuntimeError):
        bs.load_train_split()


def test_only_one_function_can_reach_the_test_split(monkeypatch):
    src = SRC.read_text()
    assert bs.functions_referencing(src, "_US_TEST_SPLIT") == {"load_us_test_subset_for_final_eval"}
    assert src.count("_US_TEST_SPLIT") == 2  # definition + the single use
    assert 'split="ca_test"' not in src and "split='ca_test'" not in src
    # the named function does read it, once, and returns the seeded subset with its hash
    calls = []
    _fake_datasets(monkeypatch, calls, bs.N_TRAIN, bs.N_TEST)
    out = bs.load_us_test_subset_for_final_eval()
    assert calls == [("FiscalNote/billsum", "test")]
    assert out["indices"] == bs.subset_indices() and len(out["texts"]) == 1000
    assert out["hash"] == bs.rows_hash(out["texts"], out["summaries"]) and out["n_full_test"] == bs.N_TEST


def test_training_side_calls_never_touch_test(monkeypatch):
    calls = []
    _fake_datasets(monkeypatch, calls, bs.N_TRAIN, bs.N_TEST)
    bs.load_train_split()
    bs.make_splits(3).pool(200)
    assert {c[1] for c in calls} == {"train"}


# ------------------------------------------------------------------ prompt equality with the API

@pytest.mark.parametrize("text", [
    "SECTION 1. SHORT TITLE.\n\nThis Act may be cited as the Example Act.",
    "Unicode § 2 — “quoted” text\n\n  indented\n",
    "",
])
def test_rendered_prompt_is_byte_identical_to_api(text):
    ctx = {"task_description": api_prompts.BILLSUM_DESCRIPTION}
    api = api_prompts.build_prompt("summarisation", [], text, ctx)
    msgs = bs.build_messages(text)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"].encode() == api.system.encode()
    assert msgs[1]["content"].encode() == api.user.encode()
    assert bs.BILLSUM_DESCRIPTION == api_prompts.BILLSUM_DESCRIPTION


# ------------------------------------------------------------------ tokenisation

class FakeTok:
    """ChatML-like template with an <|im_end|> closing token; one token per word."""
    eos_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(w) % 1000 + 2 for w in text.split()]}


def test_tokenize_example_masks_prompt_and_keeps_completion():
    tok = FakeTok()
    ex = bs.tokenize_example(tok, "a bill text", "the summary", 10_000)
    n = ex["n_prompt"]
    assert ex["labels"][:n] == [-100] * n
    assert ex["labels"][n:] == ex["input_ids"][n:] and len(ex["labels"]) == len(ex["input_ids"])
    assert ex["input_ids"][:n] == bs.prompt_ids(tok, "a bill text")
    prompt, completion = bs.render_pair(tok, "a bill text", "the summary")
    assert completion.startswith("the summary") and completion.endswith("<|im_end|>\n")


def test_tokenize_example_raises_instead_of_truncating():
    tok = FakeTok()
    text = " ".join(["word"] * 500)
    full = bs.tokenize_example(tok, text, "s", 10_000)
    n = len(full["input_ids"])
    assert bs.tokenize_example(tok, text, "s", n)["input_ids"] == full["input_ids"]  # exactly at the cap
    with pytest.raises(ValueError, match="no truncation"):
        bs.tokenize_example(tok, text, "s", n - 1)


def test_propose_max_length_rule():
    assert bs.propose_max_length(7970) == 8704  # (7970 + 256) rounded up to a multiple of 512
    assert bs.propose_max_length(8000, margin=0, round_to=512) == 8192
    assert bs.propose_max_length(100) % bs.LENGTH_ROUND == 0 and bs.propose_max_length(100) >= 100 + bs.LENGTH_MARGIN


def test_select_trainable_drops_only_under_0p1_percent():
    idx = list(range(2000))
    lengths = {i: 100 for i in idx}
    lengths[5] = 9000
    kept, dropped = bs.select_trainable(idx, lengths, 8704)  # 1/2000 = 0.05% < 0.1%
    assert dropped == [5] and 5 not in kept and len(kept) == 1999
    lengths[6] = 9000  # 2/2000 = 0.1%: not allowed, raise the cap instead
    with pytest.raises(ValueError):
        bs.select_trainable(idx, lengths, 8704)
    kept, dropped = bs.select_trainable(idx, {i: 1 for i in idx}, 8704)
    assert dropped == [] and kept == idx


# ------------------------------------------------------------------ 5-gram Jaccard

def test_shingles_and_jaccard():
    assert bs.jaccard(bs.shingles("a b c d e f"), bs.shingles("A, B. c d e f!")) == 1.0  # case/punct ignored
    a, b = bs.shingles("a b c d e f"), bs.shingles("a b c d e g")
    assert len(a) == 2 and bs.jaccard(a, b) == pytest.approx(1 / 3)  # {abcde} shared of 3 distinct
    assert bs.jaccard(bs.shingles("x y z"), bs.shingles("x y z")) == 1.0  # short text: one shingle
    assert bs.jaccard(set(), set()) == 0.0 and bs.shingles("") == set()
    assert bs.jaccard(bs.shingles("a b c d e"), bs.shingles("v w x y z")) == 0.0


def test_near_dup_index_matches_brute_force():
    pool = [
        "the secretary shall establish a program to assist small businesses in rural areas of the state",
        "no funds may be used to carry out any activity under this section unless the secretary certifies",
        "a bill to amend the internal revenue code to provide for a tax credit for energy efficient homes",
    ]
    queries = [
        pool[0],  # exact duplicate
        pool[0].replace("rural", "urban"),  # near duplicate
        "an entirely unrelated sentence about fishing quotas in the northern pacific region",
    ]
    res = bs.max_jaccard_to_pool(queries, pool)
    for q, (val, pos) in zip(queries, res):
        brute = [bs.jaccard(bs.shingles(q), bs.shingles(p)) for p in pool]
        assert val == pytest.approx(max(brute))
        if val > 0:
            assert pos == brute.index(max(brute))
    assert res[0] == (1.0, 0) and bs.NEAR_DUP_THRESHOLD <= res[0][0]
    assert 0.0 < res[1][0] < 1.0 and res[1][1] == 0
    assert res[2] == (0.0, None)


def test_item_id():
    assert bs.item_id("train", 7) == "billsum-train-7"
