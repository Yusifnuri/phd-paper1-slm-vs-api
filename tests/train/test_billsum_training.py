"""Offline tests of the BillSum training branch: config defaults, tokenisation rules, probe helpers."""

import pytest

from src.train import probe, train, tuning
from src.train.config import BILLSUM_MAX_LENGTH, TrainConfig, config_from_args, run_id


class FakeTok:
    eos_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<s>{m['role']}\n{m['content']}</s>\n" for m in msgs)
        return out + ("<s>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 50) for c in text]}


def test_billsum_config_defaults_and_banking77_unchanged():
    b = TrainConfig(model="m", task="billsum", n=1000)
    assert b.max_length == BILLSUM_MAX_LENGTH == 8704
    assert b.micro == 1 and b.batch == 16 and b.grad_accum == 16 and b.use_grad_checkpointing
    c = TrainConfig(model="m", n=1000)
    assert c.task == "banking77" and c.max_length == 1024 and c.micro == c.batch and not c.use_grad_checkpointing
    assert TrainConfig(model="m", task="billsum", max_length=4096).max_length == 4096
    assert TrainConfig(model="m", task="billsum", grad_checkpointing=False).use_grad_checkpointing is False
    with pytest.raises(ValueError):
        TrainConfig(model="m", task="nope")
    with pytest.raises(ValueError):
        TrainConfig(model="m", val_cap=0)


def test_cli_task_flags_and_run_id_prefix():
    c = config_from_args(["--model", "qwen3-4b", "--task", "billsum", "--n", "200", "--val-cap", "300",
                          "--no-grad-checkpointing"])
    assert c.task == "billsum" and c.val_cap == 300 and c.use_grad_checkpointing is False
    assert run_id("a/Qwen3", "lora", 200, 101, 1, "random") == "Qwen3__lora__n200__seed101__pool1"  # unchanged
    assert run_id("a/Qwen3", "lora", 200, 101, 1, "random", "billsum") == "billsum__Qwen3__lora__n200__seed101__pool1"


def test_val_cap_sets_the_tuning_validation_size():
    assert [tuning.val_size(n, 300) for n in (200, 1000, 3000)] == [200, 300, 300]
    assert [tuning.val_size(n) for n in (200, 1000, 3000)] == [200, 1000, 1000]


def test_tokenize_drops_over_cap_train_items_only_when_rare_and_raises_for_val():
    from src.train import billsum
    short = len(billsum.tokenize_example(FakeTok(), "short bill", "a summary", 10 ** 9)["input_ids"])
    cfg = TrainConfig(model="m", task="billsum", n=10, max_length=short + 50)
    texts = ["short bill"] * 2000 + ["x" * 5000]
    targets = ["a summary"] * 2001
    ex, dropped = train._tokenize(cfg, FakeTok(), texts, targets, list(range(2001)), train=True)
    assert dropped == [2000] and len(ex) == 2000  # 1/2001 < 0.1%
    with pytest.raises(ValueError):  # more than 0.1% over the cap: the cap must be raised instead
        train._tokenize(cfg, FakeTok(), texts[:10] + ["x" * 5000], targets[:11], list(range(11)), train=True)
    with pytest.raises(ValueError):  # validation/test-side items are never dropped or truncated
        train._tokenize(cfg, FakeTok(), texts, targets, [2000], train=False)


def test_banking77_tokenize_path_unchanged():
    cfg = TrainConfig(model="m", n=10, max_length=100_000)  # char-level fake tokens: the 77-label prompt is long
    ex, dropped = train._tokenize(cfg, FakeTok(), ["my card is lost"], ["lost_or_stolen_card"], [0], train=True)
    assert dropped == [] and len(ex) == 1 and ex[0]["labels"][0] == -100


def test_probe_helpers():
    lengths = {i: 100 + i for i in range(200)}
    assert probe.longest_fraction(lengths, 0.01) == [199, 198]
    assert probe.longest_fraction({1: 5}, 0.01) == [1]
    with pytest.raises(ValueError):
        probe.longest_fraction({}, 0.01)
    p = probe.projected_seconds(3000, mean_tokens=2200.0, tokens_per_second=5000.0)
    assert p["schedule_steps"] == 940 and p["seconds"] == pytest.approx(940 * 16 * 2200 / 5000)
    assert probe.go_no_go(1.2 * 3600).startswith("keep the design")
    assert probe.go_no_go(1.5 * 3600).startswith("cuts 1 and 4")
    assert probe.go_no_go(2.1 * 3600).startswith("cuts 1, 5 and 6")
    with pytest.raises(ValueError):
        probe.projected_seconds(200, 1.0, 0)


def test_run_and_tune_have_summarisation_branches():
    src = open(train.__file__, encoding="utf-8").read()
    assert 'metric = "loss" if is_sum else "accuracy"' in src and "generation_eval" in src
    assert "gradient_checkpointing_enable" in src and "enable_input_require_grads" in src
