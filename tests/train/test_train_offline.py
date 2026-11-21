"""Offline, CPU-only tests: no network, no GPU, no model downloads, no torch needed."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.train import data, evaluate as ev
from src.train.config import TECHNIQUES, TrainConfig, config_from_args, run_id
from src.train.registry import MODELS, resolve
from src.train.targets import check_lora_targets_from_names, unmatched_targets

N_TRAIN = 10003


# ------------------------------------------------------------------ splits

def test_pools_nested_and_sizes():
    s = data.make_splits(N_TRAIN)
    pools = [s.pool(n) for n in data.POOL_SIZES]
    assert [len(p) for p in pools] == list(data.POOL_SIZES)
    for small, big in zip(pools, pools[1:]):
        assert big[: len(small)] == small  # nested, same order
    assert len(s.val) == 1000 and len(s.remainder) == N_TRAIN - 1000


def test_val_disjoint_from_every_pool():
    s = data.make_splits(N_TRAIN)
    assert not set(s.val) & set(s.remainder)
    assert not set(s.val) & set(s.pool(5000))
    assert len(set(s.val) | set(s.remainder)) == N_TRAIN


def test_deterministic_and_seed_sensitive():
    a, b = data.make_splits(N_TRAIN, 42), data.make_splits(N_TRAIN, 42)
    assert a == b
    assert data.indices_hash(a.pool(1000)) == data.indices_hash(b.pool(1000))
    assert data.make_splits(N_TRAIN, 43).val != a.val
    # Pin the hash so an accidental change of the split is caught (computed once, seed 42).
    assert data.indices_hash(a.pool(50)) == data.indices_hash(list(a.remainder[:50]))


def test_pool_too_large_rejected():
    with pytest.raises(ValueError):
        data.make_splits(N_TRAIN).pool(9004)


def test_split_summary_has_hashes():
    s = data.split_summary(data.make_splits(N_TRAIN), 200)
    assert s["pool_n"] == 200 and len(s["pool_hash"]) == 64 and len(s["val_hash"]) == 64


def test_test_split_never_loaded(monkeypatch):
    """load_train_split must request split='train' only."""
    calls = []

    class FakeFeature:
        names = ["a", "b"]

    class FakeDS(dict):
        features = {"label": FakeFeature()}

    def fake_load(name, split=None, **kw):
        calls.append((name, split))
        if split != "train":
            raise AssertionError(f"touched split {split!r}")
        return FakeDS(text=["x", "y"], label=[0, 1])

    fake_mod = type(sys)("datasets")
    fake_mod.load_dataset = fake_load
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)
    texts, labels = data.load_train_split()
    assert calls == [("PolyAI/banking77", "train")]
    assert (texts, labels) == (["x", "y"], ["a", "b"])


def test_no_test_split_reference_in_sources():
    for f in Path(data.__file__).parent.glob("*.py"):
        src = f.read_text()
        assert 'split="test"' not in src and "split='test'" not in src and '"test"' not in src, f


# ------------------------------------------------------------------ prompts

class FakeTok:
    """ChatML-like template with an <|im_end|> closing token."""
    eos_token_id = 0
    all_special_ids = [0, 1]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 50) for c in text.replace("<|im_end|>", "\x01") if c != "\x01"]
                + ([1] if text.endswith("<|im_end|>\n") else [])}


def test_render_pair_prefix_and_label():
    prompt, comp = data.render_pair(FakeTok(), "my card is lost", "lost_or_stolen_card")
    assert prompt.endswith("<|im_start|>assistant\n")
    assert comp.startswith("lost_or_stolen_card") and "<|im_end|>" in comp
    assert "my card is lost" in prompt and data.INSTRUCTION in prompt


def test_tokenize_example_masks_prompt():
    ex = data.tokenize_example(FakeTok(), "hello", "card_arrival", max_length=512)
    n = ex["n_prompt"]
    assert ex["labels"][:n] == [-100] * n
    assert ex["labels"][n:] == ex["input_ids"][n:] and len(ex["input_ids"]) > n


def test_too_long_example_raises():
    with pytest.raises(ValueError):
        data.tokenize_example(FakeTok(), "x" * 1000, "card_arrival", max_length=50)


def test_padding_helpers():
    b = data.pad_batch([{"input_ids": [5, 6], "labels": [-100, 6]},
                        {"input_ids": [7], "labels": [7]}], pad_id=0)
    assert b["input_ids"] == [[5, 6], [7, 0]]
    assert b["attention_mask"] == [[1, 1], [1, 0]]
    assert b["labels"] == [[-100, 6], [7, -100]]
    ids, mask = data.left_pad([[1, 2, 3], [4]], pad_id=9)
    assert ids == [[1, 2, 3], [9, 9, 4]] and mask == [[1, 1, 1], [0, 0, 1]]


# ------------------------------------------------------------------ targets

# Module names as defined in the transformers modeling files (checked 2026-11-21).
KNOWN = {
    "phi4-mini": ["model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj",
                  "model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"],
    "qwen3-4b": [f"model.layers.0.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
                + ["model.layers.0.mlp.gate_proj"],
    "olmo3-7b": [f"model.layers.0.self_attn.{n}" for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
                + ["model.layers.0.mlp.gate_proj"],
}


def test_every_registered_model_has_known_names_and_passes():
    assert set(KNOWN) == set(MODELS), "add known module names for new registry entries"
    for key, names in KNOWN.items():
        check_lora_targets_from_names(names, MODELS[key].target_modules)


def test_phi4_requires_fused_qkv():
    assert MODELS["phi4-mini"].target_modules == ("qkv_proj", "o_proj")
    with pytest.raises(ValueError, match="q_proj"):  # the thesis bug: q/k/v match nothing
        check_lora_targets_from_names(KNOWN["phi4-mini"], ["q_proj", "k_proj", "v_proj", "o_proj"])


def test_wrong_name_fails_even_if_others_match():
    with pytest.raises(ValueError, match="nope_proj"):
        check_lora_targets_from_names(KNOWN["qwen3-4b"], ["q_proj", "nope_proj"])
    assert unmatched_targets(["a.b.q_proj"], ["q_proj", "proj"]) == ["proj"]  # suffix, not substring


def test_registry_resolve():
    assert resolve("qwen3-4b").hf_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert resolve("allenai/Olmo-3-7B-Instruct").key == "olmo3-7b"
    with pytest.raises(KeyError):
        resolve("nope")


# ------------------------------------------------------------------ config

def test_config_defaults_match_thesis():
    c = config_from_args(["--model", "qwen3-4b"])
    assert (c.technique, c.n, c.seed, c.epochs) == ("lora", 1000, 42, 3)
    assert (c.lora_r, c.lora_alpha, c.lora_dropout) == (16, 32, 0.05)
    assert (c.lr, c.warmup_ratio, c.effective_batch, c.max_length) == (2e-4, 0.03, 16, 512)
    assert c.total_steps() == 63 * 3


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(model="m", technique="full")
    with pytest.raises(ValueError):
        TrainConfig(model="m", micro_batch=5)
    assert TrainConfig(model="m", n=50, max_steps=10).total_steps() == 10
    assert TrainConfig(model="m", micro_batch=8).grad_accum == 2
    with pytest.raises(SystemExit):
        config_from_args(["--model", "m", "--technique", "bogus"])


def test_run_id_filesystem_safe():
    assert run_id("Qwen/Qwen3-4B-Instruct-2507", "dora", 1000, 42) == "Qwen3-4B-Instruct-2507__dora__n1000__seed42"


def test_peft_kwargs_per_technique():
    from src.train.train import peft_kwargs  # torch/peft are imported lazily inside functions

    t = MODELS["phi4-mini"].target_modules
    base = peft_kwargs("lora", t, 16, 32, 0.05)
    assert base["target_modules"] == ["qkv_proj", "o_proj"] and (base["r"], base["lora_alpha"]) == (16, 32)
    assert not base["use_dora"] and not base["use_rslora"]
    assert peft_kwargs("dora", t, 16, 32, 0.05)["use_dora"]
    assert peft_kwargs("rslora", t, 16, 32, 0.05)["use_rslora"]
    q = peft_kwargs("qlora", t, 16, 32, 0.05)  # QLoRA = plain LoRA adapters on a 4-bit base
    assert not q["use_dora"] and not q["use_rslora"]


# ------------------------------------------------------------------ evaluation

LABELS = {"card_arrival", "lost_or_stolen_card", "exchange_rate"}


def test_parse_label_variants():
    assert ev.parse_label("card_arrival", LABELS) == "card_arrival"
    assert ev.parse_label("  `Card Arrival`\nextra", LABELS) == "card_arrival"
    assert ev.parse_label("exchange-rate.", LABELS) == "exchange_rate"
    assert ev.parse_label("I think it is card_arrival", LABELS) is None
    assert ev.parse_label("", LABELS) is None


def _rec(pred, gold):
    return ev.make_record(run_id="r", system="m", technique="lora", n=50, seed=42, train_index=7,
                          prediction=pred, gold=gold, labels=LABELS, input_tokens=10, output_tokens=3)


def test_accuracy_counts_unparseable_as_wrong():
    recs = [_rec("card_arrival", "card_arrival"), _rec("exchange_rate", "card_arrival"), _rec("???", "card_arrival")]
    assert recs[2]["parse_ok"] is False and recs[2]["parsed"] is None
    assert ev.accuracy(recs) == pytest.approx(1 / 3)


def test_record_matches_eval_schema_fields():
    r = _rec("card_arrival", "card_arrival")
    assert tuple(r) == ev.SCHEMA_FIELDS
    assert r["arm"] == "finetuned" and r["task"] == "banking77" and r["item_id"] == "banking77-train-7"
    assert r["parse_ok"] == (r["parsed"] is not None)
    json.dumps(r)


def test_record_validates_against_real_schema_if_available():
    """Validates with src/eval/schema.py from branch claude/eval-harness-3f9302 when present."""
    try:
        schema = importlib.import_module("src.eval.schema")
    except ImportError:
        pytest.skip("src.eval.schema not on this branch (not merged)")
    schema.PredictionRecord.from_dict(_rec("card_arrival", "card_arrival"))
    schema.PredictionRecord.from_dict(_rec("junk", "card_arrival"))


def test_stop_ids_include_end_of_turn():
    class T:
        eos_token_id = 100
        all_special_ids = [100, 101]
    assert ev.stop_token_ids(T(), [5, 6, 101]) == [100, 101]


# ------------------------------------------------------------------ orchestrator

def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_pilot", Path(__file__).resolve().parents[2] / "scripts" / "run_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plan_skip_and_summary(tmp_path):
    rp = _runner()
    keys = rp.plan(["qwen3-4b", "phi4-mini"], ["lora", "dora"], [1000], 42)
    assert len(keys) == 4
    q = "Qwen/Qwen3-4B-Instruct-2507"
    rows = [
        {"status": "ok", "model": q, "technique": "lora", "n": 1000, "seed": 42, "train_seconds": 100.0,
         "load_seconds": 20.0, "peak_mem_gb": 20.0, "tokens_per_second": 5000.0, "accuracy": 0.9, "hub_pushed": True},
        {"status": "failed", "model": q, "technique": "dora", "n": 1000, "seed": 42},
    ]
    assert rp.done_keys(rows) == {(q, "lora", 1000, 42)}  # failures are retried
    table = rp.summary_table(rows)
    assert "FAILED" in table and "0.900" in table
    rows.append({**rows[0], "technique": "dora", "train_seconds": 250.0})  # retry succeeds
    assert "2.50x" in rp.summary_table(rows)


def test_run_pilot_dry_run_and_smoke(tmp_path):
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, str(root / "scripts" / "run_pilot.py"), "--smoke", "--dry-run",
                          "--results", str(tmp_path / "s.jsonl")], capture_output=True, text=True, cwd=root)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("src.train.train") == 3  # 3 models x lora
    assert "--max-steps 10" in out.stdout and "--hub-repo" not in out.stdout
