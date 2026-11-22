"""BillSum summary generation and ROUGE-L evaluation of the fine-tuned model, offline.

No GPU and no torch: generation is injected, or torch is replaced by a tiny stub. The BANKING77 path is
protected by (1) the call into model.generate being unchanged when the new options are not used and (2) a
run() record whose key set is the one the code produced before the BillSum evaluation existed.
"""

import contextlib
import json
import sys
import types

import pytest

from src.train import billsum, evaluate as ev, summarise, train
from src.train.config import TrainConfig, config_from_args


class Tok:
    """Whitespace tokens as ints; chat template like the training tests."""
    eos_token_id = 0
    all_special_ids = [0, 99]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<s>{m['role']}\n{m['content']}</s>\n" for m in msgs)
        return out + ("<s>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (len(w) % 50) for w in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


def fake_gen(outputs):
    def fn(queries):
        assert len(queries) == len(outputs)
        return outputs
    return fn


def g(text, out_tokens=5, stopped=True):
    return {"text": text, "input_tokens": 100, "output_tokens": out_tokens, "stopped": stopped}


ITEMS = [(3, "bill a", "the bill funds water projects"), (9, "bill b", "the bill raises taxes on tobacco"),
         (11, "bill c", "the bill names a post office")]


def _eval(outputs, items=ITEMS):
    return summarise.evaluate_summaries(
        None, Tok(), run_id="r", system="m", technique="lora", n=200, seed=101, items=items, stop_ids=[0],
        pad_id=0, batch_size=2, generate_fn=fake_gen(outputs))


def test_records_scores_and_length_statistics():
    r = _eval([g("Summary: the bill funds water projects"), g("", 0), g("tax tobacco", 1024, stopped=False)])
    a, b, c = r["records"]
    assert a["arm"] == "finetuned" and a["task"] == "billsum" and a["item_id"] == "billsum-train-3"
    assert a["parsed"] == "the bill funds water projects" and a["parse_ok"] and a["extra"]["rouge_l"] == pytest.approx(1.0)
    assert b["parsed"] is None and b["parse_ok"] is False and b["extra"]["rouge_l"] == 0.0  # empty: kept, scored 0
    assert c["extra"]["hit_cap"] is True and c["extra"]["finish_reason"] == "length"
    assert a["extra"]["gold_tokens"] == 5 and a["extra"]["token_ratio"] == pytest.approx(1.0)
    assert a["extra"]["pred_words"] == 5 and a["extra"]["gold_words"] == 5 and a["extra"]["word_ratio"] == 1.0
    assert c["extra"]["token_ratio"] == pytest.approx(1024 / 6)
    assert r["n_eval"] == 3 and r["n_unparseable"] == 1 and r["n_hit_cap"] == 1
    assert r["rouge_l"] == pytest.approx((1.0 + 0.0 + c["extra"]["rouge_l"]) / 3)
    assert r["gen_tokens_median"] == 5 and r["token_ratio_median"] == pytest.approx(1.0)
    schema = pytest.importorskip("src.eval.schema")
    parsed = [schema.PredictionRecord.from_dict(x) for x in r["records"]]
    schema.check_run_consistency(parsed)


def test_generator_length_mismatch_is_an_error():
    with pytest.raises(RuntimeError):
        summarise.evaluate_summaries(None, Tok(), run_id="r", system="m", technique="lora", n=1, seed=1, items=ITEMS,
                                     stop_ids=[0], pad_id=0, batch_size=1, generate_fn=lambda q: [g("x")])


def test_rouge_l_matches_metrics_module_and_none_scores_zero():
    from src.eval.metrics import rouge_l
    preds, golds = ["a b c d", None, "x y"], ["a b c e", "a b", "x y z"]
    assert summarise.rouge_l_scores(preds, golds) == pytest.approx(list(rouge_l(preds, golds).per_item))
    assert summarise.rouge_l_scores([None], ["a b"]) == [0.0]


def test_length_fields_handle_zero_denominators():
    f = summarise.length_fields("", "", 0, 0, False)
    assert f["token_ratio"] is None and f["word_ratio"] is None and f["pred_words"] == 0
    assert summarise.length_fields("a b", "a", None, None, None)["token_ratio"] is None
    with pytest.raises(ValueError):
        summarise.aggregate([])


def test_summary_stop_ids_use_the_billsum_template():
    class T(Tok):
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return super().apply_chat_template(msgs, tokenize, add_generation_prompt)
    ids = summarise.summary_stop_ids(T(), ["gold"])
    assert 0 in ids  # eos always stops


# ---------------------------------------------------------------- config

def test_gen_eval_flags_default_off_and_billsum_only():
    c = TrainConfig(model="m", task="billsum")
    assert c.gen_eval is False and c.gen_limit is None and c.gen_batch == 8
    c = config_from_args(["--model", "qwen3-4b", "--task", "billsum", "--gen-eval", "--gen-limit", "20", "--gen-batch", "4"])
    assert c.gen_eval and c.gen_limit == 20 and c.gen_batch == 4
    assert config_from_args(["--model", "qwen3-4b", "--task", "billsum"]).gen_eval is False
    with pytest.raises(ValueError):
        TrainConfig(model="m", gen_eval=True)  # BANKING77
    with pytest.raises(ValueError):
        TrainConfig(model="m", task="billsum", gen_limit=0)


# ---------------------------------------------------------------- generate_predictions with a stub torch

class FT:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, idx):
        rows_sl, cols_sl = idx
        return FT([r[cols_sl] for r in self.rows[rows_sl]])

    def tolist(self):
        return self.rows


class FakeModel:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def parameters(self):
        yield types.SimpleNamespace(device="cpu")

    def eval(self):
        pass

    def generate(self, **kw):
        self.calls.append(kw)
        ids = kw["input_ids"].rows
        return FT([r + self.reply for r in ids])


@pytest.fixture
def stub_torch(monkeypatch):
    mod = types.ModuleType("torch")
    mod.inference_mode = lambda: contextlib.nullcontext()
    mod.tensor = lambda x, device=None: FT(x)
    monkeypatch.setitem(sys.modules, "torch", mod)


def test_banking77_generate_call_is_unchanged_without_the_new_options(stub_torch):
    m = FakeModel([7, 8, 0, 0])
    out = ev.generate_predictions(m, Tok(), ["my card", "lost"], [0], 0, 2, 32)
    assert set(m.calls[0]) == {"input_ids", "attention_mask", "max_new_tokens", "do_sample", "eos_token_id", "pad_token_id"}
    assert m.calls[0]["max_new_tokens"] == 32 and m.calls[0]["do_sample"] is False and m.calls[0]["eos_token_id"] == [0]
    assert [o["output_tokens"] for o in out] == [2, 2] and all(o["stopped"] for o in out)


def test_billsum_generation_uses_billsum_prompt_left_padding_cache_and_cap(stub_torch):
    m = FakeModel([7, 8, 9])  # no stop token: capped
    out = ev.generate_predictions(m, Tok(), ["short", "a much longer bill text here"], [0], 5, 2, 3,
                                  prompt_fn=billsum.prompt_ids, use_cache=True)
    call = m.calls[0]
    assert call["use_cache"] is True and call["max_new_tokens"] == 3
    rows = call["input_ids"].rows
    assert rows[0][0] == 5 and rows[1][0] != 5  # shorter prompt left-padded with pad id (batch sorted by length)
    assert call["attention_mask"].rows[0][0] == 0
    assert all(not o["stopped"] and o["output_tokens"] == 3 for o in out)
    assert out[0]["input_tokens"] < out[1]["input_tokens"]


# ---------------------------------------------------------------- train.run

class SaveModel(FakeModel):
    def save_pretrained(self, path):
        import pathlib
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _patch_run(monkeypatch, tmp_path, task):
    from src.train import env, registry

    monkeypatch.setattr(train, "_build_model", lambda spec, cfg: (SaveModel([1]), 0.1, 7))
    monkeypatch.setattr(train, "train_loop", lambda *a, **k: {
        "train_seconds": 1.0, "steps": 4, "epochs_effective": 1.0, "trained_tokens": 10, "loss_tokens": 5,
        "tokens_per_second": 10.0, "final_train_loss": 1.0, "first_train_loss": 2.0, "peak_mem_bytes": 1e9,
        "schedule_steps": 4, "checkpoint_seconds": 0.0, "early_stopped": False})
    monkeypatch.setattr(train, "val_loss", lambda *a, **k: 1.5)
    monkeypatch.setattr(train, "_tokenize", lambda cfg, tok, texts, targets, idx, train: (
        [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}] * max(1, len(idx)), []))
    monkeypatch.setattr(train, "_release_gpu", lambda: None)
    monkeypatch.setattr(env, "gpu_info", lambda: {"gpu": "none"})
    monkeypatch.setattr(env, "lib_versions", lambda: {})
    monkeypatch.setattr(env, "git_commit", lambda: "abc")
    spec = registry.resolve("qwen3-4b")
    if task == "billsum":
        texts = [f"bill {i}" for i in range(billsum.N_TRAIN)]
        golds = [f"gold summary {i}" for i in range(billsum.N_TRAIN)]
        splits = billsum.make_splits(1)
    else:
        from src.train import data, task_spec
        texts = [f"q{i}" for i in range(data.N_TRAIN)]
        golds = [task_spec.BANKING77_LABELS[i % 77] for i in range(data.N_TRAIN)]
        splits = data.make_splits(len(texts), 1, sampling="random", labels=golds)
    return (spec, texts, golds, splits, Tok(), 0)


def _cfg(tmp_path, **kw):
    return TrainConfig(model="qwen3-4b", n=200, replicate=1, results=str(tmp_path / "res.jsonl"),
                       predictions_dir=str(tmp_path / "pred"), checkpoint_dir=str(tmp_path / "ck"), **kw)


# Key set of a BANKING77 run record, written down from the code BEFORE the BillSum evaluation was added
# (tests below fail if the new options change a BANKING77 record).
BANKING_RECORD_KEYS = {
    "status", "run_id", "model", "model_key", "technique", "n", "seed", "task", "replicate", "epochs", "lr",
    "effective_batch", "micro_batch", "s_min", "grad_checkpointing", "schedule_total", "stop_step", "max_length",
    "frozen_arm", "selected_by_validation", "selection", "longest_example_tokens", "dropped_over_cap",
    "target_modules", "lora_r", "lora_alpha", "lora_dropout", "max_steps", "trainable_params", "load_seconds",
    "val_loss", "peak_mem_gb", "train_seconds", "steps", "epochs_effective", "trained_tokens", "loss_tokens",
    "tokens_per_second", "final_train_loss", "first_train_loss", "peak_mem_bytes", "schedule_steps",
    "checkpoint_seconds", "early_stopped", "accuracy", "n_eval", "n_unparseable", "eval_seconds", "eval_split",
    "generation_eval", "prompt", "pool_seed", "val_seed", "sampling", "val_size", "val_hash", "pool_n", "pool_hash",
    "gpu", "libs", "git_commit", "timestamp_utc", "hub_repo", "hub_pushed", "hub_error",
}


def test_banking77_run_record_and_predictions_unchanged(monkeypatch, tmp_path):
    prep = _patch_run(monkeypatch, tmp_path, "banking77")
    from src.train import task_spec

    def fake_eval(model, tok, **kw):
        recs = [ev.make_record(run_id=kw["run_id"], system=kw["system"], technique="lora", n=kw["n"], seed=kw["seed"],
                               train_index=i, prediction=task_spec.BANKING77_LABELS[0], gold=g, input_tokens=9,
                               output_tokens=3) for i, _, g in kw["items"]]
        return {"records": recs, "accuracy": 0.5, "n_eval": len(recs), "n_unparseable": 0, "eval_seconds": 0.2}

    monkeypatch.setattr(train, "evaluate", fake_eval)
    monkeypatch.setattr(summarise, "evaluate_summaries", lambda *a, **k: pytest.fail("BANKING77 must not summarise"))
    rec = train.run(_cfg(tmp_path), prep=prep)
    assert set(rec) == BANKING_RECORD_KEYS
    assert rec["generation_eval"] is True and rec["accuracy"] == 0.5 and rec["n_eval"] == 1000
    lines = (tmp_path / "pred").glob("*.jsonl")
    first = json.loads(next(lines).read_text().splitlines()[0])
    assert set(first["extra"]) == {"split", "pool_seed", "pool_hash"} and first["arm"] == "finetuned"


def test_billsum_run_without_gen_eval_is_loss_only_as_before(monkeypatch, tmp_path):
    prep = _patch_run(monkeypatch, tmp_path, "billsum")
    monkeypatch.setattr(summarise, "evaluate_summaries", lambda *a, **k: pytest.fail("generation is off by default"))
    rec = train.run(_cfg(tmp_path, task="billsum"), prep=prep)
    assert rec["generation_eval"] is False and rec["accuracy"] is None and rec["n_unparseable"] is None
    assert "summary_eval" not in rec and rec["val_loss"] == 1.5 and rec["n_eval"] == 1000
    assert not list((tmp_path / "pred").glob("*")) if (tmp_path / "pred").exists() else True


def test_billsum_run_with_gen_eval_limit_writes_rouge_records(monkeypatch, tmp_path):
    prep = _patch_run(monkeypatch, tmp_path, "billsum")

    def fake_generate(model, tok, queries, stop_ids, pad_id, batch_size, max_new_tokens, prompt_fn=None, use_cache=None):
        assert max_new_tokens == 1024 and use_cache is True and prompt_fn is billsum.prompt_ids and batch_size == 4
        return [g("gold summary " + q.split()[1]) for q in queries]  # the bill's own gold summary

    monkeypatch.setattr(ev, "generate_predictions", fake_generate)
    rec = train.run(_cfg(tmp_path, task="billsum", gen_eval=True, gen_limit=6, gen_batch=4), prep=prep)
    assert rec["generation_eval"] is True and rec["n_eval"] == 6 and rec["accuracy"] is None
    s = rec["summary_eval"]
    assert s["rouge_l"] == pytest.approx(1.0) and s["n_eval"] == 6 and "records" not in s
    assert rec["gen_limit"] == 6 and rec["max_new_tokens"] == 1024 and rec["val_loss"] == 1.5
    path = next((tmp_path / "pred").glob("*.jsonl"))
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(lines) == 6 and lines[0]["extra"]["pool_seed"] == 1 and lines[0]["extra"]["rouge_l"] == pytest.approx(1.0)
    schema = pytest.importorskip("src.eval.schema")
    schema.check_run_consistency([schema.PredictionRecord.from_dict(x) for x in lines])
