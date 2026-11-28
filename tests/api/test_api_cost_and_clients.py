"""Cost estimator, config, and provider clients against fake SDK objects."""

from types import SimpleNamespace

import pytest

from src.api.clients import BATCH_DONE, BATCH_PENDING, AnthropicClient, GoogleClient, OpenAIClient
from src.api.cost import PLANNED_N, estimate_prompts, estimate_tokens, planned_grid, price_usd
from src.api.prompts import Prompt


def test_config_has_confirmed_ids_and_sources(specs):
    assert specs["anthropic"].model_id == "claude-haiku-5-5"
    assert specs["openai"].model_id == "gpt-6-luna"
    assert specs["google"].model_id == "gemini-3.1-flash-lite"
    for s in specs.values():
        assert s.model_id_confirmed and s.price_source.startswith("https://") and s.price_read_date
        assert 0 < s.batch_discount < 1 and s.logprobs is False


def test_token_estimate_and_price(specs):
    a, o = specs["anthropic"], specs["openai"]
    assert estimate_tokens(4000, o) == 1000
    assert estimate_tokens(4000, a) == 1300  # Haiku tokenizer multiplier
    assert price_usd(o, 1_000_000, 1_000_000, batch=False) == pytest.approx(0.60)
    assert price_usd(o, 1_000_000, 1_000_000, batch=True) == pytest.approx(0.30)
    g = specs["google"]
    assert price_usd(g, 2_000_000, 0, batch=False) == pytest.approx(0.50)


def test_estimate_prompts(specs):
    o = specs["openai"]
    e = estimate_prompts([Prompt("s" * 400, "u" * 3600)] * 10, o, max_output_tokens=32, expected_output_chars=40)
    assert e.input_tokens == 10_000 and e.output_tokens_ceiling == 320 and e.output_tokens_expected == 100
    assert e.usd_standard_ceiling == pytest.approx((10_000 * 0.10 + 320 * 0.50) / 1e6)


def test_planned_grid_shape(specs):
    rows = planned_grid()
    assert len(rows) == 2 * 2 * len(PLANNED_N)  # two tasks x two providers (L-41)
    assert {r.provider for r in rows} == {"openai", "anthropic"}
    assert all(r.n_items == 1000 for r in rows)
    bill = [r for r in rows if r.task == "billsum" and r.provider == "openai"][0]
    # 5 shots of ~11.5k chars plus the query: roughly 15-16k tokens per item.
    assert 14_000 < bill.input_tokens / bill.n_items < 17_000


# --- clients with fake SDKs -------------------------------------------------

class _Obj(SimpleNamespace):
    def model_dump(self, mode="json"):
        def conv(v):
            if isinstance(v, _Obj):
                return v.model_dump()
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v
        return {k: conv(v) for k, v in vars(self).items()}


def test_anthropic_client_params_and_usage(specs):
    captured = {}

    def create(**kw):
        captured.update(kw)
        return _Obj(model="claude-haiku-5-5", stop_reason="end_turn",
                    content=[_Obj(type="text", text="exchange_rate")],
                    usage=_Obj(input_tokens=120, output_tokens=4))

    sdk = SimpleNamespace(messages=SimpleNamespace(create=create))
    c = AnthropicClient(specs["anthropic"], sdk=sdk).complete(Prompt("SYS", "USER"), 32)
    assert captured["thinking"] == {"type": "disabled"} and "temperature" not in captured
    assert captured["system"] == "SYS" and captured["messages"][0]["content"] == "USER"
    assert (c.text, c.input_tokens, c.output_tokens) == ("exchange_rate", 120, 4)
    assert c.latency_ms is not None and c.raw["model"] == "claude-haiku-5-5"


def test_anthropic_batch_cycle(specs):
    created = {}
    results = [
        _Obj(custom_id="r1", result=_Obj(type="succeeded", message={"model": "m", "content": [{"type": "text", "text": "b"}],
                                                                   "usage": {"input_tokens": 5, "output_tokens": 1}})),
        _Obj(custom_id="r0", result=_Obj(type="errored", error={"type": "overloaded"})),
    ]
    status = iter(["in_progress", "ended"])
    batches = SimpleNamespace(
        create=lambda requests: created.setdefault("r", requests) and _Obj(id="msgbatch_1"),
        retrieve=lambda bid: _Obj(processing_status=next(status)),
        results=lambda bid: iter(results),
    )
    cl = AnthropicClient(specs["anthropic"], sdk=SimpleNamespace(messages=SimpleNamespace(batches=batches)))
    bid = cl.batch_submit([("r0", Prompt("s", "a")), ("r1", Prompt("s", "b"))], 16)
    assert bid == "msgbatch_1" and created["r"][0]["params"]["thinking"] == {"type": "disabled"}
    assert cl.batch_poll(bid, interval_s=0, sleep=lambda s: None) == BATCH_DONE
    res = {r.custom_id: r for r in cl.batch_collect(bid)}
    assert res["r1"].completion.text == "b" and res["r0"].completion is None


def test_openai_client_reasoning_none(specs):
    captured = {}

    def create(**kw):
        captured.update(kw)
        return _Obj(model="gpt-6-luna", status="completed",
                    output=[_Obj(type="message", content=[_Obj(type="output_text", text="card_arrival")])],
                    usage=_Obj(input_tokens=50, output_tokens=3, output_tokens_details=_Obj(reasoning_tokens=0)))

    sdk = SimpleNamespace(responses=SimpleNamespace(create=create))
    c = OpenAIClient(specs["openai"], sdk=sdk).complete(Prompt("SYS", "USER"), 32)
    assert captured["reasoning"] == {"effort": "none"} and captured["instructions"] == "SYS"
    assert captured["store"] is False and "temperature" not in captured
    assert (c.text, c.input_tokens, c.output_tokens) == ("card_arrival", 50, 3)
    assert c.extra["reasoning_tokens"] == 0


def test_openai_batch_cycle(specs):
    uploaded = {}

    def files_create(file, purpose):
        uploaded["purpose"] = purpose
        uploaded["lines"] = file.read().decode().splitlines()
        return _Obj(id="file_in")

    out_line = ('{"custom_id": "r0", "response": {"status_code": 200, "body": {"model": "gpt-6-luna", '
                '"output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}], '
                '"usage": {"input_tokens": 9, "output_tokens": 2}}}}')
    err_line = '{"custom_id": "r1", "response": null, "error": {"code": "batch_expired"}}'
    files = SimpleNamespace(create=files_create,
                            content=lambda fid: SimpleNamespace(text=out_line if fid == "file_out" else err_line))
    batches = SimpleNamespace(
        create=lambda **kw: uploaded.setdefault("batch", kw) and _Obj(id="batch_1"),
        retrieve=lambda bid: _Obj(status="completed", output_file_id="file_out", error_file_id="file_err"),
    )
    cl = OpenAIClient(specs["openai"], sdk=SimpleNamespace(files=files, batches=batches))
    assert cl.batch_submit([("r0", Prompt("s", "a")), ("r1", Prompt("s", "b"))], 16) == "batch_1"
    assert uploaded["purpose"] == "batch" and uploaded["batch"]["endpoint"] == "/v1/responses"
    assert '"effort": "none"' in uploaded["lines"][0]
    assert cl.batch_status("batch_1") == BATCH_DONE
    res = {r.custom_id: r for r in cl.batch_collect("batch_1")}
    assert res["r0"].completion.output_tokens == 2 and res["r1"].completion is None


def test_google_client_counts_thinking_as_output(specs):
    captured = {}

    def gen(model, contents, config):
        captured.update(model=model, contents=contents, config=config)
        return _Obj(model_version="gemini-3.1-flash-lite",
                    candidates=[_Obj(content=_Obj(parts=[_Obj(text="summary text", thought=None)]), finish_reason="STOP")],
                    usage_metadata=_Obj(prompt_token_count=200, candidates_token_count=30, thoughts_token_count=7))

    sdk = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    c = GoogleClient(specs["google"], sdk=sdk).complete(Prompt("SYS", "USER"), 256)
    assert captured["config"]["temperature"] == 0
    assert captured["config"]["thinking_config"] == {"thinking_level": "MINIMAL"}
    assert captured["config"]["system_instruction"] == "SYS"
    assert (c.text, c.input_tokens, c.output_tokens) == ("summary text", 200, 37)


def test_google_batch_inline_cycle(specs):
    sent = {}
    states = iter(["JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED"])
    inl = [{"metadata": {"key": "r0"}, "response": {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                                                    "usage_metadata": {"prompt_token_count": 3, "candidates_token_count": 1}}},
           {"metadata": {"key": "r1"}, "error": {"code": 500}}]
    batches = SimpleNamespace(
        create=lambda model, src, config: sent.update(model=model, src=src) or _Obj(name="batches/1"),
        get=lambda name: _Obj(state=next(states), dest={"inlined_responses": inl}),
    )
    cl = GoogleClient(specs["google"], sdk=SimpleNamespace(batches=batches))
    assert cl.batch_submit([("r0", Prompt("s", "a")), ("r1", Prompt("s", "b"))], 16) == "batches/1"
    assert sent["model"] == "gemini-3.1-flash-lite" and sent["src"][0]["metadata"] == {"key": "r0"}
    assert cl.batch_status("batches/1") == BATCH_PENDING
    assert cl.batch_status("batches/1") == BATCH_DONE
    res = {r.custom_id: r for r in cl.batch_collect("batches/1")}
    assert res["r0"].completion.text == "ok" and res["r1"].completion is None
