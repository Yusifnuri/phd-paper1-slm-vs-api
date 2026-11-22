"""Provider clients behind one interface.

Each client turns a `Prompt` into a `Completion` (text, provider-billed input
and output tokens, latency, raw response). Each also implements the provider's
batch API in three steps (submit, status, collect), plus a shared `poll`.

Settings, checked against the provider docs on 2026-11-21:

- **Anthropic (claude-haiku-5-5).**
  - Calls Messages API `messages.create` with `thinking={"type": "disabled"}`. Thinking is otherwise on by default and billed as output.
  - Sends no `temperature`: Haiku 5.5 rejects non-default sampling parameters.
  - Batch: `messages.batches.create / retrieve / results`. `processing_status == "ended"` means done, and results arrive in any order, keyed by `custom_id`.
- **OpenAI (gpt-6-luna).**
  - Calls Responses API `responses.create` with `reasoning={"effort": "none"}` and `store=False`.
  - Sends no `temperature`: the model page does not state that it is supported.
  - Billed output tokens include any reasoning tokens.
  - Batch: upload JSONL (`purpose="batch"`), then `batches.create(endpoint="/v1/responses", completion_window="24h")`, poll `status`, and download `output_file_id` and `error_file_id`.
- **Google (gemini-3.1-flash-lite).**
  - Calls `models.generate_content` with `temperature=0` and `thinking_level=minimal`. Thinking cannot be fully disabled on 3.1 Flash-Lite.
  - Billed output is `candidates_token_count + thoughts_token_count`.
  - Batch: `batches.create(model, src=...)`, polling `state`.
    - Requests go inline below about 15 MB (the docs' inline limit is about 20 MB).
    - Larger jobs go through an uploaded JSONL file of `{"key", "request"}` lines.

Clients never read keys themselves. The SDKs read ANTHROPIC_API_KEY,
OPENAI_API_KEY and GEMINI_API_KEY / GOOGLE_API_KEY from the environment, which
`run.py` fills from `.env`. Every SDK object can be injected, so tests use
fakes and never touch the network.
"""

from __future__ import annotations

import json
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from src.api.config import ModelSpec
from src.api.prompts import Prompt


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float | None
    raw: dict[str, Any]
    model_id: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Outcome for one request in a batch: a completion, or an error string."""

    custom_id: str
    completion: Completion | None
    error: str | None = None


BATCH_PENDING, BATCH_DONE, BATCH_FAILED = "pending", "done", "failed"


class BatchFailed(RuntimeError):
    pass


def _dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn(mode="json") if attr == "model_dump" else fn()
            except TypeError:
                return fn()
    return {"repr": repr(obj)}


class ChatClient(ABC):
    provider: str

    def __init__(self, spec: ModelSpec, sdk: Any = None, clock: Callable[[], float] = time.perf_counter):
        self.spec = spec
        self._sdk = sdk
        self._clock = clock

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def sdk(self) -> Any:
        if self._sdk is None:
            self._sdk = self._make_sdk()
        return self._sdk

    @abstractmethod
    def _make_sdk(self) -> Any: ...

    @abstractmethod
    def complete(self, prompt: Prompt, max_output_tokens: int) -> Completion: ...

    @abstractmethod
    def batch_submit(self, requests: Sequence[tuple[str, Prompt]], max_output_tokens: int) -> str:
        """Submit (custom_id, prompt) pairs; return the provider's batch id."""

    @abstractmethod
    def batch_status(self, batch_id: str) -> str:
        """Return BATCH_PENDING, BATCH_DONE or BATCH_FAILED."""

    @abstractmethod
    def batch_collect(self, batch_id: str) -> list[BatchResult]: ...

    def batch_poll(self, batch_id: str, interval_s: float = 60.0, timeout_s: float = 26 * 3600,
                   sleep: Callable[[float], None] = time.sleep) -> str:
        waited = 0.0
        while True:
            status = self.batch_status(batch_id)
            if status != BATCH_PENDING:
                return status
            if waited >= timeout_s:
                raise TimeoutError(f"batch {batch_id} still pending after {waited:.0f}s")
            sleep(interval_s)
            waited += interval_s

    def _timed(self, fn: Callable[[], Any]) -> tuple[Any, float]:
        t0 = self._clock()
        out = fn()
        return out, (self._clock() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(ChatClient):
    provider = "anthropic"

    def _make_sdk(self) -> Any:
        import anthropic

        return anthropic.Anthropic()

    def _params(self, prompt: Prompt, max_output_tokens: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_output_tokens,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
            "thinking": {"type": "disabled"},
        }
        if self.spec.supports_temperature:
            params["temperature"] = 0
        return params

    def _to_completion(self, msg: dict[str, Any], latency_ms: float | None) -> Completion:
        text = "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")
        usage = msg.get("usage") or {}
        return Completion(
            text=text,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
            raw=msg,
            model_id=msg.get("model") or self.model_id,
            extra={"stop_reason": msg.get("stop_reason")},
        )

    def complete(self, prompt: Prompt, max_output_tokens: int) -> Completion:
        resp, ms = self._timed(lambda: self.sdk.messages.create(**self._params(prompt, max_output_tokens)))
        return self._to_completion(_dump(resp), ms)

    def batch_submit(self, requests, max_output_tokens):
        batch = self.sdk.messages.batches.create(
            requests=[{"custom_id": cid, "params": self._params(p, max_output_tokens)} for cid, p in requests]
        )
        return _dump(batch)["id"]

    def batch_status(self, batch_id):
        b = _dump(self.sdk.messages.batches.retrieve(batch_id))
        return BATCH_DONE if b.get("processing_status") == "ended" else BATCH_PENDING

    def batch_collect(self, batch_id):
        out = []
        for item in self.sdk.messages.batches.results(batch_id):
            d = _dump(item)
            result = d.get("result") or {}
            if result.get("type") == "succeeded":
                out.append(BatchResult(d["custom_id"], self._to_completion(result["message"], None)))
            else:
                out.append(BatchResult(d["custom_id"], None, error=json.dumps(result)[:2000]))
        return out


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _openai_text(body: dict[str, Any]) -> str:
    if body.get("output_text"):
        return body["output_text"]
    parts = []
    for item in body.get("output") or []:
        if item.get("type") == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "".join(parts)


class OpenAIClient(ChatClient):
    provider = "openai"

    def _make_sdk(self) -> Any:
        import openai

        return openai.OpenAI()

    def _body(self, prompt: Prompt, max_output_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "instructions": prompt.system,
            "input": prompt.user,
            "reasoning": {"effort": "none"},
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if self.spec.supports_temperature:
            body["temperature"] = 0
        return body

    def _to_completion(self, body: dict[str, Any], latency_ms: float | None) -> Completion:
        usage = body.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        return Completion(
            text=_openai_text(body),
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
            raw=body,
            model_id=body.get("model") or self.model_id,
            extra={"reasoning_tokens": details.get("reasoning_tokens"), "status": body.get("status")},
        )

    def complete(self, prompt: Prompt, max_output_tokens: int) -> Completion:
        resp, ms = self._timed(lambda: self.sdk.responses.create(**self._body(prompt, max_output_tokens)))
        body = _dump(resp)
        if "output_text" not in body and getattr(resp, "output_text", None) is not None:
            body["output_text"] = resp.output_text
        return self._to_completion(body, ms)

    def batch_submit(self, requests, max_output_tokens):
        lines = [
            json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/responses",
                        "body": self._body(p, max_output_tokens)})
            for cid, p in requests
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "batch_input.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with open(path, "rb") as fh:
                uploaded = self.sdk.files.create(file=fh, purpose="batch")
        batch = self.sdk.batches.create(
            input_file_id=_dump(uploaded)["id"], endpoint="/v1/responses", completion_window="24h"
        )
        return _dump(batch)["id"]

    def batch_status(self, batch_id):
        status = _dump(self.sdk.batches.retrieve(batch_id)).get("status")
        if status == "completed":
            return BATCH_DONE
        if status in ("failed", "expired", "cancelled"):
            # Expired batches may still have partial output; collect() reads whatever exists.
            return BATCH_DONE if status == "expired" else BATCH_FAILED
        return BATCH_PENDING

    def _read_file(self, file_id: str | None) -> list[dict[str, Any]]:
        if not file_id:
            return []
        content = self.sdk.files.content(file_id)
        text = content.text if hasattr(content, "text") else str(content)
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    def batch_collect(self, batch_id):
        b = _dump(self.sdk.batches.retrieve(batch_id))
        out = []
        for line in self._read_file(b.get("output_file_id")):
            resp = line.get("response") or {}
            if resp.get("status_code") == 200 and resp.get("body"):
                out.append(BatchResult(line["custom_id"], self._to_completion(resp["body"], None)))
            else:
                out.append(BatchResult(line["custom_id"], None, error=json.dumps(line.get("error") or resp)[:2000]))
        for line in self._read_file(b.get("error_file_id")):
            out.append(BatchResult(line["custom_id"], None, error=json.dumps(line.get("error"))[:2000]))
        return out


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

_GOOGLE_INLINE_LIMIT_BYTES = 15_000_000
_GOOGLE_DONE = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
_GOOGLE_FAILED = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def _g(d: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Read a field that may be snake_case (SDK dump) or camelCase (REST JSON)."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


class GoogleClient(ChatClient):
    provider = "google"

    def _make_sdk(self) -> Any:
        from google import genai

        return genai.Client()

    def _config(self, prompt: Prompt, max_output_tokens: int) -> dict[str, Any]:
        cfg: dict[str, Any] = {"system_instruction": prompt.system, "max_output_tokens": max_output_tokens}
        if self.spec.supports_temperature:
            cfg["temperature"] = 0
        if self.spec.thinking_level:
            cfg["thinking_config"] = {"thinking_level": self.spec.thinking_level.upper()}
        return cfg

    def _rest_request(self, prompt: Prompt, max_output_tokens: int) -> dict[str, Any]:
        gen: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
        if self.spec.supports_temperature:
            gen["temperature"] = 0
        if self.spec.thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": self.spec.thinking_level.upper()}
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt.user}]}],
            "systemInstruction": {"parts": [{"text": prompt.system}]},
            "generationConfig": gen,
        }

    def _to_completion(self, resp: dict[str, Any], latency_ms: float | None) -> Completion:
        texts = []
        cands = _g(resp, "candidates", default=[]) or []
        if cands:
            content = _g(cands[0], "content", default={}) or {}
            for part in _g(content, "parts", default=[]) or []:
                if not part.get("thought") and part.get("text"):
                    texts.append(part["text"])
        um = _g(resp, "usage_metadata", "usageMetadata", default={}) or {}
        prompt_tok = int(_g(um, "prompt_token_count", "promptTokenCount", default=0) or 0)
        cand_tok = int(_g(um, "candidates_token_count", "candidatesTokenCount", default=0) or 0)
        thought_tok = int(_g(um, "thoughts_token_count", "thoughtsTokenCount", default=0) or 0)
        finish = _g(cands[0], "finish_reason", "finishReason") if cands else None
        return Completion(
            text="".join(texts),
            input_tokens=prompt_tok,
            output_tokens=cand_tok + thought_tok,
            latency_ms=latency_ms,
            raw=resp,
            model_id=_g(resp, "model_version", "modelVersion", default=None) or self.model_id,
            extra={"thoughts_tokens": thought_tok, "finish_reason": finish},
        )

    def complete(self, prompt: Prompt, max_output_tokens: int) -> Completion:
        resp, ms = self._timed(lambda: self.sdk.models.generate_content(
            model=self.model_id, contents=prompt.user, config=self._config(prompt, max_output_tokens)))
        return self._to_completion(_dump(resp), ms)

    def batch_submit(self, requests, max_output_tokens):
        size = sum(p.n_chars for _, p in requests)
        if size < _GOOGLE_INLINE_LIMIT_BYTES:
            src: Any = [
                {"contents": [{"role": "user", "parts": [{"text": p.user}]}],
                 "config": self._config(p, max_output_tokens),
                 "metadata": {"key": cid}}
                for cid, p in requests
            ]
        else:
            lines = [json.dumps({"key": cid, "request": self._rest_request(p, max_output_tokens)})
                     for cid, p in requests]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "batch_input.jsonl"
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                uploaded = self.sdk.files.upload(file=str(path),
                                                 config={"display_name": path.name, "mime_type": "jsonl"})
            src = _dump(uploaded).get("name") or uploaded.name
        job = self.sdk.batches.create(model=self.model_id, src=src, config={"display_name": "slm-vs-api"})
        return _dump(job).get("name") or job.name

    def _job(self, batch_id: str) -> dict[str, Any]:
        return _dump(self.sdk.batches.get(name=batch_id))

    def batch_status(self, batch_id):
        state = self._job(batch_id).get("state")
        state = state.get("name") if isinstance(state, dict) else str(state).split(".")[-1]
        if state in _GOOGLE_DONE:
            return BATCH_DONE
        if state in _GOOGLE_FAILED:
            return BATCH_FAILED
        return BATCH_PENDING

    def batch_collect(self, batch_id):
        dest = self._job(batch_id).get("dest") or {}
        out = []
        inlined = dest.get("inlined_responses")
        if inlined:
            for item in inlined:
                key = (item.get("metadata") or {}).get("key")
                if item.get("response"):
                    out.append(BatchResult(key, self._to_completion(item["response"], None)))
                else:
                    out.append(BatchResult(key, None, error=json.dumps(item.get("error"))[:2000]))
        elif dest.get("file_name"):
            data = self.sdk.files.download(file=dest["file_name"])
            text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                line = json.loads(ln)
                if line.get("response"):
                    out.append(BatchResult(line.get("key"), self._to_completion(line["response"], None)))
                else:
                    out.append(BatchResult(line.get("key"), None, error=json.dumps(line.get("error"))[:2000]))
        return out


CLIENTS: dict[str, type[ChatClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "google": GoogleClient,
}


def make_client(spec: ModelSpec, sdk: Any = None) -> ChatClient:
    return CLIENTS[spec.client_kind](spec, sdk=sdk)
