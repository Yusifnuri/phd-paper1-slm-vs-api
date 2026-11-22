"""Shared fixtures for the API-arm tests. Nothing here touches the network."""

from __future__ import annotations

import hashlib
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api.clients import BATCH_DONE, BatchResult, ChatClient, Completion  # noqa: E402
from src.api.config import load_specs  # noqa: E402
from src.api.prompts import Example  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test tries to open a socket."""
    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


class BagOfWordsEmbedder:
    """Deterministic toy embedder: hashed bag of lower-cased words."""

    name = "toy-bow-64"

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim
                out[i, h] += 1.0
        return out


class FakeClient(ChatClient):
    """Echo-style client: answers with a fixed label, counts calls, simulates batches."""

    provider = "fake"

    def __init__(self, spec, answer="card_arrival", fail_ids=()):
        super().__init__(spec, sdk=object())
        self.answer = answer
        self.calls = 0
        self.batches: dict[str, list] = {}
        self.submits = 0
        self.fail_ids = set(fail_ids)

    def _make_sdk(self):
        raise AssertionError("FakeClient never builds a real SDK")

    def _comp(self, prompt):
        return Completion(text=self.answer, input_tokens=len(prompt.user) // 4, output_tokens=3,
                          latency_ms=12.5, raw={"fake": True}, model_id=self.model_id, extra={"stop_reason": "end_turn"})

    def complete(self, prompt, max_output_tokens):
        self.calls += 1
        if any(prompt.user.endswith(f"Input: {q}\nLabel:") for q in self.fail_ids):
            raise RuntimeError("simulated provider error")
        return self._comp(prompt)

    def batch_submit(self, requests, max_output_tokens):
        self.submits += 1
        bid = f"batch_{self.submits}"
        self.batches[bid] = list(requests)
        return bid

    def batch_status(self, batch_id):
        return BATCH_DONE

    def batch_collect(self, batch_id):
        return [BatchResult(cid, Completion(text=self.answer, input_tokens=10, output_tokens=2, latency_ms=None,
                                            raw={}, model_id=self.model_id))
                for cid, _ in reversed(self.batches[batch_id])]  # out of order on purpose


@pytest.fixture
def specs():
    return load_specs()


@pytest.fixture
def toy_pool():
    return [
        Example("p0", "where is my new card", "card_arrival"),
        Example("p1", "my card has not arrived yet", "card_arrival"),
        Example("p2", "how do I top up by bank transfer", "top_up_by_bank_transfer"),
        Example("p3", "top up failed with my bank transfer", "top_up_failed"),
        Example("p4", "what is the exchange rate today", "exchange_rate"),
        Example("p5", "exchange rate for euros to pounds", "exchange_rate"),
        Example("p6", "I want to close my account", "terminate_account"),
        Example("p7", "please delete my account now", "terminate_account"),
    ]


@pytest.fixture
def toy_test():
    return [Example(f"t{i}", q, g) for i, (q, g) in enumerate([
        ("my new card has not arrived", "card_arrival"),
        ("exchange rate euros", "exchange_rate"),
        ("close my account please", "terminate_account"),
        ("bank transfer top up", "top_up_by_bank_transfer"),
        ("where is my card", "card_arrival"),
    ])]


@pytest.fixture
def labels():
    return ["card_arrival", "top_up_by_bank_transfer", "top_up_failed", "exchange_rate", "terminate_account"]


@pytest.fixture
def cls_context(labels):
    return {"task_description": "Classify banking messages by intent.", "labels": labels}
