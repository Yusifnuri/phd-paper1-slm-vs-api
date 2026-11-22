"""Toy embedder, toy data and fake generator for the ICL tests (offline)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.icl.generate import Generation  # noqa: E402

WORDS = {"card_arrival": "card arrive when", "exchange_rate": "exchange rate euro",
         "lost_or_stolen_card": "lost stolen phone"}


class ToyEmbedder:
    """Bag-of-characters vectors: deterministic, needs no model download."""
    name = "toy@0"

    def encode(self, texts):
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out


def toy_loader():
    texts, labels = [], []
    for j in range(1200):
        lab = list(WORDS)[j % 3]
        texts.append(f"{WORDS[lab]} {'x' * (j % 7)}")
        labels.append(lab)
    return texts, labels


def _label_for(query: str) -> str:
    for lab, words in WORDS.items():
        if words in query:
            return lab
    return "card_arrival"


class FakeGenerator:
    """Answers correctly when the prompt has at least `min_shots` shots, otherwise always 'card_arrival'.
    Records every call so tests can check what was sent."""
    name = "fake"

    def __init__(self, min_shots: int = 8, junk_every: int = 0):
        self.min_shots, self.junk_every = min_shots, junk_every
        self.calls = []  # (prompts, max_new_tokens, choices)

    def generate(self, prompts, max_new_tokens, choices=None):
        self.calls.append((list(prompts), max_new_tokens, choices))
        out = []
        for i, p in enumerate(prompts):
            query = p.user.rsplit("Input: ", 1)[1].rsplit("\nLabel:", 1)[0]
            n_shots = p.user.count("Label:") - 1
            text = _label_for(query) if n_shots >= self.min_shots else "card_arrival"
            if choices is not None:
                assert text in choices
            elif self.junk_every and i % self.junk_every == 0:
                text = "I am not sure"
            out.append(Generation(text=text, input_tokens=len(p.system.split()) + len(p.user.split()),
                                  output_tokens=3, finish_reason="stop"))
        return out


# ---------------------------------------------------------------- BillSum toys

TOPICS = ("water", "roads", "schools", "tax", "veterans")


def toy_billsum_loader():
    """18,949 short bills ('bill about <topic> <n>') whose gold summary names the topic."""
    texts, summaries = [], []
    for j in range(18949):
        topic = TOPICS[j % len(TOPICS)]
        texts.append(f"bill about {topic} {topic} act number {j % 13}")
        summaries.append(f"This bill concerns {topic}")
    return texts, summaries


class FakeSumGenerator:
    """Writes the gold-style summary when the prompt has at least `min_shots` shots, otherwise an unrelated
    sentence; `empty_every` makes every i-th output empty; `cap_every` marks every i-th as cut by the cap."""
    name = "fake-sum"

    def __init__(self, min_shots: int = 2, empty_every: int = 0, cap_every: int = 0):
        self.min_shots, self.empty_every, self.cap_every = min_shots, empty_every, cap_every
        self.calls = []

    def generate(self, prompts, max_new_tokens, choices=None):
        self.calls.append((list(prompts), max_new_tokens, choices))
        out = []
        for i, p in enumerate(prompts):
            query = p.user.rsplit("Document:\n", 1)[1].rsplit("\n\nSummary:", 1)[0]
            n_shots = p.user.count("Summary:\n")
            topic = query.split()[2]
            text = f"This bill concerns {topic}" if n_shots >= self.min_shots else "Nothing relevant is said here"
            if self.empty_every and i % self.empty_every == 0:
                text = ""
            capped = bool(self.cap_every and i % self.cap_every == 0)
            out.append(Generation(text=text, input_tokens=len(p.user.split()), output_tokens=1024 if capped else 5,
                                  finish_reason="length" if capped else "stop"))
        return out
