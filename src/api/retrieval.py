"""Choose in-context examples from a task's N-example training pool.

Three techniques, matching the `technique` field of the output records:

- ``icl_retrieved``: the k pool examples most similar to the test input, by
  cosine similarity of sentence embeddings.
- ``icl_random``: k pool examples drawn uniformly without replacement. The draw
  is seeded per (seed, item_id), so it is reproducible and independent of the
  order in which items are processed (which matters for resumed runs).
- ``zero_shot``: no examples.

Embedding model: BAAI/bge-small-en-v1.5.

- Licence: MIT on the model card (checked 2026-11-21), so commercial use and
  publication of results are allowed.
- Size: 33.4M parameters and 384-dimensional vectors. It runs on a laptop CPU,
  and embedding a 5,000-example pool takes minutes, not hours, without a GPU.
- Retrieval quality is strong for its size. The alternative,
  sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), is similar in size but
  older and weaker.
- Maximum input is 512 tokens. Longer texts (BillSum bills, about 2.6k tokens)
  are truncated by the encoder, so similarity is judged on the opening of the
  bill, which carries the title and purpose clauses. This is a stated
  limitation, not a bug.
- The card's query instruction prefix is meant for short-query-to-passage
  search. Here both sides are the same kind of text (test input vs pool
  input), so no prefix is used, and the embeddings are symmetric.

Embeddings are cached on disk as .npy files, keyed by the SHA-256 of the
embedder name and the exact list of texts, so a changed pool never reuses a
stale cache.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from src.api.prompts import Example

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Pinned Hub revision (commit sha reported by huggingface.co/api/models/BAAI/bge-small-en-v1.5
# on 2026-11-22, lastModified 2024-02-22; licence mit). Protocol section 2: never revision=None.
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TECHNIQUES = ("icl_retrieved", "icl_random", "zero_shot")


class Embedder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers (imported lazily)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 revision: str | None = DEFAULT_EMBEDDING_REVISION,
                 device: str | None = None, batch_size: int = 64):
        self.name = model_name if revision is None else f"{model_name}@{revision}"
        self._model_name = model_name
        self._revision = revision
        self._device = device
        self._batch_size = batch_size
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, revision=self._revision, device=self._device)
        vecs = self._model.encode(list(texts), batch_size=self._batch_size,
                                  normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def _cache_key(embedder_name: str, texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(embedder_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def embed_cached(embedder: Embedder, texts: Sequence[str], cache_dir: str | Path | None) -> np.ndarray:
    """Embed `texts`, reading and writing a disk cache when `cache_dir` is set."""
    if cache_dir is None:
        return np.asarray(embedder.encode(texts), dtype=np.float32)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(embedder.name, texts)
    path = cache_dir / f"{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = np.asarray(embedder.encode(texts), dtype=np.float32)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vecs)
    tmp.replace(path)
    meta = {"embedder": embedder.name, "n": len(texts), "dim": int(vecs.shape[1]) if vecs.ndim == 2 else None}
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return vecs


def _l2_normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Retriever:
    """k-nearest-neighbour retrieval over a fixed pool."""

    def __init__(self, pool: Sequence[Example], embedder: Embedder, cache_dir: str | Path | None = None):
        if not pool:
            raise ValueError("empty pool")
        ids = [ex.id for ex in pool]
        if len(set(ids)) != len(ids):
            raise ValueError("pool example ids must be unique")
        self.pool = list(pool)
        self.embedder = embedder
        self.cache_dir = cache_dir
        self._pool_vecs = _l2_normalise(embed_cached(embedder, [ex.input for ex in self.pool], cache_dir))

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        q = _l2_normalise(embed_cached(self.embedder, list(queries), self.cache_dir))
        return q @ self._pool_vecs.T

    def top_k_many(self, queries: Sequence[str], k: int) -> list[list[Example]]:
        """For each query, the k most similar pool examples, most similar first.
        Ties are broken by pool order (stable sort), so results are deterministic."""
        if k <= 0:
            return [[] for _ in queries]
        k = min(k, len(self.pool))
        sims = self.scores(queries)
        out = []
        for row in sims:
            order = np.argsort(-row, kind="stable")[:k]
            out.append([self.pool[i] for i in order])
        return out

    def top_k(self, query: str, k: int) -> list[Example]:
        return self.top_k_many([query], k)[0]


def random_k(pool: Sequence[Example], k: int, seed: int, item_id: str) -> list[Example]:
    """k examples sampled without replacement, seeded by (seed, item_id)."""
    if k <= 0:
        return []
    digest = hashlib.sha256(f"{seed}\x00{item_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    k = min(k, len(pool))
    return rng.sample(list(pool), k)


def zero_shot(*_args, **_kwargs) -> list[Example]:
    return []
