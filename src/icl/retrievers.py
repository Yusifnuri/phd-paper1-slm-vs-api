"""Retrievers for choosing BillSum in-context examples, and the offline comparison of them.

Protocol section 2: bge-small-en-v1.5 sees only the first 512 tokens of a bill (about a quarter of an
average ~1.9k-token bill), so similarity is judged on the opening. Before any test read the
protocol compares three selectors by the ROUGE-L of the retrieved examples' gold summaries
against the query's gold summary on 300 validation items:

    bge-head   pinned bge embeddings of the bill head (the encoder truncates at 512 tokens)   [default]
    bm25       Okapi BM25 over the whole bill text
    tfidf      TF-IDF (sublinear tf, smoothed idf) cosine over the whole bill text

`compare_retrievers` is pure (an injected embedder and rouge function make it testable offline); the CLI
(`--compare-retrievers`) writes its result as a results line and the researcher passes the winner to
`--retriever` for tuning. The chosen retriever is frozen in the arm, and finals must use it.

BM25 and TF-IDF are implemented here with scipy.sparse (scipy ships with sentence-transformers and is
pinned in requirements/eval.txt): no scikit-learn and no rank_bm25 dependency. Fitting uses the pool only
(idf and average length are computed on the pool), so a query never influences its own statistics.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.api.prompts import Example
from src.api.retrieval import Embedder, embed_cached

RETRIEVERS = ("bge-head", "bm25", "tfidf")
DEFAULT_RETRIEVER = "bge-head"
BM25_K1, BM25_B = 1.5, 0.75
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


def _counts_matrix(docs: Sequence[Sequence[str]], vocab: Dict[str, int]):
    """Sparse (n_docs x |vocab|) term-count matrix; out-of-vocabulary terms are dropped."""
    from scipy import sparse

    rows, cols, vals = [], [], []
    for r, toks in enumerate(docs):
        counts: Dict[int, int] = {}
        for t in toks:
            j = vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        rows.extend([r] * len(counts))
        cols.extend(counts.keys())
        vals.extend(counts.values())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=np.float64)


def _fit(pool_texts: Sequence[str], query_texts: Sequence[str]):
    pool_tok = [tokenize(t) for t in pool_texts]
    vocab: Dict[str, int] = {}
    for toks in pool_tok:
        for t in toks:
            vocab.setdefault(t, len(vocab))
    pool_c = _counts_matrix(pool_tok, vocab)
    query_c = _counts_matrix([tokenize(t) for t in query_texts], vocab)
    df = np.asarray((pool_c > 0).sum(axis=0)).ravel()
    return pool_c, query_c, df


def tfidf_sims(pool_texts: Sequence[str], query_texts: Sequence[str]) -> np.ndarray:
    """Cosine of TF-IDF vectors, shape (n_queries, n_pool). tf = 1 + ln(count); idf = ln((1+N)/(1+df)) + 1."""
    from scipy import sparse

    pool_c, query_c, df = _fit(pool_texts, query_texts)
    idf = np.log((1.0 + pool_c.shape[0]) / (1.0 + df)) + 1.0

    def weigh(m):
        m = m.copy()
        m.data = 1.0 + np.log(m.data)
        m = m.multiply(idf.reshape(1, -1)).tocsr()
        norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norm[norm == 0] = 1.0
        return sparse.diags(1.0 / norm) @ m

    return np.asarray((weigh(query_c) @ weigh(pool_c).T).todense(), dtype=np.float32)


def bm25_sims(pool_texts: Sequence[str], query_texts: Sequence[str],
              k1: float = BM25_K1, b: float = BM25_B) -> np.ndarray:
    """Okapi BM25 score of every pool bill for every query, shape (n_queries, n_pool).
    idf = ln(1 + (N - df + 0.5) / (df + 0.5)); query terms count once each."""
    pool_c, query_c, df = _fit(pool_texts, query_texts)
    n = pool_c.shape[0]
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
    dl = np.asarray(pool_c.sum(axis=1)).ravel()
    avgdl = dl.mean() if dl.size and dl.mean() > 0 else 1.0
    w = pool_c.tocoo()
    denom = w.data + k1 * (1.0 - b + b * dl[w.row] / avgdl)
    from scipy import sparse

    weights = sparse.csr_matrix((idf[w.col] * w.data * (k1 + 1.0) / denom, (w.row, w.col)), shape=pool_c.shape)
    q = query_c.copy()
    q.data = np.ones_like(q.data)
    return np.asarray((q @ weights.T).todense(), dtype=np.float32)


def bge_sims(pool_texts: Sequence[str], query_texts: Sequence[str], embedder: Embedder,
             cache_dir: Optional[str] = None) -> np.ndarray:
    """Cosine of the embedder's vectors (same normalisation as ShotSelector)."""
    def unit(m: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(m, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return m / nrm

    pv = unit(embed_cached(embedder, list(pool_texts), cache_dir))
    qv = unit(embed_cached(embedder, list(query_texts), cache_dir))
    return qv @ pv.T


def similarity(name: str, pool_texts: Sequence[str], query_texts: Sequence[str], *,
               embedder: Optional[Embedder] = None, cache_dir: Optional[str] = None) -> np.ndarray:
    """(n_queries, n_pool) similarity matrix of retriever `name`."""
    if name == "bge-head":
        if embedder is None:
            raise ValueError("bge-head needs an embedder")
        return bge_sims(pool_texts, query_texts, embedder, cache_dir)
    if name == "bm25":
        return bm25_sims(pool_texts, query_texts)
    if name == "tfidf":
        return tfidf_sims(pool_texts, query_texts)
    raise ValueError(f"retriever must be one of {RETRIEVERS}, got {name!r}")


def _topk(row: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-row, kind="stable")[:k]  # ties by pool order, as ShotSelector


def compare_retrievers(
    pool: Sequence[Example], queries: Sequence[Example], *, embedder: Optional[Embedder] = None,
    cache_dir: Optional[str] = None, names: Sequence[str] = RETRIEVERS, k: int = 3, seed: int = 0,
    rouge_fn: Optional[Callable[[Sequence[str], Sequence[str]], Sequence[float]]] = None,
    sims: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Score retrievers by the ROUGE-L of retrieved vs gold summaries (Example.output is the gold summary).

    For each query and retriever, the top-k pool bills are retrieved; each retrieved bill's gold summary is
    scored against the query's gold summary. Reported per retriever: mean over queries of the top-1 score and
    of the mean top-k score. `best` is the retriever with the highest top-k mean (ties: earlier in `names`,
    so bge-head wins ties). A seeded random-pool reference (same k) is reported but never selectable.
    `sims` (name -> matrix) may be passed to skip recomputation. Validation items only; no test data.
    """
    if not pool or not queries:
        raise ValueError("pool and queries must be non-empty")
    if rouge_fn is None:
        from src.train.summarise import rouge_l_scores as rouge_fn  # noqa: F811
    k = min(k, len(pool))
    pool_texts, query_texts = [e.input for e in pool], [e.input for e in queries]
    gold = [q.output for q in queries]

    def score(idx_lists: Sequence[Sequence[int]]) -> Dict[str, float]:
        flat_pred = [pool[j].output for idxs in idx_lists for j in idxs]
        flat_gold = [g for g, idxs in zip(gold, idx_lists) for _ in idxs]
        s = np.asarray(rouge_fn(flat_pred, flat_gold), dtype=float).reshape(len(idx_lists), k)
        return {"top1": float(s[:, 0].mean()), "topk": float(s.mean())}

    out: Dict[str, Any] = {"k": k, "n_queries": len(queries), "pool_size": len(pool), "retrievers": {}}
    for name in names:
        m = (sims or {}).get(name)
        if m is None:
            m = similarity(name, pool_texts, query_texts, embedder=embedder, cache_dir=cache_dir)
        out["retrievers"][name] = score([_topk(row, k) for row in m])
    rng = random.Random(seed)
    out["random_reference"] = score([rng.sample(range(len(pool)), k) for _ in queries])
    out["best"] = max(names, key=lambda n: (out["retrievers"][n]["topk"], -list(names).index(n)))
    out["criterion"] = "mean ROUGE-L of the top-k retrieved gold summaries against the query's gold summary"
    return out
