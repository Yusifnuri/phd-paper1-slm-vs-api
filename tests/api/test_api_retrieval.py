from conftest import BagOfWordsEmbedder

from src.api.retrieval import Retriever, embed_cached, random_k, zero_shot


def test_top_k_returns_nearest_neighbours(toy_pool, tmp_path):
    r = Retriever(toy_pool, BagOfWordsEmbedder(), cache_dir=tmp_path)
    assert {e.id for e in r.top_k("exchange rate for euros", 2)} == {"p4", "p5"}
    assert {e.id for e in r.top_k("delete my account", 2)} == {"p6", "p7"}
    best = r.top_k("my card has not arrived yet", 1)[0]
    assert best.id == "p1"  # exact duplicate ranks first


def test_top_k_is_sorted_and_deterministic(toy_pool):
    r = Retriever(toy_pool, BagOfWordsEmbedder())
    q = "top up by bank transfer"
    sims = r.scores([q])[0]
    ids = [e.id for e in r.top_k(q, 5)]
    order = [int(i[1:]) for i in ids]
    assert list(sims[order]) == sorted(sims[order], reverse=True)
    assert ids == [e.id for e in r.top_k(q, 5)]


def test_k_larger_than_pool_and_k_zero(toy_pool):
    r = Retriever(toy_pool, BagOfWordsEmbedder())
    assert len(r.top_k("card", 100)) == len(toy_pool)
    assert r.top_k("card", 0) == []


def test_embeddings_are_cached_on_disk(toy_pool, tmp_path):
    emb = BagOfWordsEmbedder()
    texts = [e.input for e in toy_pool]
    a = embed_cached(emb, texts, tmp_path)
    b = embed_cached(emb, texts, tmp_path)
    assert emb.calls == 1
    assert (a == b).all()
    assert len(list(tmp_path.glob("*.npy"))) == 1
    embed_cached(emb, texts[:-1], tmp_path)  # different pool -> new cache entry
    assert emb.calls == 2


def test_random_k_is_seeded_and_reproducible(toy_pool):
    a = [e.id for e in random_k(toy_pool, 4, seed=13, item_id="t1")]
    b = [e.id for e in random_k(toy_pool, 4, seed=13, item_id="t1")]
    assert a == b
    assert len(set(a)) == 4  # without replacement
    other_seed = [tuple(e.id for e in random_k(toy_pool, 4, seed=s, item_id="t1")) for s in range(20)]
    assert len(set(other_seed)) > 1
    other_items = [tuple(e.id for e in random_k(toy_pool, 4, seed=13, item_id=f"t{i}")) for i in range(20)]
    assert len(set(other_items)) > 1


def test_random_k_independent_of_pool_object_identity(toy_pool):
    a = [e.id for e in random_k(list(toy_pool), 3, seed=1, item_id="x")]
    b = [e.id for e in random_k(tuple(toy_pool), 3, seed=1, item_id="x")]
    assert a == b


def test_zero_shot_is_empty(toy_pool):
    assert zero_shot(toy_pool, 32) == []
