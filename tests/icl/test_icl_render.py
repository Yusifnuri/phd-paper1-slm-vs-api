"""The local ICL prompt is byte-identical to the API arm's and to the fine-tuned arm's zero-shot prompt."""

import pytest

from src.api import prompts, retrieval
from src.api.prompts import Example
from src.api.run import plan_items
from src.icl import render as R
from src.train import task_spec

POOL = [Example(f"p{i}", f"pool text {chr(97 + i % 26)}{i} card", lab)
        for i, lab in enumerate(["card_arrival", "exchange_rate", "lost_or_stolen_card"] * 6)]
ITEMS = [Example("t1", "when does my card arrive", "card_arrival"),
         Example("t2", "what is the exchange rate", "exchange_rate")]
CTX = R.TASKS["banking77"].context


def _api(technique, k, embedder=None, seed=None, items=ITEMS):
    return plan_items(template="classification", context=CTX, technique=technique, k=k, seed=seed, pool=POOL,
                      items=items, embedder=embedder, cache_dir=None, verbalised_confidence=False)


def test_zero_shot_equals_api_and_finetuned_arm():
    sel = R.ShotSelector("zero_shot", [], ITEMS)
    for it, shots, planned in zip(ITEMS, sel.shots(0), _api("zero_shot", 0)):
        p = R.render("banking77", shots, it.input)
        assert (p.system, p.user) == (planned.prompt.system, planned.prompt.user)
        assert p.system == task_spec.system_prompt()
        assert p.user == task_spec.user_prompt(it.input)
        assert R.chat_messages(p) == task_spec.build_messages(it.input)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_retrieved_equals_api_with_nearest_last(embedder, k):
    sel = R.ShotSelector("icl_retrieved", POOL, ITEMS, embedder=embedder)
    api = _api("icl_retrieved", k, embedder)
    for it, shots, planned in zip(ITEMS, sel.shots(k), api):
        p = R.render("banking77", shots, it.input)
        assert (p.system, p.user) == (planned.prompt.system, planned.prompt.user)
        assert [s.id for s in shots] == planned.shot_ids
    # the nearest neighbour is the last shot, right before the query
    nearest = retrieval.Retriever(POOL, embedder).top_k(ITEMS[0].input, 1)[0]
    assert sel.shots(k)[0][-1].id == nearest.id
    assert 0.0 < sel.max_cosine[0] <= 1.0001


@pytest.mark.parametrize("k", [3, 16])
def test_random_equals_api_and_is_order_independent(k):
    sel = R.ShotSelector("icl_random", POOL, ITEMS, seed=7)
    api = _api("icl_random", k, seed=7)
    for it, shots, planned in zip(ITEMS, sel.shots(k), api):
        p = R.render("banking77", shots, it.input)
        assert (p.system, p.user) == (planned.prompt.system, planned.prompt.user)
    rev = R.ShotSelector("icl_random", POOL, ITEMS[::-1], seed=7).shots(k)[::-1]
    assert [[s.id for s in x] for x in rev] == [[s.id for s in x] for x in sel.shots(k)]
    other = R.ShotSelector("icl_random", POOL, ITEMS, seed=8).shots(k)
    assert [[s.id for s in x] for x in other] != [[s.id for s in x] for x in sel.shots(k)]


def test_shared_pieces_come_from_the_api_module():
    # Same objects, not copies: no second implementation of the prompt.
    assert R.build_prompt is prompts.build_prompt
    assert R.random_k is retrieval.random_k
    assert R.TASKS["banking77"].context["task_description"] is prompts.BANKING77_DESCRIPTION


def test_billsum_rendering_k_le_3_matches_api():
    ctx = R.TASKS["billsum"].context
    pool = [Example(f"b{i}", f"BILL {i} text", f"Summary {i}.") for i in range(5)]
    q = [Example("q", "QUERY BILL", "gold")]
    assert R.TASKS["billsum"].ks == (1, 2, 3) and max(R.TASKS["billsum"].ks) <= 3
    shots = R.ShotSelector("icl_random", pool, q, seed=1).shots(3)[0]
    p = R.render("billsum", shots, "QUERY BILL")
    planned = plan_items(template="summarisation", context=ctx, technique="icl_random", k=3, seed=1, pool=pool,
                         items=q, embedder=None, cache_dir=None, verbalised_confidence=False)[0]
    assert (p.system, p.user) == (planned.prompt.system, planned.prompt.user)
    assert p.user.endswith("Document:\nQUERY BILL\n\nSummary:")
    assert R.TASKS["billsum"].max_new_tokens == 1024


def test_parser_is_the_finetuned_arms():
    t = R.TASKS["banking77"]
    for text, want in [("card_arrival", "card_arrival"), ("Label: Card-Arrival\nextra", "card_arrival"),
                       ("reverted_card_payment?", "reverted_card_payment?"),
                       ("refund_not_showing_up", "Refund_not_showing_up"), ("", None), ("card arrival please", None)]:
        assert t.parse(text) == (want, want is not None)
        assert t.parse(text)[0] == task_spec.parse_label(text)
    for lab in task_spec.BANKING77_LABELS:
        assert t.parse(lab) == (lab, True)


def test_selector_validation():
    with pytest.raises(ValueError):
        R.ShotSelector("icl_random", POOL, ITEMS)
    with pytest.raises(ValueError):
        R.ShotSelector("icl_retrieved", POOL, ITEMS)
    with pytest.raises(ValueError):
        R.ShotSelector("nope", POOL, ITEMS)
    with pytest.raises(ValueError):
        R.ShotSelector("icl_random", POOL + POOL[:1], ITEMS, seed=1)
