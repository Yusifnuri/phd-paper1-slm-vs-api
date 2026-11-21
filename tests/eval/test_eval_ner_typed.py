"""Unit tests for src/eval/ner_typed.py, ported from the thesis repository's tests/test_ner_typed.py.

The end-to-end test that reproduced the thesis's stored scores is not ported
(it needs the thesis results and CoNLL-2003). The last two tests are new.
"""

import pytest

from src.eval.ner_typed import (
    bio_tags, bio_to_entities, lenient_sentence_f1, micro_prf, parse_api_entities,
)

TOKENS = ["SOCCER", "-", "JAPAN", "GET", "LUCKY", "WIN", ",", "CHINA", "IN", "SURPRISE", "DEFEAT", "."]
GOLD = ["O", "O", "B-LOC", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O"]  # CoNLL's own labels


def test_bio_decoding_keeps_types():
    assert bio_to_entities(TOKENS, GOLD) == [("LOC", "JAPAN"), ("PER", "CHINA")]
    assert bio_to_entities(["New", "York", "Times"], ["B-ORG", "I-ORG", "I-ORG"]) == [("ORG", "New York Times")]
    assert bio_to_entities(["a", "b"], ["I-PER", "O"]) == []  # orphan I- is O, as in the thesis decoder


def test_bio_tags_parses_generated_ids():
    assert bio_tags("[5, 0, 6]") == ["B-LOC", "O", "I-LOC"]
    assert bio_tags("no ids here") is None


def test_typing_is_what_separates_the_instruments():
    gold = [[("LOC", "JAPAN"), ("PER", "CHINA")]]
    pred = [[("LOC", "Japan"), ("LOC", "China")]]
    assert lenient_sentence_f1(gold[0], pred[0]) == 1.0
    assert micro_prf(gold, pred, typed=False)["f1"] == 1.0
    assert micro_prf(gold, pred)["f1"] == 0.5
    assert micro_prf(gold, pred, case_sensitive=True)["f1"] == 0.0


def test_micro_counts_duplicates():
    gold = [[("LOC", "China"), ("LOC", "China")]]
    assert micro_prf(gold, [[("LOC", "China")]])["recall"] == 0.5


def test_typed_micro_equals_seqeval_on_token_aligned_spans():
    seqeval_metrics = pytest.importorskip("seqeval.metrics")
    pred = ["O", "O", "B-LOC", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O"]
    ours = micro_prf([bio_to_entities(TOKENS, GOLD)], [bio_to_entities(TOKENS, pred)], case_sensitive=True)
    assert ours["f1"] == seqeval_metrics.f1_score([GOLD], [pred]) == 0.5


def test_api_parsing():
    assert parse_api_entities('```json\n{"LOC": ["Japan"], "PER": []}\n```') == [("LOC", "Japan")]
    assert parse_api_entities('{"PER": ["Nadim Ladki"], "ORG": null}') == [("PER", "Nadim Ladki")]
    assert parse_api_entities('{"PER": ["truncated') is None


def test_api_parsing_with_custom_types_and_bare_strings():
    # A bare string is one entity, not one entity per character.
    assert parse_api_entities('{"PER": "Ann"}') == [("PER", "Ann")]
    assert parse_api_entities('{"PER": {"x": 1}}') is None
    assert parse_api_entities('{"person": ["Ann"], "location": ["Baku"]}', entity_types=None) == [
        ("person", "Ann"), ("location", "Baku"),
    ]


def test_bio_tags_custom_map():
    assert bio_tags("[1, 0]", id2tag={0: "O", 1: "B-person"}) == ["B-person", "O"]
