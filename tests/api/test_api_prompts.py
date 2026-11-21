import pytest

from src.api.prompts import (
    Example,
    TaskTemplate,
    build_prompt,
    get_template,
    parse_free_text,
    parse_output,
    register_template,
    registered_templates,
    split_confidence,
)


def test_classification_prompt_structure(cls_context):
    shots = [Example("a", "where is my card", "card_arrival"), Example("b", "rate today", "exchange_rate")]
    p = build_prompt("classification", shots, "close my account", cls_context)
    assert "Classify banking messages by intent." in p.system
    for lab in cls_context["labels"]:
        assert f"- {lab}" in p.system
    assert p.user.index("Input: where is my card\nLabel: card_arrival") < p.user.index("Input: rate today")
    assert p.user.endswith("Input: close my account\nLabel:")
    assert p.user.count("Label:") == 3


def test_zero_shot_prompt_has_no_examples(cls_context):
    p = build_prompt("classification", [], "close my account", cls_context)
    assert "Examples:" not in p.user
    assert p.user == "Input: close my account\nLabel:"


def test_summarisation_prompt():
    shots = [Example("s1", "BILL TEXT ONE", "Summary one."), Example("s2", "BILL TEXT TWO", "Summary two.")]
    p = build_prompt("summarisation", shots, "QUERY BILL", {"task_description": "Summarise bills."})
    assert "Summarise bills." in p.system
    assert "Document:\nBILL TEXT ONE\n\nSummary:\nSummary one." in p.user
    assert p.user.endswith("Document:\nQUERY BILL\n\nSummary:")
    assert p.user.count("---") == 4  # header | shot | shot | instruction | query: 5 parts, 4 separators


def test_missing_context_raises():
    with pytest.raises(ValueError, match="labels"):
        build_prompt("classification", [], "q", {"task_description": "x"})


def test_register_new_task_template():
    name = "ner_test_only"
    tpl = TaskTemplate(name=name, system_instruction="Tag entities of types {types}.",
                       parse=parse_free_text, max_output_tokens=128,
                       example_format="Sentence: {input}\nEntities: {output}",
                       query_format="Sentence: {input}\nEntities:", required_context=("types",))
    register_template(tpl)
    assert name in registered_templates()
    p = build_prompt(name, [Example("e", "Ada lives in Paris", "PER: Ada; LOC: Paris")], "Bob met Ann",
                     {"types": ["PER", "LOC"]})
    assert "- PER" in p.system
    assert p.user.endswith("Sentence: Bob met Ann\nEntities:")
    with pytest.raises(ValueError):
        register_template(tpl)
    assert get_template(name) is tpl


@pytest.mark.parametrize("text,expected,ok", [
    ("card_arrival", "card_arrival", True),
    ("  Card Arrival.\n", "card_arrival", True),
    ("Label: exchange_rate", "exchange_rate", True),
    ("`terminate_account`", "terminate_account", True),
    ("card arrived late", "card arrived late", False),
    ("", None, False),
])
def test_label_parsing(text, expected, ok, cls_context):
    parsed, parse_ok, conf = parse_output("classification", text, cls_context)
    assert parsed == expected and parse_ok is ok and conf is None


def test_verbalised_confidence(cls_context):
    p = build_prompt("classification", [], "q", cls_context, verbalised_confidence=True)
    assert "Confidence:" in p.system
    parsed, ok, conf = parse_output("classification", "exchange_rate\nConfidence: 0.85", cls_context,
                                    verbalised_confidence=True)
    assert (parsed, ok, conf) == ("exchange_rate", True, 0.85)
    assert split_confidence("x\nConfidence: 85%") == ("x", 0.85)
    assert split_confidence("x") == ("x", None)
