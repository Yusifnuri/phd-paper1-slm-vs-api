"""Task-agnostic prompt builder: system instruction + k examples + query.

A task is supported by registering a `TaskTemplate` under a name. The builder
itself never looks at the task: it formats each example with
`example_format`, the query with `query_format`, and joins them. Task-specific
values (for example the label set) are passed as `context` and substituted into
the system instruction with `str.format`.

Adding NER, sentiment or SQL means writing one `TaskTemplate` (instruction,
formats, a parser, an output cap) and calling `register_template`.

The prompt is one system string and one user message for every provider, so
the same prompt text is sent to all three APIs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# parse(text, context) -> (parsed_value, parse_ok)
Parser = Callable[[str, dict[str, Any]], tuple[Any, bool]]


@dataclass(frozen=True)
class Example:
    """One labelled example. `output` is the gold answer as text."""

    id: str
    input: str
    output: str


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str

    @property
    def n_chars(self) -> int:
        return len(self.system) + len(self.user)


@dataclass(frozen=True)
class TaskTemplate:
    name: str
    system_instruction: str
    parse: Parser
    max_output_tokens: int
    example_format: str = "Input: {input}\nOutput: {output}"
    query_format: str = "Input: {input}\nOutput:"
    examples_header: str = "Examples:"
    query_header: str = "Now answer for this input."
    separator: str = "\n\n"
    required_context: tuple[str, ...] = field(default_factory=tuple)


CONFIDENCE_INSTRUCTION = (
    "After your answer, write one more line of the form 'Confidence: <p>', where <p> "
    "is your probability, between 0 and 1, that the answer is correct."
)
_CONF_RE = re.compile(r"^\s*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%?\s*$", re.IGNORECASE | re.MULTILINE)

_REGISTRY: dict[str, TaskTemplate] = {}


def register_template(template: TaskTemplate, *, overwrite: bool = False) -> TaskTemplate:
    if template.name in _REGISTRY and not overwrite:
        raise ValueError(f"template {template.name!r} is already registered")
    _REGISTRY[template.name] = template
    return template


def get_template(name: str) -> TaskTemplate:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown template {name!r}; registered: {sorted(_REGISTRY)}") from None


def registered_templates() -> list[str]:
    return sorted(_REGISTRY)


def build_prompt(
    template: TaskTemplate | str,
    examples: Sequence[Example],
    query: str,
    context: dict[str, Any] | None = None,
    *,
    verbalised_confidence: bool = False,
) -> Prompt:
    """Build the prompt. Examples appear in the order given."""
    tpl = get_template(template) if isinstance(template, str) else template
    ctx = dict(context or {})
    missing = [k for k in tpl.required_context if k not in ctx]
    if missing:
        raise ValueError(f"template {tpl.name!r} needs context keys {missing}")
    fmt_ctx = {k: _format_context_value(v) for k, v in ctx.items()}
    system = tpl.system_instruction.format(**fmt_ctx)
    if verbalised_confidence:
        system = f"{system}\n\n{CONFIDENCE_INSTRUCTION}"

    parts: list[str] = []
    if examples:
        shots = tpl.separator.join(
            tpl.example_format.format(input=ex.input, output=ex.output) for ex in examples
        )
        parts.append(f"{tpl.examples_header}{tpl.separator}{shots}")
        parts.append(tpl.query_header)
    parts.append(tpl.query_format.format(input=query))
    return Prompt(system=system, user=tpl.separator.join(parts))


def split_confidence(text: str) -> tuple[str, float | None]:
    """Remove a trailing 'Confidence: p' line; return (answer_text, p or None)."""
    matches = list(_CONF_RE.finditer(text))
    if not matches:
        return text, None
    m = matches[-1]
    raw = float(m.group(1))
    if m.group(0).rstrip().endswith("%") or raw > 1.0:
        raw = raw / 100.0
    value = raw if 0.0 <= raw <= 1.0 else None
    answer = (text[: m.start()] + text[m.end():]).strip()
    return answer, value


def parse_output(
    template: TaskTemplate | str,
    text: str,
    context: dict[str, Any] | None = None,
    *,
    verbalised_confidence: bool = False,
) -> tuple[Any, bool, float | None]:
    """Return (parsed, parse_ok, confidence)."""
    tpl = get_template(template) if isinstance(template, str) else template
    confidence = None
    if verbalised_confidence:
        text, confidence = split_confidence(text)
    parsed, ok = tpl.parse(text, dict(context or {}))
    return parsed, ok, confidence


def _format_context_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {v}" for v in value)
    return value


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _norm_label(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().strip("`'\".,;:*").strip().lower())


def parse_label(text: str, context: dict[str, Any]) -> tuple[Any, bool]:
    """First non-empty line, normalised, must match one label exactly (case- and
    separator-insensitive). No fuzzy matching: a near miss is a format failure.
    On failure `parsed` is None (shared schema: parse_ok == (parsed is not None));
    the raw text is kept in the record's `prediction` field."""
    labels = context.get("labels") or []
    lookup = {_norm_label(lab): lab for lab in labels}
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None, False
    first = lines[0].strip()
    first = re.sub(r"^(output|label|answer|intent)\s*:\s*", "", first, flags=re.IGNORECASE)
    key = _norm_label(first)
    if key in lookup:
        return lookup[key], True
    return None, False


def parse_free_text(text: str, context: dict[str, Any]) -> tuple[Any, bool]:
    cleaned = re.sub(r"^\s*(output|summary)\s*:\s*", "", text.strip(), flags=re.IGNORECASE).strip()
    return (cleaned, True) if cleaned else (None, False)


# ---------------------------------------------------------------------------
# Shipped templates
# ---------------------------------------------------------------------------

CLASSIFICATION = register_template(
    TaskTemplate(
        name="classification",
        system_instruction=(
            "{task_description}\n\n"
            "Answer with exactly one label from this list, copied verbatim, and nothing else:\n"
            "{labels}"
        ),
        parse=parse_label,
        max_output_tokens=32,
        example_format="Input: {input}\nLabel: {output}",
        query_format="Input: {input}\nLabel:",
        required_context=("task_description", "labels"),
    )
)

SUMMARISATION = register_template(
    TaskTemplate(
        name="summarisation",
        system_instruction=(
            "{task_description}\n\n"
            "Write only the summary, as plain prose, with no heading or preamble."
        ),
        parse=parse_free_text,
        # BillSum train summaries reach 4,995 chars (~1.25k tokens at 4 chars/token,
        # ~1.6k with Haiku's ~1.3x tokenizer), so 2048 avoids truncating the longest.
        max_output_tokens=2048,
        example_format="Document:\n{input}\n\nSummary:\n{output}",
        query_format="Document:\n{input}\n\nSummary:",
        separator="\n\n---\n\n",
        required_context=("task_description",),
    )
)

# Default task contexts for the tasks already chosen in docs/PLAN.md.
BANKING77_DESCRIPTION = (
    "You classify online-banking customer messages by intent."
)
BILLSUM_DESCRIPTION = (
    "You summarise US legislative bills. Summarise the bill in the style of the example summaries."
)
