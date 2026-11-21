"""Per-item prediction record written by run.py (one JSON object per line).

The field list matches the one agreed with the evaluation harness
(`src/eval/schema.py`, built in a parallel session). This module carries its
own JSON Schema so the API arm can validate its output without importing
src/eval. tests/api also validates against src.eval.schema when that module
exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import jsonschema

TECHNIQUES = ("icl_retrieved", "icl_random", "zero_shot")

RECORD_FIELDS = (
    "run_id", "system", "arm", "technique", "task", "n_budget", "seed", "item_id",
    "prediction", "parsed", "gold", "parse_ok", "confidence", "latency_ms",
    "input_tokens", "output_tokens", "extra",
)

RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": list(RECORD_FIELDS),
    "properties": {
        "run_id": {"type": "string", "minLength": 1},
        "system": {"type": "string", "minLength": 1},
        "arm": {"const": "api"},
        "technique": {"enum": list(TECHNIQUES)},
        "task": {"type": "string", "minLength": 1},
        # Shared schema: zero_shot uses no pool, so its n_budget is 0.
        "n_budget": {"type": "integer", "minimum": 0},
        "seed": {"type": ["integer", "null"]},
        "item_id": {"type": "string", "minLength": 1},
        "prediction": {"type": "string"},
        "parsed": {},
        "gold": {},
        "parse_ok": {"type": "boolean"},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        # null for batch-API results, where per-request latency is not observable.
        "latency_ms": {"type": ["number", "null"], "minimum": 0},
        "input_tokens": {"type": "integer", "minimum": 0},
        "output_tokens": {"type": "integer", "minimum": 0},
        "extra": {"type": "object"},
    },
}

_VALIDATOR = jsonschema.Draft202012Validator(RECORD_SCHEMA)


def validate_record(record: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the record does not match."""
    _VALIDATOR.validate(record)
    if record["technique"] == "icl_random" and record["seed"] is None:
        raise jsonschema.ValidationError("icl_random records need a seed")
    if record["technique"] != "icl_random" and record["seed"] is not None:
        raise jsonschema.ValidationError("seed must be null unless technique is icl_random")
    if (record["technique"] == "zero_shot") != (record["n_budget"] == 0):
        raise jsonschema.ValidationError("n_budget must be 0 exactly when technique is zero_shot")
    if record["parse_ok"] != (record["parsed"] is not None):
        raise jsonschema.ValidationError("parse_ok must equal (parsed is not None)")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                # A torn last line from an interrupted write; that item is redone.
                continue
    return out


def done_item_ids(path: str | Path) -> set[str]:
    return {r["item_id"] for r in read_records(path) if "item_id" in r}


def append_records(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Validate and append records, flushing each line. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            validate_record(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
    return n
