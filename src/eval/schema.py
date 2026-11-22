"""
Shared per-instance prediction schema, used by both the fine-tuned and API arms.

One JSONL record per test item per system run. Fields:

    run_id         str         unique id of one system run (one model, technique, N, seed)
    system         str         model or API id, e.g. "Qwen/Qwen3-4B-Instruct-2507"
    arm            str         "finetuned" | "api"
    technique      str         e.g. "lora", "qlora", "dora", "rslora", "full",
                               "icl_retrieved", "icl_random", "zero_shot"
    task           str         task name, e.g. "banking77"
    n_budget       int         labelled-example budget N; 0 for zero-shot
    seed           int | null  training or sampling seed; null if not applicable
    item_id        str         test item id, stable across systems
    prediction     str         raw output text
    parsed         any | null  normalised prediction, null if unparseable
    gold           any         gold answer (JSON-serialisable)
    parse_ok       bool        whether the raw output could be parsed
    confidence     float|null  confidence in [0, 1], null if unknown
    latency_ms     float|null  wall-clock latency for this item, null if unknown
    input_tokens   int | null  null if unknown
    output_tokens  int | null  null if unknown
    extra          dict        anything else (provider usage, retrieved ids, ...)

Clarifications to the field list agreed between sessions (validated below):

- `latency_ms` may be null, like the token counts (batched runs often have no
  per-item latency).
- `parse_ok` must equal `parsed is not None`. An empty but valid parse (for
  example no entities found) is `[]`, not null.
- `technique == "zero_shot"` requires `n_budget == 0`.
- Unknown top-level fields are rejected, to catch typos; put extras in `extra`.
- Within one `run_id`, the run-level fields (system, arm, technique, task,
  n_budget, seed) must be identical, and each `item_id` may appear only once.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

ARMS = ("finetuned", "api")
RUN_LEVEL_FIELDS = ("system", "arm", "technique", "task", "n_budget", "seed")

PathLike = Union[str, Path]


class SchemaError(ValueError):
    """A prediction record or file violates the shared schema."""


class AlignmentError(ValueError):
    """Runs to be compared do not cover the same test items or golds."""


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction for one test item by one system run."""

    run_id: str
    system: str
    arm: str
    technique: str
    task: str
    n_budget: int
    seed: Optional[int]
    item_id: str
    prediction: str
    parsed: Any
    gold: Any
    parse_ok: bool
    confidence: Optional[float]
    latency_ms: Optional[float]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_record(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRecord":
        """Build a record from a dict, rejecting missing or unknown fields."""
        if not isinstance(d, dict):
            raise SchemaError(f"record must be a JSON object, got {type(d).__name__}")
        names = [f for f in cls.__dataclass_fields__]
        unknown = sorted(set(d) - set(names))
        if unknown:
            raise SchemaError(f"unknown fields {unknown}; put extras under 'extra'")
        missing = [n for n in names if n not in d and n != "extra"]
        if missing:
            raise SchemaError(f"missing fields {missing}")
        return cls(**{n: d.get(n, {}) if n == "extra" else d[n] for n in names})

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a plain JSON-serialisable dict."""
        return asdict(self)


def validate_record(r: PredictionRecord) -> None:
    """Raise SchemaError if a record violates the field types or invariants."""
    for name in ("run_id", "system", "technique", "task", "item_id"):
        v = getattr(r, name)
        if not isinstance(v, str) or not v:
            raise SchemaError(f"{name} must be a non-empty string, got {v!r}")
    if not isinstance(r.prediction, str):
        raise SchemaError(f"prediction must be a string, got {type(r.prediction).__name__}")
    if r.arm not in ARMS:
        raise SchemaError(f"arm must be one of {ARMS}, got {r.arm!r}")
    if not _is_int(r.n_budget) or r.n_budget < 0:
        raise SchemaError(f"n_budget must be a non-negative int, got {r.n_budget!r}")
    if r.technique == "zero_shot" and r.n_budget != 0:
        raise SchemaError(f"zero_shot requires n_budget 0, got {r.n_budget}")
    if r.seed is not None and not _is_int(r.seed):
        raise SchemaError(f"seed must be an int or null, got {r.seed!r}")
    if not isinstance(r.parse_ok, bool):
        raise SchemaError(f"parse_ok must be a bool, got {r.parse_ok!r}")
    if r.parse_ok != (r.parsed is not None):
        raise SchemaError(
            f"parse_ok={r.parse_ok} but parsed is {'null' if r.parsed is None else 'set'} "
            f"(item {r.item_id!r}, run {r.run_id!r})"
        )
    if r.confidence is not None:
        if not _is_number(r.confidence) or not 0.0 <= float(r.confidence) <= 1.0:
            raise SchemaError(f"confidence must be in [0, 1] or null, got {r.confidence!r}")
    if r.latency_ms is not None and (not _is_number(r.latency_ms) or r.latency_ms < 0):
        raise SchemaError(f"latency_ms must be a non-negative number or null, got {r.latency_ms!r}")
    for name in ("input_tokens", "output_tokens"):
        v = getattr(r, name)
        if v is not None and (not _is_int(v) or v < 0):
            raise SchemaError(f"{name} must be a non-negative int or null, got {v!r}")
    if not isinstance(r.extra, dict):
        raise SchemaError(f"extra must be a dict, got {type(r.extra).__name__}")
    try:
        json.dumps(r.gold)
        json.dumps(r.parsed)
        json.dumps(r.extra)
    except (TypeError, ValueError) as e:
        raise SchemaError(f"gold, parsed and extra must be JSON-serialisable: {e}") from e


def check_run_consistency(records: Iterable[PredictionRecord]) -> None:
    """Raise SchemaError if a run_id has conflicting run-level fields or duplicate items."""
    first: Dict[str, PredictionRecord] = {}
    seen = set()
    for r in records:
        key = (r.run_id, r.item_id)
        if key in seen:
            raise SchemaError(f"duplicate item {r.item_id!r} in run {r.run_id!r}")
        seen.add(key)
        ref = first.setdefault(r.run_id, r)
        for name in RUN_LEVEL_FIELDS:
            if getattr(r, name) != getattr(ref, name):
                raise SchemaError(
                    f"run {r.run_id!r} mixes {name}={getattr(ref, name)!r} and {getattr(r, name)!r}"
                )


def load_predictions(paths: Union[PathLike, Sequence[PathLike]]) -> List[PredictionRecord]:
    """Read and validate one or more JSONL prediction files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: List[PredictionRecord] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    records.append(PredictionRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, SchemaError, TypeError) as e:
                    raise SchemaError(f"{path}:{lineno}: {e}") from e
    check_run_consistency(records)
    return records


def write_predictions(path: PathLike, records: Iterable[PredictionRecord]) -> None:
    """Write records to a JSONL file, one per line."""
    records = list(records)
    check_run_consistency(records)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


@dataclass
class AlignedRuns:
    """Several runs restricted to, and ordered by, one common list of item ids."""

    item_ids: List[str]
    runs: Dict[str, List[PredictionRecord]]

    def field(self, run_id: str, name: str) -> List[Any]:
        """Return one field for every item of a run, in item_ids order."""
        return [getattr(r, name) for r in self.runs[run_id]]

    @property
    def gold(self) -> List[Any]:
        """Return the gold answers in item_ids order (identical across runs)."""
        return self.field(next(iter(self.runs)), "gold")


def align_runs(
    records: Iterable[PredictionRecord], run_ids: Optional[Sequence[str]] = None
) -> AlignedRuns:
    """Align runs on identical item_id sets, failing loudly on any mismatch."""
    records = list(records)
    check_run_consistency(records)
    by_run: Dict[str, Dict[str, PredictionRecord]] = {}
    for r in records:
        by_run.setdefault(r.run_id, {})[r.item_id] = r
    if run_ids is None:
        run_ids = sorted(by_run)
    missing_runs = [rid for rid in run_ids if rid not in by_run]
    if missing_runs:
        raise AlignmentError(f"no records for runs {missing_runs}")
    if not run_ids:
        raise AlignmentError("no runs to align")

    tasks = {next(iter(by_run[rid].values())).task for rid in run_ids}
    if len(tasks) > 1:
        raise AlignmentError(f"runs span several tasks: {sorted(tasks)}")

    ref_id = run_ids[0]
    ref_items = set(by_run[ref_id])
    problems = []
    for rid in run_ids[1:]:
        items = set(by_run[rid])
        if items != ref_items:
            only_ref = sorted(ref_items - items)
            only_run = sorted(items - ref_items)
            problems.append(
                f"run {rid!r} vs {ref_id!r}: {len(only_run)} extra items "
                f"(e.g. {only_run[:5]}), {len(only_ref)} missing items (e.g. {only_ref[:5]})"
            )
    if problems:
        raise AlignmentError("item sets differ:\n  " + "\n  ".join(problems))

    item_ids = sorted(ref_items)
    gold_mismatch = [
        (i, rid)
        for i in item_ids
        for rid in run_ids[1:]
        if json.dumps(by_run[rid][i].gold, sort_keys=True)
        != json.dumps(by_run[ref_id][i].gold, sort_keys=True)
    ]
    if gold_mismatch:
        raise AlignmentError(
            f"{len(gold_mismatch)} (item, run) pairs disagree with {ref_id!r} on gold, "
            f"e.g. {gold_mismatch[:5]}"
        )
    return AlignedRuns(item_ids=item_ids, runs={rid: [by_run[rid][i] for i in item_ids] for rid in run_ids})
