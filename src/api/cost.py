"""Cost estimator for API runs. It must run before any spend.

Token counting:
- This module calls no tokenizer and no network. Prompt tokens are estimated as
  ceil(chars / chars_per_token * token_multiplier).
- chars_per_token = 4.0 for every provider (configs/api_models.yaml). This is
  the usual rule of thumb for English, and it matches the BillSum figure in
  docs/research (10,268 chars ~ 2.6k tokens).
- token_multiplier is 1.3 for Haiku 5.5, whose new tokenizer gives about 30%
  more tokens (verified finding), and 1.0 for the others.
- The 50-item pilot should replace this with provider-billed counts.

Output tokens are estimated two ways:
- **ceiling**: max_output_tokens for every item. This is a hard upper bound on
  billed output for Anthropic and OpenAI. For Gemini it is near-certain, since
  thinking runs at "minimal". The budget gate in run.py uses the ceiling.
- **expected**: the mean length of the gold outputs, through the same chars
  fallback. This is reported for planning only.

The planned grid uses mean text lengths measured from the Hugging Face
dataset-statistics API (datasets-server.huggingface.co/statistics, read on
2026-11-21, full train splits):
- BANKING77 `text`: mean 59.47 chars (10,003 rows). There are 77 label names,
  mean 20.4 chars.
- BillSum `text`: mean 10,272.54 chars; `summary`: mean 1,185.64 chars
  (18,949 rows).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Sequence

from src.api.config import DEFAULT_CONFIG, ModelSpec, default_grid_providers, load_specs
from src.api.prompts import (
    BANKING77_DESCRIPTION,
    BILLSUM_DESCRIPTION,
    Example,
    Prompt,
    build_prompt,
    get_template,
)


def estimate_tokens(n_chars: int, spec: ModelSpec) -> int:
    if n_chars <= 0:
        return 0
    return math.ceil(n_chars / spec.chars_per_token * spec.token_multiplier)


def price_usd(spec: ModelSpec, input_tokens: int, output_tokens: int, batch: bool) -> float:
    usd = (input_tokens * spec.input_usd_per_mtok + output_tokens * spec.output_usd_per_mtok) / 1e6
    return usd * (1.0 - spec.batch_discount) if batch else usd


@dataclass
class Estimate:
    provider: str
    model_id: str
    task: str
    technique: str
    n_budget: int | None
    k: int
    n_items: int
    input_tokens: int
    output_tokens_ceiling: int
    output_tokens_expected: int
    usd_standard_ceiling: float
    usd_batch_ceiling: float
    usd_standard_expected: float
    usd_batch_expected: float

    def cap_usd(self, batch: bool) -> float:
        """The figure compared with --budget-usd."""
        return self.usd_batch_ceiling if batch else self.usd_standard_ceiling

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_prompts(
    prompts: Sequence[Prompt],
    spec: ModelSpec,
    *,
    max_output_tokens: int,
    expected_output_chars: float | None = None,
    task: str = "",
    technique: str = "",
    n_budget: int | None = None,
    k: int = 0,
) -> Estimate:
    in_tok = sum(estimate_tokens(p.n_chars, spec) for p in prompts)
    n = len(prompts)
    out_ceiling = n * max_output_tokens
    if expected_output_chars is None:
        out_expected = out_ceiling
    else:
        out_expected = n * min(max_output_tokens, estimate_tokens(int(round(expected_output_chars)), spec))
    return Estimate(
        provider=spec.provider, model_id=spec.model_id, task=task, technique=technique,
        n_budget=n_budget, k=k, n_items=n, input_tokens=in_tok,
        output_tokens_ceiling=out_ceiling, output_tokens_expected=out_expected,
        usd_standard_ceiling=price_usd(spec, in_tok, out_ceiling, False),
        usd_batch_ceiling=price_usd(spec, in_tok, out_ceiling, True),
        usd_standard_expected=price_usd(spec, in_tok, out_expected, False),
        usd_batch_expected=price_usd(spec, in_tok, out_expected, True),
    )


# ---------------------------------------------------------------------------
# Planned grid (docs/PLAN.md Step 4/5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskShape:
    """Mean sizes of one task, used to synthesise representative prompts."""

    task: str
    template: str
    k: int
    mean_input_chars: float
    mean_output_chars: float
    context: dict


def _banking77_labels() -> list[str]:
    # 77 placeholder names with the measured mean length (20.4 chars), so the label
    # list in the system prompt has the right size without shipping the dataset here.
    return [("label_%02d_" % i).ljust(20, "x") for i in range(77)]


PLANNED_TASKS = (
    TaskShape("banking77", "classification", k=32, mean_input_chars=59.47, mean_output_chars=20.4,
              context={"task_description": BANKING77_DESCRIPTION, "labels": _banking77_labels()}),
    TaskShape("billsum", "summarisation", k=5, mean_input_chars=10272.54, mean_output_chars=1185.64,
              context={"task_description": BILLSUM_DESCRIPTION}),
)
PLANNED_N = (50, 200, 1000, 5000)
PLANNED_ITEMS = 1000


def _filler(n_chars: float) -> str:
    n = int(round(n_chars))
    return ("lorem ipsum " * (n // 12 + 1))[:n]


def representative_prompt(shape: TaskShape) -> Prompt:
    ex = Example(id="x", input=_filler(shape.mean_input_chars), output=_filler(shape.mean_output_chars))
    return build_prompt(shape.template, [ex] * shape.k, _filler(shape.mean_input_chars), shape.context)


def planned_grid(providers: Sequence[str] | None = None, config_path=DEFAULT_CONFIG,
                 n_items: int = PLANNED_ITEMS) -> list[Estimate]:
    specs = load_specs(config_path)
    providers = list(providers or default_grid_providers(config_path))
    rows = []
    for shape in PLANNED_TASKS:
        tpl = get_template(shape.template)
        prompt = representative_prompt(shape)
        for name in providers:
            for n in PLANNED_N:
                rows.append(estimate_prompts(
                    [prompt] * n_items, specs[name], max_output_tokens=tpl.max_output_tokens,
                    expected_output_chars=shape.mean_output_chars, task=shape.task,
                    technique="icl_retrieved", n_budget=n, k=shape.k))
    return rows


def format_grid(rows: Sequence[Estimate]) -> str:
    hdr = (f"{'task':<10} {'provider':<10} {'N':>5} {'k':>3} {'items':>6} {'in tok/item':>11} "
           f"{'std exp':>9} {'batch exp':>9} {'std ceil':>9} {'batch ceil':>10}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r.task:<10} {r.provider:<10} {r.n_budget:>5} {r.k:>3} {r.n_items:>6} "
            f"{r.input_tokens // max(r.n_items, 1):>11,} "
            f"{r.usd_standard_expected:>9.2f} {r.usd_batch_expected:>9.2f} "
            f"{r.usd_standard_ceiling:>9.2f} {r.usd_batch_ceiling:>10.2f}")
    lines.append("-" * len(hdr))
    by_task: dict[str, list[Estimate]] = {}
    for r in rows:
        by_task.setdefault(r.task, []).append(r)
    for task, rs in list(by_task.items()) + [("TOTAL", list(rows))]:
        lines.append(
            f"{task:<10} {'all':<10} {'':>5} {'':>3} {sum(r.n_items for r in rs):>6} {'':>11} "
            f"{sum(r.usd_standard_expected for r in rs):>9.2f} {sum(r.usd_batch_expected for r in rs):>9.2f} "
            f"{sum(r.usd_standard_ceiling for r in rs):>9.2f} {sum(r.usd_batch_ceiling for r in rs):>10.2f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Dry-run cost estimate for the planned API grid (no API calls).")
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--items", type=int, default=PLANNED_ITEMS)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args(argv)
    rows = planned_grid(args.providers, args.config, args.items)
    print("Planned grid: icl_retrieved, BANKING77 (k=32) and BillSum (k=5), "
          f"{args.items} test items per cell. USD.")
    print("exp = output at mean gold length; ceil = output at max_output_tokens (the budget gate uses ceil).")
    print("Tokens: chars/4.0 (x1.3 for Haiku 5.5). Controls (icl_random per seed, zero_shot) are extra.\n")
    print(format_grid(rows))


if __name__ == "__main__":
    main()
