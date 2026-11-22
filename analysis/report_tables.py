"""Render the contrast table and the cost / break-even table as Markdown and CSV (no plotting).

    python analysis/report_tables.py contrasts rows.json --out-dir tables/   # rows from build_contrasts.analyze
    python analysis/report_tables.py cost cost_inputs.json --out-dir tables/

Contrast table: d = FT - ICL in points (fraction x 100), positive = FT better; conditional on the fixed
training set. Labels are those of analysis/inference.py; the word "equivalent" is never printed.
Cost table: computed from supplied inputs only (nothing is defaulted or invented); each input row is
    {"cell", "c_train", "c_api", "c_ft", "a_ft", "a_api", optional "u" [..], "volumes" [..],
     optional "var_denominator", "var_c_train", "cov" (Fieller interval, marginal form)}
with c_api and c_ft in USD per query (c_ft at saturation; utilisation u divides it), c_train in USD.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference as inf  # noqa: E402

DEFAULT_U = (1.0, 0.5, 0.2)
DEFAULT_VOLUMES = (1e3, 1e4, 1e5, 1e6, 1e7)
CONTRAST_NOTE = ("d = fine-tuned minus in-context, in points (x100); positive = fine-tuned better. Conditional "
                 "on the fixed training set. Holm over the family; p unadjusted shown beside. Decision labels: "
                 "'FT better' / 'ICL better' only if Holm p < 0.05, otherwise 'inconclusive'. "
                 "'Practically equal' is descriptive only (95% CI inside the margin; never for BillSum).")


def _pts(x: Optional[float]) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:+.2f}"


def _ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"


def _p(x: float) -> str:
    return "<0.0001" if x < 1e-4 else f"{x:.4f}"


def _flag(v: Optional[bool]) -> str:
    return "n/a" if v is None else ("yes" if v else "no")


def contrast_table(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Header + string rows for the six-contrast table."""
    header = ["Task", "N", "Items", "R", "Delta (points)", "95% CI", "99.2% CI", "p (unadj.)", "p (Holm)",
              "Decision", "Practically equal (descr.)", "Underpowered a priori", "Note"]
    body = []
    for r in rows:
        if r["label"].lower().startswith("equiv"):
            raise ValueError("'equivalent' is not an allowed label")
        body.append([r["task"], str(r["n_budget"]), str(r["n_items"]), str(r["R"]), _pts(r["delta"]),
                     _ci(r["ci95_low"], r["ci95_high"]), _ci(r["ci99_2_low"], r["ci99_2_high"]),
                     _p(r["p"]), _p(r["p_holm"]), r["label"], _flag(r.get("practically_equal_descriptive")),
                     _flag(r.get("underpowered_a_priori")), r.get("note", "")])
    return {"header": header, "rows": body, "note": CONTRAST_NOTE}


def cost_table(inputs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Break-even and cost-per-correct rows, one per (cell, u). Values 'never' when the denominator <= 0."""
    vols = None
    out_rows: List[List[str]] = []
    for row in inputs:
        for key in ("cell", "c_train", "c_api", "c_ft", "a_ft", "a_api"):
            if key not in row:
                raise ValueError(f"cost input row missing {key!r}: {row}")
        v = tuple(row.get("volumes", DEFAULT_VOLUMES))
        vols = vols or v
        if v != vols:
            raise ValueError("all cost rows must use the same volumes")
        for u in row.get("u", DEFAULT_U):
            q_m = inf.break_even_marginal(row["c_train"], row["c_api"], row["c_ft"], u)
            q_c = inf.break_even_per_correct(row["c_train"], row["a_ft"], row["a_api"], row["c_api"], row["c_ft"], u)
            cells = [row["cell"], f"{u:g}", _q(q_m), _q(q_c)]
            if "var_denominator" in row:
                den = row["c_api"] - row["c_ft"] / u
                be = inf.break_even_with_ci(row["c_train"], row.get("var_c_train", 0.0), den,
                                            row["var_denominator"], row.get("cov", 0.0))
                cells.append(be.status if be.status != "finite" else f"[{be.ci_low:,.0f}, {be.ci_high:,.0f}]")
            else:
                cells.append("not computed")
            for q in vols:
                cells.append(_usd(inf.cost_per_correct(row["c_train"], row["c_ft"] / u, row["a_ft"], q)))
                cells.append(_usd(inf.cost_per_correct(0.0, row["c_api"], row["a_api"], q)))
            out_rows.append(cells)
    if vols is None:
        raise ValueError("no cost rows")
    header = ["Cell", "u", "Q* marginal (queries)", "Q* per correct (queries)", "Q* marginal Fieller 95%"]
    for q in vols:
        header += [f"FT cost/correct @{q:,.0f}", f"API cost/correct @{q:,.0f}"]
    return {"header": header, "rows": out_rows,
            "note": ("USD; Q* 'never' when c_API <= c_FT/u (no break-even); cost per correct = (C_train + Q c)/(Q a). "
                     "Inputs are as supplied; none are defaulted.")}


def _q(x: float) -> str:
    return "never" if math.isinf(x) else f"{x:,.0f}"


def _usd(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x:.6g}"


def to_markdown(t: Dict[str, Any]) -> str:
    lines = ["| " + " | ".join(t["header"]) + " |", "|" + "|".join("---" for _ in t["header"]) + "|"]
    lines += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in t["rows"]]
    return "\n".join(lines) + f"\n\n{t['note']}\n"


def to_csv(t: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(t["header"])
    w.writerows(t["rows"])
    return buf.getvalue()


def write_tables(t: Dict[str, Any], out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{stem}.md"), os.path.join(out_dir, f"{stem}.csv")]
    with open(paths[0], "w", encoding="utf-8") as fh:
        fh.write(to_markdown(t))
    with open(paths[1], "w", encoding="utf-8", newline="") as fh:
        fh.write(to_csv(t))
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("contrasts", "cost"))
    ap.add_argument("json_file")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    data = json.load(open(a.json_file))
    t = contrast_table(data) if a.kind == "contrasts" else cost_table(data)
    for p in write_tables(t, a.out_dir, a.kind):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
