"""Type I error of the FINAL contrast code at the real design (review 06d, F8), and the BillSum minimal
detectable gaps read from the committed simulation file. CPU only, synthetic data only.

    python analysis/type1_check.py --sims 4000 --out analysis/reference_outputs/type1_final_code.json

The data generator is power_sim2.sim_b77 (its accuracy levels, correlations and pool-effect sizes are
ASSUMPTIONS); the test is analysis/inference.crossed_contrast, the function the paper will use. True
gap 0, conditional estimand (centred on the 9,003 usable rows), BANKING77 n = 3,080.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import norm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import inference as inf  # noqa: E402
from power_sim2 import sim_b77  # noqa: E402

N_ITEMS, SEED = 3080, 20261128
ALPHAS = (0.05, 0.05 / 6)


def type1(sims: int, R: int = 8, seed: int = SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for N, reg in itertools.product((200, 1000, 3000), ("opt", "pess")):
        p = np.array([inf.crossed_contrast(sim_b77(rng, N_ITEMS, R, N, reg, 0.0, "cond")).p
                      for _ in range(sims)])
        row = {"N": N, "R": R, "n_items": N_ITEMS, "regime": reg, "sims": sims}
        for a in ALPHAS:
            rate = float((p < a).mean())
            row[f"rate_at_{a:.4f}"] = rate
            row[f"mc_se_at_{a:.4f}"] = float(np.sqrt(rate * (1 - rate) / sims))
        rows.append(row)
        print(row, flush=True)
    return rows


def billsum_mdg(path: str, power: float = 0.80):
    """Gap (ROUGE-L fraction) at `power`, interpolated linearly on the probit scale between the two
    bracketing simulated gaps; cells with n = 1,000, R = 5, N = 1,000."""
    out = {}
    for row in json.load(open(path)):
        if (row["n"], row["R"], row["N"]) != (1000, 5, 1000):
            continue
        pts = sorted((float(g), p) for g, p in row["pw"].items())
        for (g0, p0), (g1, p1) in zip(pts, pts[1:]):
            if p0 < power <= p1:
                z0, z1, z = (norm.ppf(min(max(v, 1e-6), 1 - 1e-6)) for v in (p0, p1, power))
                out[str(row["sigma_rep"])] = float(g0 + (g1 - g0) * (z - z0) / (z1 - z0))
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"seed": SEED, "type1_banking77": type1(a.sims),
           "billsum_mdg_80pct": billsum_mdg(os.path.join(_HERE, "reference_outputs", "sim2_billsum.json"))}
    print(res["billsum_mdg_80pct"])
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
