"""Power / type-I simulation for the confirmatory contrast (FT vs best ICL), one task x one N.
Latent probit model, calibrated to pilot discordance (see 04-statistics.md). Assumptions marked A#.
Run: python3 -I power_sim.py  (numpy + scipy only)"""
import numpy as np, json, sys, itertools
from scipy.stats import norm, multivariate_normal, t as tdist
from scipy.optimize import brentq

def disc(p1, p2, rho):  # P(c1 != c2) for probit correctness with latent corr rho
    a1, a2 = norm.ppf(p1), norm.ppf(p2)
    both = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([a1, a2])
    return p1 + p2 - 2 * both

def calibrate(p, D_b, D_w):
    """rho_b: latent corr between arms (same item, same replicate) matching between-arm discordance D_b.
       rho_w: corr between two replicates of the same arm matching within-arm discordance D_w."""
    rb = brentq(lambda r: disc(p, p, r) - D_b, 0.01, 0.999)
    rw = brentq(lambda r: disc(p, p, r) - D_w, 0.01, 0.999)
    return rb, min(rw, 0.999)

def simulate(rng, n, R, pI, delta, rb, rw, tau_acc, rho_rep=0.5):
    """Latent Z_{a,i,r} = mu_a + sqrt(rb)*b_i + sqrt(rw-rb)*g_{a,i} + tau_lat*w_{a,r} + sqrt(1-rw-tau^2)*e_{a,i,r}.
    Items are drawn FRESH each sim (test items are a random sample of the item population)."""
    pF = pI + delta
    muI, muF = norm.ppf(pI), norm.ppf(pF)
    # replicate-level SD of accuracy (tau_acc) -> latent shift SD
    tl = lambda p: tau_acc / norm.pdf(norm.ppf(p))
    tI, tF = tl(pI), tl(pF)
    b = rng.standard_normal((n, 1), dtype=np.float32)
    gI = rng.standard_normal((n, 1), dtype=np.float32); gF = rng.standard_normal((n, 1), dtype=np.float32)
    w0 = rng.standard_normal((R,)); wI = rng.standard_normal((R,))
    wF = rho_rep * wI + np.sqrt(1 - rho_rep**2) * w0            # A5: pool effect correlated across arms
    sg = np.sqrt(max(rw - rb, 0)); 
    def arm(mu, g, wr, t, tmax):
        se = np.sqrt(max(1 - rw - t**2, 1e-6))
        e = rng.standard_normal((n, R), dtype=np.float32)
        z = mu + np.sqrt(rb) * b + sg * g + t * wr[None, :] + se * e
        return (z > 0)
    cI = arm(muI, gI, wI, tI, 0); cF = arm(muF, gF, wF, tF, 0)
    return cF.astype(np.int8) - cI.astype(np.int8)       # d_{i,r} in {-1,0,1}

def tests(d, alphas):
    n, R = d.shape
    dbar = d.mean(); di = d.mean(1); dr = d.mean(0)
    out = {}
    # A1 naive pooled McNemar over all n*R pairs (normal approx)
    n10, n01 = (d == 1).sum(), (d == -1).sum()
    z1 = (n10 - n01) / np.sqrt(max(n10 + n01, 1)); out['A1_naive_pooled_McNemar'] = (z1, np.inf)
    # A2 item-level paired t on replicate-averaged differences (ignores replicate variance)
    out['A2_item_mean_t'] = (dbar / (di.std(ddof=1) / np.sqrt(n) + 1e-12), n - 1)
    # A3 crossed items x replicates variance components (quasi-F / Satterthwaite)
    ms_i = R * ((di - dbar) ** 2).sum() / (n - 1)
    ms_r = n * ((dr - dbar) ** 2).sum() / (R - 1)
    res = d - di[:, None] - dr[None, :] + dbar
    ms_e = (res ** 2).sum() / ((n - 1) * (R - 1))
    s_i = max((ms_i - ms_e) / R, 0); s_r = max((ms_r - ms_e) / n, 0)
    var = s_i / n + s_r / R + ms_e / (n * R)
    num = (s_i / n + s_r / R + ms_e / (n * R)) ** 2
    den = (s_i / n) ** 2 / (n - 1) + (s_r / R) ** 2 / (R - 1) + (ms_e / (n * R)) ** 2 / ((n - 1) * (R - 1))
    out['A3_crossed_VC_Satterthwaite'] = (dbar / np.sqrt(var + 1e-18), num / (den + 1e-30))
    # A4 replicate-level paired t (df = R-1), conditional on the test set
    out['A4_replicate_t'] = (dr.mean() / (dr.std(ddof=1) / np.sqrt(R) + 1e-12), R - 1)
    return out

def reject(stat, df, alpha):
    z, df = stat
    crit = norm.ppf(1 - alpha / 2) if np.isinf(df) else tdist.ppf(1 - alpha / 2, max(df, 1))
    return z > crit  # one direction: FT better (for type I use abs)

def run(cfg, nsim, seed):
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(nsim):
        d = simulate(rng, cfg['n'], cfg['R'], cfg['pI'], cfg['delta'], cfg['rb'], cfg['rw'], cfg['tau'], cfg.get('rho_rep', .5))
        for k, st in tests(d, None).items():
            for a in cfg['alphas']:
                acc.setdefault((k, a), []).append(abs(st[0]) > (norm.ppf(1 - a / 2) if np.isinf(st[1]) else tdist.ppf(1 - a / 2, max(st[1], 1))) if cfg['delta'] == 0 else reject(st[0:2], st[1], a))
    return {f"{k}|a={a}": float(np.mean(v)) for (k, a), v in acc.items()}

if __name__ == '__main__':
    mode = sys.argv[1]
    ALPHAS = [0.05, 0.025, 0.0083]   # per-contrast levels: single, Holm step 1 for 2 contrasts, Holm step 1 for 6 contrasts
    regimes = {   # (pI, D_b, D_w, tau_acc) ; 'pilot' = pilot levels, 'pess' = 1.5x discordance, 2x replicate SD   (A1-A4)
        'N1000_pilot': (0.75, 0.16, 0.10, 0.010), 'N1000_pess': (0.75, 0.24, 0.15, 0.020),
        'N5000_pilot': (0.90, 0.06, 0.045, 0.005), 'N5000_pess': (0.90, 0.09, 0.07, 0.010)}
    res = []
    if mode == 'typeI':
        for name, (p, Db, Dw, tau) in regimes.items():
            rb, rw = calibrate(p, Db, Dw)
            for R in (3, 5, 10):
                cfg = dict(n=1500, R=R, pI=p, delta=0.0, rb=rb, rw=rw, tau=tau, alphas=[0.05])
                r = run(cfg, 1500, 1); res.append(dict(regime=name, R=R, n=1500, **r)); print(res[-1], flush=True)
    else:
        for name, (p, Db, Dw, tau) in regimes.items():
            rb, rw = calibrate(p, Db, Dw)
            for n, R, dl in itertools.product((1000, 3080), (3, 5, 8, 12, 20), (0.01, 0.02, 0.03, 0.05)):
                cfg = dict(n=n, R=R, pI=p, delta=dl, rb=rb, rw=rw, tau=tau, alphas=ALPHAS)
                r = run(cfg, 1000, 7); res.append(dict(regime=name, n=n, R=R, delta=dl, **r)); print(res[-1], flush=True)
    json.dump(res, open(f'sim_{mode}.json', 'w'))
