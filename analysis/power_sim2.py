"""Round 2: overlapping finite-population pools (BANKING77 M=9003 rows), Holm-6 first-step alpha,
minimal detectable gap (MDG), continuous BillSum scores. Imports power_sim.tests (crossed items x replicates).
Run: python3 -I power_sim2.py {typeI|mdg|billsum}"""
import numpy as np, json, sys, itertools
from scipy.stats import norm, t as tdist
sys.path.insert(0, '.')
from power_sim import tests

M = 9003
LEV = {200: dict(p=.70, tau=(.015, .030)), 1000: dict(p=.80, tau=(.010, .020)), 3000: dict(p=.87, tau=(.007, .014))}
RHO = {'opt': (.85, .93), 'pess': (.65, .85)}   # (rho_b between arms, rho_w within arm); assumptions A1-A2

def sim_b77(rng, n, R, N, reg, delta, truth, rho_rep=.5):
    p = LEV[N]['p']; tau_acc = LEV[N]['tau'][0 if reg == 'opt' else 1]
    rb, rw = RHO[reg]
    muI, muF = norm.ppf(p), norm.ppf(p + delta)
    tl = lambda q: tau_acc / norm.pdf(norm.ppf(q))
    # pool effects from a finite population of M rows: pool mean of row effects, scaled to unit SD for iid pools
    zI = rng.standard_normal(M); z0 = rng.standard_normal(M); zF = rho_rep * zI + np.sqrt(1 - rho_rep**2) * z0
    cI, cF = (zI.mean(), zF.mean()) if truth == 'cond' else (0., 0.)   # 'cond': centred on the 9,003 rows
    wI = np.empty(R); wF = np.empty(R)
    for r in range(R):
        idx = rng.choice(M, N, replace=False)
        wI[r] = np.sqrt(N) * (zI[idx].mean() - cI); wF[r] = np.sqrt(N) * (zF[idx].mean() - cF)
    b = rng.standard_normal((n, 1), dtype=np.float32); gI = rng.standard_normal((n, 1), dtype=np.float32); gF = rng.standard_normal((n, 1), dtype=np.float32)
    sg = np.sqrt(rw - rb)
    def arm(mu, g, w, t):
        se = np.sqrt(1 - rw - t**2); e = rng.standard_normal((n, R), dtype=np.float32)
        return (mu + np.sqrt(rb) * b + sg * g + t * w[None, :] + se * e) > 0
    return arm(muF, gF, wF, tl(p + delta)).astype(np.int8) - arm(muI, gI, wI, tl(p)).astype(np.int8)

def a3(d, f=None):
    """crossed-model z/df; if f is given, super-population correction: s_r*(1/R + f/(1-f)) instead of s_r/R"""
    n, R = d.shape; dbar = d.mean(); di = d.mean(1); dr = d.mean(0)
    ms_i = R * ((di - dbar)**2).sum() / (n - 1); ms_r = n * ((dr - dbar)**2).sum() / (R - 1)
    res = d - di[:, None] - dr[None, :] + dbar; ms_e = (res**2).sum() / ((n - 1) * (R - 1))
    s_i = max((ms_i - ms_e) / R, 0); s_r = max((ms_r - ms_e) / n, 0)
    cr = (1 / R) if f is None else (1 / R + f / (1 - f))
    v = s_i / n + s_r * cr + ms_e / (n * R)
    den = (s_i / n)**2 / (n - 1) + (s_r * cr)**2 / (R - 1) + (ms_e / (n * R))**2 / ((n - 1) * (R - 1))
    return dbar / np.sqrt(v + 1e-18), v**2 / (den + 1e-30)

def rej(z, df, a, two=False):
    c = tdist.ppf(1 - a / 2, max(df, 1)); return (abs(z) if two else z) > c

if __name__ == '__main__':
    mode = sys.argv[1]; rng = np.random.default_rng(11); out = []
    if mode == 'typeI':
        for N, R, reg, truth in itertools.product((200, 1000, 3000), (5, 8), ('opt', 'pess'), ('cond', 'super')):
            ra = rc = 0; S = 2000
            for _ in range(S):
                d = sim_b77(rng, 1500, R, N, reg, 0.0, truth)
                ra += rej(*a3(d), .05, True); rc += rej(*a3(d, N / M), .05, True)
            out.append(dict(N=N, R=R, reg=reg, truth=truth, A3=ra / S, A3_super_corrected=rc / S)); print(out[-1], flush=True)
    elif mode == 'mdg':
        grid = [.01, .02, .03, .04, .05, .06, .08, .10]
        for N, R, reg in itertools.product((200, 1000, 3000), (5, 8, 10), ('opt', 'pess')):
            row = dict(N=N, R=R, reg=reg, pw={})
            for dl in grid:
                c = {'cond_.0083': 0, 'cond_.025': 0, 'super_corr_.0083': 0}; S = 500
                for _ in range(S):
                    d = sim_b77(rng, 3080, R, N, reg, dl, 'cond'); z, df = a3(d)
                    c['cond_.0083'] += rej(z, df, .0083); c['cond_.025'] += rej(z, df, .025)
                    d = sim_b77(rng, 3080, R, N, reg, dl, 'super'); z, df = a3(d, N / M)
                    c['super_corr_.0083'] += rej(z, df, .0083)
                row['pw'][dl] = {k: v / S for k, v in c.items()}
            out.append(row); print(N, R, reg, {k: [row['pw'][g][k] for g in grid] for k in row['pw'][grid[0]]}, flush=True)
    else:  # continuous BillSum, d_ir = delta + a_i + b_r + e_ir (ROUGE-L F1 units)
        si, se = .04, .055
        for n, R, sr, N in [(1000, 5, .002, 1000), (1000, 5, .004, 1000), (1000, 5, .008, 1000), (1000, 3, .004, 1000), (1000, 8, .004, 1000), (500, 5, .004, 1000), (1000, 5, .004, 200), (1000, 5, .008, 3000)]:
            row = dict(n=n, R=R, sigma_rep=sr, N=N, pw={})
            f = N / 18949
            for dl in [.002, .004, .006, .008, .010, .015, .020, .030]:
                c = 0; cc = 0; S = 800
                for _ in range(S):
                    d = dl + si * rng.standard_normal((n, 1)) + sr * rng.standard_normal((1, R)) + se * rng.standard_normal((n, R))
                    c += rej(*a3(d), .0083)
                row['pw'][dl] = c / S
            out.append(row); print(row, flush=True)
    json.dump(out, open(f'sim2_{mode}.json', 'w'))
