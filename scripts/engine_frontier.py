"""Engine-level calibration frontier: marginal coverage vs per-engine reliability.

The knob is the fraction of calibration engines that must reach the target
coverage. Sweeping it traces the trade-off between marginal over-coverage and
per-engine reliability, which is the quantity a maintenance planner consumes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

TARGET = 0.90


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink(y, lo, hi, a=0.1):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def engine_q(units_cal, E_cal, engine_frac):
    cand = np.unique(np.quantile(E_cal, np.linspace(0.5, 1.0, 300)))
    ok = []
    for qq in cand:
        covs = [np.mean(E_cal[units_cal == u] <= qq) for u in np.unique(units_cal)]
        ok.append(np.mean(np.array(covs) >= TARGET))
    ok = np.array(ok)
    hit = np.where(ok >= engine_frac)[0]
    return float(cand[hit[0]]) if hit.size else float(cand[-1])


root = Path("experiments/punet_matrix")
FRACS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
rows = []
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    for seed in [42, 7]:
        for fold in range(5):
            pq = root / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
            if not pq.exists():
                continue
            c = np.genfromtxt(pq, delimiter=",", names=True)
            cal, tst = split(c["unit"])
            y, lo, hi, units = c["y_true"], c["y_lo"], c["y_hi"], c["unit"]
            E = np.maximum(lo - y, y - hi)
            for f in FRACS:
                qq = (q(E[cal], TARGET * (1 + 1 / cal.sum())) if f == 0.0
                      else engine_q(units[cal], E[cal], f))
                L, H = lo[tst] - qq, hi[tst] + qq
                ut = units[tst]
                cov = float(np.mean((y[tst] >= L) & (y[tst] <= H)))
                emin = float(min(np.mean((y[tst][ut == u] >= L[ut == u]) & (y[tst][ut == u] <= H[ut == u]))
                                 for u in np.unique(ut)))
                rows.append(dict(subset=sub, seed=seed, fold=fold, frac=f,
                                 cov=cov, emin=emin, wink=wink(y[tst], L, H)))

df = pd.DataFrame(rows)
print("engine_frac  marginal_cov  engine_min  winkler")
for f in FRACS:
    d = df[df.frac == f]
    print("  %.2f         %.3f         %.3f      %.1f" % (f, d["cov"].mean(), d["emin"].mean(), d["wink"].mean()))
print()
print("per-subset at frac=0.5:")
d = df[df.frac == 0.5]
print(d.groupby("subset")[["cov", "emin", "wink"]].mean().round(3).to_string())
df.to_csv("docs/tables/engine_frontier.csv", index=False)
