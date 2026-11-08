"""Is engine-level calibration better than simply widening a marginal interval?

Compare, at matched marginal coverage, the per-engine minimum coverage of
(a) standard CQR inflated to that coverage, and (b) engine-level calibration.
If (b) does not beat (a), engine-level calibration is just inflation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

TARGET = 0.90
DELTAS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def stats(y, lo, hi, units, qq):
    L, H = lo - qq, hi + qq
    cov = float(np.mean((y >= L) & (y <= H)))
    emin = float(min(np.mean((y[units == u] >= L[units == u]) & (y[units == u] <= H[units == u]))
                     for u in np.unique(units)))
    return cov, emin


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
            ut = units[tst]
            # engine-level at frac 0.9
            qe = engine_q(units[cal], E[cal], 0.9)
            cov_e, emin_e = stats(y[tst], lo[tst], hi[tst], ut, qe)
            # marginal curve
            curve = []
            for dl in DELTAS:
                qm = q(E[cal], (1 - dl) * (1 + 1 / cal.sum()))
                cm, em = stats(y[tst], lo[tst], hi[tst], ut, qm)
                curve.append((cm, em))
            curve.sort()
            # interpolate marginal emin at the engine-level coverage
            xs = np.array([x for x, _ in curve]); ys = np.array([y2 for _, y2 in curve])
            emin_matched = float(np.interp(cov_e, xs, ys))
            rows.append(dict(subset=sub, seed=seed, fold=fold, cov_e=cov_e,
                             emin_engine=emin_e, emin_marginal_matched=emin_matched))

df = pd.DataFrame(rows)
print("cells:", len(df))
print("engine-level at frac 0.9: marginal coverage %.3f | engine_min %.3f" % (df["cov_e"].mean(), df["emin_engine"].mean()))
print("marginal CQR interpolated to the same coverage: engine_min %.3f" % df["emin_marginal_matched"].mean())
d = df["emin_engine"] - df["emin_marginal_matched"]
print("engine-level better in %d/%d cells | mean gain %+.3f" % (int((d > 0).sum()), len(d), d.mean()))
print()
print(df.groupby("subset")[["emin_engine", "emin_marginal_matched"]].mean().round(3).to_string())
df.to_csv("docs/tables/engine_vs_inflation.csv", index=False)
