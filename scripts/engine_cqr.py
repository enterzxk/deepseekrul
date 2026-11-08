"""Engine-level conformal calibration.

Standard CQR targets marginal (window-level) coverage, which leaves individual
engines badly covered. Here the calibration target is moved to the engine level:
choose q_hat so that a fraction (1-delta) of CALIBRATION ENGINES reaches coverage
>= 1-delta, then apply that q_hat to test windows.

This directly optimises the per-engine reliability that maintenance consumes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

TARGET = 0.90


def split(u, frac=0.3, seed=42):
    rng = np.random.default_rng(seed); uu = np.unique(u).copy(); rng.shuffle(uu)
    n = max(1, int(round(uu.size * frac)))
    return np.isin(u, uu[:n]), np.isin(u, uu[n:])


def wink(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def engine_qhat(units_cal, E_cal, target=TARGET, engine_frac=TARGET):
    """Smallest q such that engine_frac of calibration engines reach target."""
    cand = np.unique(np.quantile(E_cal, np.linspace(0.5, 1.0, 200)))
    ok = []
    for qq in cand:
        covs = []
        for u in np.unique(units_cal):
            m = units_cal == u
            covs.append(np.mean(E_cal[m] <= qq))
        ok.append(np.mean(np.array(covs) >= target))
    ok = np.array(ok)
    idx = np.argmax(ok >= engine_frac)
    return float(cand[idx]) if ok.max() >= engine_frac else float(cand[-1])


def eval_q(y, lo, hi, tst_units, qq, alpha=0.1):
    L, H = lo - qq, hi + qq
    cov = np.mean((y >= L) & (y <= H))
    per_engine = [np.mean((y[tst_units == u] >= L[tst_units == u]) & (y[tst_units == u] <= H[tst_units == u]))
                  for u in np.unique(tst_units)]
    return cov, float(np.min(per_engine)), wink(y, L, H, alpha)


root = Path("experiments/punet_matrix")
rows = []
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    for seed in [42, 7]:
        for fold in range(5):
            d = root / sub / f"f{fold}_s{seed}"
            pq = d / "preds_cqr.csv"
            if not pq.exists():
                continue
            c = np.genfromtxt(pq, delimiter=",", names=True)
            cal, tst = split(c["unit"])
            y, lo, hi, units = c["y_true"], c["y_lo"], c["y_hi"], c["unit"]
            E = np.maximum(lo - y, y - hi)
            # standard marginal CQR
            qm = q(E[cal], TARGET * (1 + 1 / cal.sum()))
            cov_m, emin_m, w_m = eval_q(y[tst], lo[tst], hi[tst], units[tst], qm)
            # engine-level
            qe = engine_qhat(units[cal], E[cal])
            cov_e, emin_e, w_e = eval_q(y[tst], lo[tst], hi[tst], units[tst], qe)
            rows.append(dict(subset=sub, seed=seed, fold=fold,
                             cov_m=cov_m, emin_m=emin_m, w_m=w_m,
                             cov_e=cov_e, emin_e=emin_e, w_e=w_e, q_m=qm, q_e=qe))

df = pd.DataFrame(rows)
print("cells:", len(df))
print(df.groupby("subset")[["cov_m", "emin_m", "w_m", "cov_e", "emin_e", "w_e"]].mean().round(3).to_string())
print()
print("overall: marginal-CQR engine_min %.3f | engine-CQR engine_min %.3f" % (df.emin_m.mean(), df.emin_e.mean()))
print("overall: marginal-CQR coverage %.3f | engine-CQR coverage %.3f" % (df.cov_m.mean(), df.cov_e.mean()))
print("overall: marginal-CQR Winkler %.1f | engine-CQR Winkler %.1f" % (df.w_m.mean(), df.w_e.mean()))
print("engine_min improved in %d/%d cells" % (int((df.emin_e > df.emin_m).sum()), len(df)))
df.to_csv("docs/tables/engine_level.csv", index=False)
