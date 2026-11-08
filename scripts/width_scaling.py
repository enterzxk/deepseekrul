"""Width-rescaling sweep (Codex D001 decision C).

For fixed centre c and learned half-width s, test

    E_i(a) = |y_i - c_i| - a * s_i
    q_a    = Quantile_{0.90}(E(a))
    interval = [c - (a*s + q_a), c + (a*s + q_a)]

a=0 -> constant width, a=1 -> current CQR, a>1 -> amplify the learned width.

Selection uses engine-disjoint inner splits of the CALIBRATION engines only;
the outer test engines are never used for selection.
"""

import numpy as np
import pandas as pd
from pathlib import Path

GRID = [0.0, 0.5, 1.0, 2.0, 4.0]
ALPHA = 0.1


def wink(y, lo, hi, a=ALPHA):
    return float(np.mean((hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def halfwidth(s, qa, a):
    return np.maximum(0.0, a * s + qa)


def eval_a(y, c, s, qa, a):
    h = halfwidth(s, qa, a)
    return wink(y, c - h, c + h)


root = Path("experiments/punet_matrix")
rows = []
for sub in ["FD001", "FD002", "FD003", "FD004"]:
    for seed in [42, 7, 2024, 13]:
        for fold in range(5):
            p = root / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
            if not p.exists():
                continue
            d = np.genfromtxt(p, delimiter=",", names=True)
            units = d["unit"]
            rng = np.random.default_rng(42)
            uu = np.unique(units).copy(); rng.shuffle(uu)
            n_cal = max(2, int(round(uu.size * 0.3)))
            cal_u, test_u = uu[:n_cal], uu[n_cal:]
            # inner split of calibration engines
            rng2 = np.random.default_rng(7)
            uu2 = cal_u.copy(); rng2.shuffle(uu2)
            half = max(1, len(uu2) // 2)
            inner_cal_u, inner_val_u = uu2[:half], uu2[half:]
            if len(inner_val_u) == 0:
                inner_val_u = inner_cal_u
            y, c = d["y_true"], (d["y_lo"] + d["y_hi"]) / 2.0
            s = (d["y_hi"] - d["y_lo"]) / 2.0
            m_ic = np.isin(units, inner_cal_u)
            m_iv = np.isin(units, inner_val_u)
            m_cal = np.isin(units, cal_u)
            m_test = np.isin(units, test_u)
            if m_ic.sum() < 20 or m_iv.sum() < 20:
                continue
            E_ic = np.abs(y[m_ic] - c[m_ic]) - 0.0 * s[m_ic]
            # inner selection
            best_a, best_w = None, np.inf
            inner = {}
            for a in GRID:
                qa = q(np.abs(y[m_ic] - c[m_ic]) - a * s[m_ic], (1 - ALPHA) * (1 + 1 / m_ic.sum()))
                w = eval_a(y[m_iv], c[m_iv], s[m_iv], qa, a)
                inner[a] = w
                if w < best_w:
                    best_a, best_w = a, w
            # outer evaluation at every a (for the record) and at the selected a
            out = {}
            for a in GRID:
                qa = q(np.abs(y[m_cal] - c[m_cal]) - a * s[m_cal], (1 - ALPHA) * (1 + 1 / m_cal.sum()))
                out[a] = eval_a(y[m_test], c[m_test], s[m_test], qa, a)
            rows.append(dict(subset=sub, seed=seed, fold=fold, a_star=best_a,
                             inner_best=best_w, inner_at_1=inner[1.0],
                             **{f"test_a{a}": out[a] for a in GRID}))

df = pd.DataFrame(rows)
print("cells:", len(df))
print("\nselected a* distribution:", df["a_star"].value_counts().sort_index().to_dict())
print("\nmean test Winkler by fixed a:")
for a in GRID:
    print("  a=%.1f  %6.1f" % (a, df[f"test_a{a}"].mean()))
sel = df.apply(lambda r: r[f"test_a{r.a_star}"], axis=1)
print("\nselected a* applied: %.1f" % sel.mean())
print("a=0 (constant):  %.1f" % df["test_a0.0"].mean())
print("a=1 (current):   %.1f" % df["test_a1.0"].mean())
print("\nper-subset: a* mean %.2f" % df.groupby("subset")["a_star"].mean().mean())
print(df.groupby("subset")[["a_star", "test_a0.0", "test_a1.0"]].mean().round(2).to_string())
df.to_csv("docs/tables/width_scaling.csv", index=False)
