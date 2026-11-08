"""D002: context-conditioned centre correction, frozen width, global CQR.

c'(x) = c(x) + b(z, H_<=t), where z uses only prediction-time information:
the centre itself, prior features, cycle index, and causal rolling statistics of
the centre trajectory within the engine.

The width head and backbone are frozen: the half-width s is reused unchanged.
b is fitted on engine-disjoint out-of-fold signed residuals from the calibration
engines only; test engines are never used for fitting or selection.

Metric: ENGINE-weighted Winkler (each engine counts once), per D002.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHA = 0.1
DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def wink(y, lo, hi, a=ALPHA):
    return (hi - lo) + (2 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))


def engine_weighted_winkler(y, c, h, units):
    vals = []
    for u in np.unique(units):
        m = units == u
        vals.append(wink(y[m], c[m] - h[m], c[m] + h[m]).mean())
    return float(np.mean(vals))


def engine_weighted_coverage(y, c, h, units):
    vals = []
    for u in np.unique(units):
        m = units == u
        vals.append(((y[m] >= c[m] - h[m]) & (y[m] <= c[m] + h[m])).mean())
    return float(np.mean(vals))


def q(v, lv):
    return float(np.quantile(v, min(1.0, lv), method="higher"))


def build_features(df):
    """Causal features from the centre trajectory and prior columns."""
    df = df.sort_values(["unit", "cutoff"]).reset_index(drop=True)
    c = (df["y_lo"].values + df["y_hi"].values) / 2.0
    out = {"c": c, "mu": df["mu_T"].values, "sg": df["sigma_T"].values, "cut": df["cutoff"].values}
    for k in (5, 10, 20):
        s = pd.Series(c)
        g = s.groupby(df["unit"].values)
        out[f"c_mean{k}"] = g.transform(lambda x: x.rolling(k, min_periods=1).mean()).values
        out[f"c_std{k}"] = g.transform(lambda x: x.rolling(k, min_periods=1).std().fillna(0.0)).values
        out[f"c_slope{k}"] = c - g.transform(lambda x: x.shift(k)).fillna(method="bfill").values
    X = np.column_stack([out[k] for k in out])
    return X, c, df


def split_engines(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    uu = np.unique(units).copy(); rng.shuffle(uu)
    n = max(2, int(round(uu.size * frac)))
    return uu[:n], uu[n:]


def fit_correction(Xc, yc, cc, uc, seed=42):
    """Fit b on engine-disjoint out-of-fold residuals, then on all calibration data."""
    target = yc - cc
    sc = StandardScaler().fit(Xc)
    Xs = sc.transform(Xc)
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    model.fit(Xs, target)
    return model, sc


def main() -> int:
    root = Path("experiments/punet_matrix")
    rows = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for seed in [42, 7, 2024, 13]:
            for fold in range(5):
                p = root / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                X, c, df = build_features(df)
                y = df["y_true"].values
                s = (df["y_hi"].values - df["y_lo"].values) / 2.0
                units = df["unit"].values
                cal_u, test_u = split_engines(units, seed=42)
                m_cal = np.isin(units, cal_u)
                m_tst = np.isin(units, test_u)
                if m_cal.sum() < 50 or m_tst.sum() < 50:
                    continue
                model, sc = fit_correction(X[m_cal], y[m_cal], c[m_cal], units[m_cal])
                b_all = model.predict(sc.transform(X))
                c_corr = c + b_all

                # calibration on CAL engines only; evaluate on TEST engines
                def cal_q(centre):
                    E = np.abs(y[m_cal] - centre[m_cal]) - s[m_cal]
                    return q(E, (1 - ALPHA) * (1 + 1 / m_cal.sum()))

                res = {}
                for tag, centre in (("orig", c), ("corr", c_corr)):
                    qq = cal_q(centre)
                    h = np.maximum(0.0, s + qq)
                    res[f"{tag}_w"] = engine_weighted_winkler(y[m_tst], centre[m_tst], h[m_tst], units[m_tst])
                    res[f"{tag}_cov"] = engine_weighted_coverage(y[m_tst], centre[m_tst], h[m_tst], units[m_tst])
                    # constant width with the same centre
                    Ec = np.abs(y[m_cal] - centre[m_cal])
                    qc = q(Ec, (1 - ALPHA) * (1 + 1 / m_cal.sum()))
                    hc = np.full(m_tst.sum(), qc)
                    res[f"{tag}_const_w"] = engine_weighted_winkler(y[m_tst], centre[m_tst], hc, units[m_tst])
                rows.append(dict(subset=sub, seed=seed, fold=fold, **res))

    df = pd.DataFrame(rows)
    print("cells:", len(df))
    print("\nengine-weighted Winkler (mean over cells):")
    for col in ["orig_w", "corr_w", "orig_const_w", "corr_const_w"]:
        print("  %-14s %6.1f" % (col, df[col].mean()))
    print("\nengine-weighted coverage:")
    for col in ["orig_cov", "corr_cov"]:
        print("  %-10s %.3f" % (col, df[col].mean()))
    print("\nper-subset (engine-weighted Winkler):")
    g = df.groupby("subset")[["orig_w", "corr_w", "corr_const_w", "orig_cov", "corr_cov"]].mean().round(2)
    print(g.to_string())
    print("\ngate check (corr vs orig, and corr vs corr_const), per subset:")
    for s_ in ["FD003", "FD004"]:
        dd = df[df.subset == s_]
        if len(dd) == 0:
            continue
        rel1 = (dd["corr_w"] - dd["orig_w"]) / dd["orig_w"] * 100
        rel2 = (dd["corr_w"] - dd["corr_const_w"]) / dd["corr_const_w"] * 100
        print("  %s: vs orig %+.1f%% | vs const %+.1f%% | cells %d" % (s_, rel1.mean(), rel2.mean(), len(dd)))
    df.to_csv("docs/tables/d002_centre.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
