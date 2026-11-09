"""Independent recomputation of the headline claims in the IJPHM v2 ledger.

Deliberately does NOT import scripts/analyze_mcd80.py, scripts/paper_evidence.py
or scripts/freeze_audit.py. The protocol definitions are re-implemented from the
documented spec (nominal level delta=0.1, engine-level 30% calibration split
with seed 42, Winkler with alpha fixed at 0.1), so an implementation bug in the
authoring scripts cannot propagate here.

Checks (ledger IDs from docs/manuscripts/ijphm_v2/CLAIM_LEDGER.md):
  C1  CQR vs constant width, nominal level: wins/80, fold wins/20, mean, median
  C3  realised coverage CQR vs constant
  C6  MC-Dropout vs same-centre constant (seeded table): wins, mean, median
  C7  per-subset split of C6
  C8  multi-OC vs single-OC fold separation + Fisher exact two-sided p

Exit code 0 if every checked number matches the ledger within tolerance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DELTA = 0.1
ALPHA = 0.1
CALIB_FRAC = 0.3
SPLIT_SEED = 42
TOL = 0.05  # percentage points


def split(units, frac=CALIB_FRAC, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def wink(y, lo, hi, a=ALPHA):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def q_level(n, delta=DELTA):
    return min(1.0, (1.0 - delta) * (1.0 + 1.0 / n))


def cqr_nominal(y, lo, hi, cal, tst):
    s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    q = float(np.quantile(s, q_level(s.size), method="higher"))
    L, H = lo[tst] - q, hi[tst] + q
    return float(np.mean((y[tst] >= L) & (y[tst] <= H))), wink(y[tst], L, H)


def scaled_nominal(y, mu, w, cal, tst):
    q = float(np.quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9),
                          q_level(int(cal.sum())), method="higher"))
    L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
    return float(np.mean((y[tst] >= L) & (y[tst] <= H))), wink(y[tst], L, H)


def fisher_two_sided(a, b, c, d):
    """Exact two-sided Fisher test on a 2x2 table [[a,b],[c,d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    obs = p(a)
    tot = 0.0
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        px = p(x)
        if px <= obs * (1 + 1e-9):
            tot += px
    return tot


def main() -> int:
    mat = Path("experiments/punet_matrix")
    rows = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for seed in [42, 7, 2024, 13]:
            for fold in range(5):
                d = mat / sub / f"f{fold}_s{seed}"
                if not (d / "preds_cqr.csv").exists():
                    continue
                c = np.genfromtxt(d / "preds_cqr.csv", delimiter=",", names=True)
                p = np.genfromtxt(d / "preds_point.csv", delimiter=",", names=True)
                cal, tst = split(c["unit"])
                mu = (p["y_lo"] + p["y_hi"]) / 2.0
                cov_c, w_c = cqr_nominal(c["y_true"], c["y_lo"], c["y_hi"], cal, tst)
                cov_k, w_k = scaled_nominal(p["y_true"], mu, np.ones_like(mu), cal, tst)

                # C4: per-engine coverage of the CQR interval on held-out engines
                y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
                s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
                q = float(np.quantile(s, q_level(s.size), method="higher"))
                L, H = lo - q, hi + q
                hit = (y >= L) & (y <= H)
                units_t = c["unit"][tst]
                per_engine = pd.Series(hit[tst]).groupby(units_t).mean()
                rows.append(dict(subset=sub, seed=seed, fold=fold,
                                 cqr=cov_c, cqr_w=w_c, const=cov_k, const_w=w_k,
                                 rel=(w_c - w_k) / w_k * 100,
                                 engine_min=float(per_engine.min()),
                                 n_engines=int(per_engine.size)))
    df = pd.DataFrame(rows)

    fold = df.groupby(["subset", "fold"]).agg(rel=("rel", "mean"), cqr=("cqr", "mean"),
                                              const=("const", "mean")).reset_index()

    out = {}
    out["C1_cells_wins"] = f"{int((df.rel < 0).sum())}/{len(df)}"
    out["C1_folds_wins"] = f"{int((fold.rel < 0).sum())}/{len(fold)}"
    out["C1_mean"] = round(float(df.rel.mean()), 2)
    out["C1_median"] = round(float(df.rel.median()), 2)
    out["C3_cov_cqr"] = round(float(df.cqr.mean()), 4)
    out["C3_cov_const"] = round(float(df["const"].mean()), 4)
    out["C3_delta_pp"] = round(float((df.cqr.mean() - df["const"].mean()) * 100), 2)
    out["C3_max_abs_cell_gap_pp"] = round(float((df.cqr - df["const"]).abs().max() * 100), 2)

    m = pd.read_csv("docs/tables/mcd80_cells_seeded.csv")
    m["rel"] = (m.mcd_nom - m.const_mc_nom) / m.const_mc_nom * 100
    out["C6_wins"] = f"{int((m.rel < 0).sum())}/{len(m)}"
    out["C6_mean"] = round(float(m.rel.mean()), 2)
    out["C6_median"] = round(float(m.rel.median()), 2)
    for s in ["FD001", "FD002", "FD003", "FD004"]:
        sub = m[m.subset == s]
        out[f"C7_{s}"] = f"{round(float(sub.rel.mean()), 2)}% ({int((sub.rel < 0).sum())}/{len(sub)})"

    # C4: minimum per-engine coverage over the 80 cells, per subset
    for s in ["FD001", "FD002", "FD003", "FD004"]:
        sub = df[df.subset == s]
        worst = sub.loc[sub.engine_min.idxmin()]
        out[f"C4_{s}_min"] = round(float(worst.engine_min), 4)
        out[f"C4_{s}"] = (f"min {worst.engine_min:.4f} "
                          f"(f{int(worst.fold)}_s{int(worst.seed)}); "
                          f"median cell-min {sub.engine_min.median():.4f}")
        out[f"C4_{s}_n_engines"] = int(sub.n_engines.median())

    foldm = m.groupby(["subset", "fold"])["rel"].mean().reset_index()
    multi = foldm[foldm.subset.isin(["FD002", "FD004"])]["rel"]
    single = foldm[foldm.subset.isin(["FD001", "FD003"])]["rel"]
    a = int((multi < 0).sum()); b = int((multi >= 0).sum())
    c = int((single < 0).sum()); dd = int((single >= 0).sum())
    out["C8_multi"] = f"{a}/{a+b} better, mean {multi.mean():+.2f}%"
    out["C8_single"] = f"{c}/{c+dd} better, mean {single.mean():+.2f}%"
    out["C8_fisher_p"] = round(fisher_two_sided(a, b, c, dd), 4)

    print(json.dumps(out, ensure_ascii=False, indent=1))

    # ---- compare with the ledger ----
    expect = {
        "C1_cells_wins": "75/80", "C1_folds_wins": "20/20", "C1_mean": -10.22,
        "C1_median": -11.30, "C3_cov_cqr": 0.8952, "C3_cov_const": 0.8994,
        "C3_delta_pp": -0.42, "C3_max_abs_cell_gap_pp": 8.84,
        "C6_wins": "43/80", "C6_mean": 9.49, "C6_median": -0.97,
        "C4_FD001_min": 0.4214, "C4_FD002_min": 0.2375,
        "C4_FD003_min": 0.4933, "C4_FD004_min": 0.1217,
        "C8_fisher_p": 0.0055,
    }
    fails = []
    for k, v in expect.items():
        got = out[k]
        if isinstance(v, str):
            ok = got == v
        else:
            tol = 1e-3 if "C4_" in k else TOL
            ok = abs(float(got) - float(v)) <= tol
        print(f"{'PASS' if ok else 'FAIL'} {k}: got {got} expected {v}")
        if not ok:
            fails.append(k)
    print(f"\n{len(expect) - len(fails)}/{len(expect)} independent checks match")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
