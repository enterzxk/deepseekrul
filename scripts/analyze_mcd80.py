"""MC-Dropout 80-cell backfill: nominal-level (deployable) + aligned diagnostic.

Why two protocols
-----------------
D015 removed the coverage-aligned comparison from the paper evidence system: it
interpolates a frontier using test labels and therefore cannot represent a
deployable selection rule. So the PRIMARY comparison here is at the nominal
level (delta = 0.1 on the same engine split), which is what a user would run.
The aligned number is kept as a post-hoc efficiency diagnostic only, clearly
labelled, because it is the protocol under which the original 10-cell
observation ("FD003 catastrophic, FD002 better") was made.

Centre confound (Codex D019 risk #2)
------------------------------------
MC-Dropout replaces the deterministic centre with the mean of 30 stochastic
forward passes. Comparing it against a constant-width interval around the
*deterministic* centre confounds two changes: centre and scale. Therefore the
scale comparison uses a control with the SAME centre:
    mcd_nom       : mu_mc +/- q * sd        (scale = dropout std)
    const_mc_nom  : mu_mc +/- q * 1         (same centre, no conditioning)
The difference between those two is attributable to the scale alone.
`const_nom` (deterministic centre, constant width) is kept as the frozen
baseline reference.

Protocol functions are copied verbatim from scripts/table_all_methods.py; the
run prints a regression check against docs/tables/cells_all.csv (max |diff| must
be ~0) so a divergence from the frozen protocol cannot pass silently.

Aggregation: the fold is the reporting unit (n=20); seeds are averaged inside a
fold. 20 folds share training data (cross-validation overlap) and 80 cells share
one 5-fold split, so no independent-sample CI or significance test is claimed;
the bootstrap over folds is descriptive only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DELTAS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def winkler(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2.0 / a) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))))


def pts_scaled(y, mu, w, cal, tst):
    out = []
    for dl in DELTAS:
        q = float(np.quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9),
                              min(1.0, (1 - dl) * (1 + 1 / cal.sum())), method="higher"))
        L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)))
    return sorted(out)


def pts_cqr(y, lo, hi, cal, tst):
    out = []
    for dl in DELTAS:
        s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
        q = float(np.quantile(s, min(1.0, (1 - dl) * (1 + 1 / s.size)), method="higher"))
        L, H = lo[tst] - q, hi[tst] + q
        out.append((float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)))
    return sorted(out)


def at(pts, target=0.90):
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0)
    return min(pts, key=lambda p: abs(p[0] - target))[1]


def at2(pts, target=0.90):
    """Like at(), but also reports whether target was bracketed.

    A method whose frontier never reaches the target coverage cannot be
    compared at that coverage; at() would silently return the nearest point,
    which can be far too wide. Cells with ok=False must be excluded from any
    aligned comparison and counted.
    """
    for (c0, w0), (c1, w1) in zip(pts, pts[1:]):
        if c0 <= target <= c1 and c1 > c0:
            t = (target - c0) / (c1 - c0)
            return w0 + t * (w1 - w0), True
    return min(pts, key=lambda p: abs(p[0] - target))[1], False


def nominal_scaled(y, mu, w, cal, tst, delta=0.1):
    q = float(np.quantile(np.abs(y[cal] - mu[cal]) / np.maximum(w[cal], 1e-9),
                          min(1.0, (1 - delta) * (1 + 1 / cal.sum())), method="higher"))
    L, H = mu[tst] - q * w[tst], mu[tst] + q * w[tst]
    return float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)


def nominal_cqr(y, lo, hi, cal, tst, delta=0.1):
    s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    q = float(np.quantile(s, min(1.0, (1 - delta) * (1 + 1 / s.size)), method="higher"))
    L, H = lo[tst] - q, hi[tst] + q
    return float(np.mean((y[tst] >= L) & (y[tst] <= H))), winkler(y[tst], L, H, 0.1)


def build_frame(mat, mcd_root, subsets=SUBSETS, seeds=SEEDS) -> pd.DataFrame:
    """Per-cell comparison frame for one MC-Dropout root."""
    rows = []
    for sub in subsets:
        for seed in seeds:
            for fold in range(5):
                d, b = Path(mat) / sub / f"f{fold}_s{seed}", Path(mcd_root) / sub / f"f{fold}_s{seed}"
                pq, pp = d / "preds_cqr.csv", d / "preds_point.csv"
                if not (pq.exists() and pp.exists()):
                    continue
                c = np.genfromtxt(pq, delimiter=",", names=True)
                p = np.genfromtxt(pp, delimiter=",", names=True)
                cal, tst = engine_split(c["unit"])
                y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
                mu = (p["y_lo"] + p["y_hi"]) / 2.0
                rec = dict(subset=sub, seed=seed, fold=fold,
                           const=at(pts_scaled(p["y_true"], mu, np.ones_like(mu), cal, tst)),
                           cqr=at(pts_cqr(y, lo, hi, cal, tst)),
                           const_nom_cov=nominal_scaled(y, mu, np.ones_like(mu), cal, tst)[0],
                           const_nom=nominal_scaled(y, mu, np.ones_like(mu), cal, tst)[1],
                           cqr_nom_cov=nominal_cqr(y, lo, hi, cal, tst)[0],
                           cqr_nom=nominal_cqr(y, lo, hi, cal, tst)[1])

                pm = b / "preds_mcd.csv"
                if pm.exists():
                    m = np.genfromtxt(pm, delimiter=",", names=True)
                    c2, t2 = engine_split(m["unit"])
                    rec["mcd"], rec["mcd_al_ok"] = at2(pts_scaled(m["y_true"], m["mu"], m["sd"], c2, t2))
                    rec["const_mc"], rec["const_mc_al_ok"] = at2(
                        pts_scaled(m["y_true"], m["mu"], np.ones_like(m["mu"]), c2, t2))
                    cov, w = nominal_scaled(m["y_true"], m["mu"], m["sd"], c2, t2)
                    rec["mcd_nom_cov"], rec["mcd_nom"] = cov, w
                    cov, w = nominal_scaled(m["y_true"], m["mu"], np.ones_like(m["mu"]), c2, t2)
                    rec["const_mc_nom_cov"], rec["const_mc_nom"] = cov, w
                    rec["mcd_sd_mean"] = float(np.mean(m["sd"]))
                    rec["sd_cv"] = float(np.std(m["sd"]) / np.mean(m["sd"]))
                rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="experiments/punet_matrix")
    ap.add_argument("--mcd", default="experiments/punet_mcd80")
    ap.add_argument("--cells-csv", default="docs/tables/cells_all.csv")
    ap.add_argument("--out", default="docs/tables/mcd80_cells.csv")
    args = ap.parse_args()

    df = build_frame(args.matrix, args.mcd)

    reg = {"checked": 0}
    if Path(args.cells_csv).exists():
        old = pd.read_csv(args.cells_csv)
        key = ["subset", "seed", "fold"]
        j = df.merge(old[key + ["const", "cqr"]], on=key, suffixes=("", "_old"))
        if len(j):
            reg = dict(checked=len(j),
                       max_abs_d_const=float(np.max(np.abs(j["const"] - j["const_old"]))),
                       max_abs_d_cqr=float(np.max(np.abs(j["cqr"] - j["cqr_old"]))))
    print(f"protocol check vs {args.cells_csv}: {reg}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_mcd = int(df["mcd"].notna().sum()) if "mcd" in df else 0
    print(f"cells: {len(df)} | with mcd: {n_mcd}")
    if n_mcd == 0:
        return 0

    df["r_scale_nom"] = (df["mcd_nom"] - df["const_mc_nom"]) / df["const_mc_nom"] * 100
    df["r_mcd_const_nom"] = (df["mcd_nom"] - df["const_nom"]) / df["const_nom"] * 100
    df["r_mcd_cqr_nom"] = (df["mcd_nom"] - df["cqr_nom"]) / df["cqr_nom"] * 100
    df["r_cqr_const_nom"] = (df["cqr_nom"] - df["const_nom"]) / df["const_nom"] * 100
    df["r_mcd_const_al"] = (df["mcd"] - df["const"]) / df["const"] * 100
    df["r_scale_al"] = (df["mcd"] - df["const_mc"]) / df["const_mc"] * 100

    def summarize(col, label):
        v = df[col].dropna()
        print(f"{label}: n={len(v)} wins={int((v < 0).sum())}/{len(v)} "
              f"mean={v.mean():+.1f}% median={v.median():+.1f}%")

    print("\n--- PRIMARY: nominal level (delta=0.1, deployable) ---")
    summarize("r_scale_nom", "  scale only   mcd vs const(same centre)")
    summarize("r_mcd_const_nom", "  mcd vs const (deterministic centre)")
    summarize("r_mcd_cqr_nom", "  mcd vs CQR")
    summarize("r_cqr_const_nom", "  CQR vs const  (reference)")
    print(f"  mean realised coverage: mcd={df['mcd_nom_cov'].mean():.3f} "
          f"const_mc={df['const_mc_nom_cov'].mean():.3f} cqr={df['cqr_nom_cov'].mean():.3f}")

    print("\n--- DIAGNOSTIC: coverage-aligned to 0.90 (D015: not paper evidence) ---")
    summarize("r_scale_al", "  scale only   mcd vs const(same centre)")
    summarize("r_mcd_const_al", "  mcd vs const")
    summarize("r_cqr_const_nom", "  (CQR reference above)")
    if "mcd_al_ok" in df:
        ok = df["mcd_al_ok"].fillna(False).astype(bool) & df["const_mc_al_ok"].fillna(False).astype(bool)
        n_bad_mcd = int((~df["mcd_al_ok"].fillna(False).astype(bool)).sum())
        n_bad_cmc = int((~df["const_mc_al_ok"].fillna(False).astype(bool)).sum())
        print(f"  frontiers that never reach 0.90 (unbracketed): "
              f"mcd={n_bad_mcd} const_mc={n_bad_cmc} of {n_mcd}")
        v = ((df.loc[ok, "mcd"] - df.loc[ok, "const_mc"]) / df.loc[ok, "const_mc"] * 100).dropna()
        print(f"  scale only, bracketed cells only (n={len(v)}): wins={int((v < 0).sum())}/{len(v)} "
              f"mean={v.mean():+.1f}% median={v.median():+.1f}%")

    print("\nper-subset mean rel. delta [%] (nominal):")
    print(df.groupby("subset")[["r_scale_nom", "r_mcd_const_nom", "r_mcd_cqr_nom"]]
            .agg(["count", "mean", "median"]).round(1).to_string())

    for col in ["r_scale_nom", "r_mcd_const_nom"]:
        fold = df.dropna(subset=[col]).groupby(["subset", "fold"])[col].mean()
        vals = fold.to_numpy()
        if len(vals) < 3:
            continue
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(10000)])
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        print(f"\nfold-level {col}: n={len(vals)} mean={vals.mean():+.1f}% "
              f"median={np.median(vals):+.1f}% wins={int((vals < 0).sum())}/{len(vals)} "
              f"bootstrap95=[{lo_ci:+.1f}%, {hi_ci:+.1f}%] (descriptive only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
