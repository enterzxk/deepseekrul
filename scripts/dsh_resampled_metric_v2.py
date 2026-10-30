"""Corrected resampling / clustering analysis for RUL error comparisons.

Fixes over dsh_resampled_metric.py (kept as superseded):
  * explicit key (subset, cell, unit, seed, cutoff) instead of unit->cutoff dict
    (the old dict silently kept only the last seed per unit/cutoff);
  * cutoffs aligned on the *intersection* actually shared by the compared cells,
    not by list position modulo;
  * per-seed metrics and cross-seed spread reported separately from the ensemble;
  * the cutoff-conditional spread is labelled as such, NOT as a population CI;
  * an engine-clustered bootstrap (all cutoffs and seeds of an engine resampled
    together) provides the interval for the actual RMSE difference.

Outputs go to results/dsh/review_r1/.
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

CUT_COL = "abs_cut"


def load_long(root, subset):
    """Return long records: subset, cell, unit, seed, cutoff, sq_err."""
    recs = []
    for path in glob.glob(str(Path(root) / "*" / subset / "*" / "protocol_c_predictions.csv")):
        cell = Path(path).parts[-4]
        seed = int(Path(path).parts[-2].split("_s")[-1])
        df = pd.read_csv(path)
        df["cell"] = cell
        df["model_seed"] = seed
        df["sq_err"] = df["signed_error"].astype(float) ** 2
        recs.append(df[["unit_id", CUT_COL, "model_seed", "sq_err", "cell"]])
    if not recs:
        return None
    out = pd.concat(recs, ignore_index=True)
    out["subset"] = subset
    return out


def per_seed_metrics(long):
    g = long.groupby(["subset", "cell", "model_seed"])
    m = g["sq_err"].mean().pow(0.5).rename("rmse").reset_index()
    return m


def ensemble_metrics(long):
    """Ensemble = mean of squared errors across seeds, then RMSE (no signed averaging)."""
    g = long.groupby(["subset", "cell", "unit_id", CUT_COL])["sq_err"].mean().reset_index()
    return g.groupby(["subset", "cell"])["sq_err"].mean().pow(0.5).rename("rmse_ensemble").reset_index()


def clustered_bootstrap_diff(long, a, b, reps=2000, seed=20261022):
    """Engine-clustered: resample engines, keep all their cutoffs and seeds."""
    la = long[long.cell == a]
    lb = long[long.cell == b]
    units = sorted(set(la.unit_id) & set(lb.unit_id))
    if len(units) < 5:
        return None
    ia = la.set_index(["unit_id", CUT_COL, "model_seed"])["sq_err"]
    ib = lb.set_index(["unit_id", CUT_COL, "model_seed"])["sq_err"]
    idx = ia.index.intersection(ib.index)
    ia, ib = ia.loc[idx], ib.loc[idx]
    unit_of = idx.get_level_values(0)
    rng = np.random.default_rng(seed)
    by_unit = {u: np.arange(len(idx))[unit_of == u] for u in units}
    diffs = []
    for _ in range(reps):
        take = rng.choice(units, len(units), replace=True)
        rows = np.concatenate([by_unit[u] for u in take])
        ra = np.sqrt(ia.iloc[rows].mean())
        rb = np.sqrt(ib.iloc[rows].mean())
        diffs.append(ra - rb)
    d = np.asarray(diffs)
    return {"a": a, "b": b, "engines": len(units),
            "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "prob_a_better": float((d < 0).mean()),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}


def conditional_cutoff_spread(long, cell, reps=2000, seed=20261022):
    """Sensitivity ONLY: one cutoff per engine, same cutoff shared by all seeds.

    The resulting percentiles are conditional on these engines/seeds/cutoffs and
    are not a population confidence interval.
    """
    d = long[long.cell == cell]
    units = sorted(d.unit_id.unique())
    cuts = sorted(d[CUT_COL].unique())
    by = {(u, c): d[(d.unit_id == u) & (d[CUT_COL] == c)]["sq_err"].to_numpy() for u in units for c in cuts}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        chosen = rng.integers(0, len(cuts), len(units))
        errs = [by[(u, cuts[k])] for u, k in zip(units, chosen)]
        flat = np.concatenate(errs)
        vals.append(float(np.sqrt(flat.mean())))
    v = np.asarray(vals)
    return {"cell": cell, "engines": len(units), "cutoffs": len(cuts),
            "mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)), "sd": float(v.std())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/order03")
    ap.add_argument("--out", default="results/dsh/review_r1")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    longs = []
    for subset in ("FD001", "FD002"):
        lo = load_long(args.root, subset)
        if lo is not None:
            longs.append(lo)
    long = pd.concat(longs, ignore_index=True)

    ps = per_seed_metrics(long); ps.to_csv(out / "per_seed_metrics.csv", index=False)
    en = ensemble_metrics(long); en.to_csv(out / "ensemble_metrics.csv", index=False)

    spread = []
    for (subset, cell) in sorted({(r.subset, r.cell) for r in long.itertuples()}):
        s = conditional_cutoff_spread(long[(long.subset == subset) & (long.cell == cell)], cell, args.reps)
        s["subset"] = subset
        spread.append(s)
    pd.DataFrame(spread).to_csv(out / "cutoff_sensitivity.csv", index=False)

    rows = []
    for subset in ("FD001", "FD002"):
        cells = sorted(long[long.subset == subset].cell.unique())
        ref = "M6_tfA" if "M6_tfA" in cells else None
        if not ref:
            continue
        for c in cells:
            if c == ref:
                continue
            r = clustered_bootstrap_diff(long[long.subset == subset], c, ref, args.reps)
            if r:
                r["subset"] = subset
                rows.append(r)
    pd.DataFrame(rows).to_csv(out / "clustered_bootstrap_vs_ref.csv", index=False)

    print("per-seed rows:", len(ps), "| ensemble rows:", len(en), "| cells:", long.cell.nunique())
    print(ps.groupby("subset").rmse.describe().round(2).to_string())
    print()
    print(pd.DataFrame(spread).round(2).to_string(index=False))
    if rows:
        print()
        print(pd.DataFrame(rows)[["subset", "a", "b", "median_diff", "ci_low", "ci_high", "prob_a_better"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
