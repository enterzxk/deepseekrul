"""Resampling-based deployment metric (DSH primary metric).

A single cutoff per engine makes RMSE depend heavily on which life stage the
cutoff lands in; rankings can flip between cutoff realisations. This script
reports the distribution over random single-cutoff draws instead:

  - common random numbers across cells (paired comparison),
  - mean / 5% / 95% of RMSE over draws,
  - P(cell A better than cell B) and median RMSE difference.

Usage: python scripts/dsh_resampled_metric.py --root experiments/order03 --out results/dsh
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd


def load_cut(root, tag, subset):
    fs = glob.glob(str(Path(root) / tag / subset / "*" / "protocol_c_predictions.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f) for f in fs])


def engine_errors(df):
    out = {}
    for unit, g in df.groupby("unit_id"):
        out[unit] = g.set_index("abs_cut")["signed_error"].to_dict()
    return out


def draw_rmse(eng, units, choice, seeds):
    errs = []
    for u in units:
        m = eng.get(u)
        if not m:
            continue
        cuts = list(m.keys())
        errs.append(m[cuts[choice[u] % len(cuts)]])
    return float(np.sqrt(np.mean(np.square(errs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/order03")
    ap.add_argument("--out", default="results/dsh")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--ref", default="M6_tfA")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, pairs = [], []

    for subset in ("FD001", "FD002"):
        cells = sorted({p.name for p in Path(args.root).glob("*") if p.is_dir()})
        data, engs = {}, {}
        for tag in cells:
            df = load_cut(args.root, tag, subset)
            if df is None or df.empty:
                continue
            data[tag] = df
            engs[tag] = engine_errors(df)
        if not data:
            continue
        units = sorted(set.intersection(*[set(e.keys()) for e in engs.values()]))
        if len(units) < 5:
            continue
        # common random numbers: one cutoff choice per (draw, engine)
        choices = [{u: int(rng.integers(0, 10**6)) for u in units} for _ in range(args.draws)]
        vals = {t: np.array([draw_rmse(engs[t], units, c, None) for c in choices]) for t in data}
        for t, v in vals.items():
            rows.append({"subset": subset, "cell": t, "engines": len(units),
                         "mean": float(v.mean()), "p05": float(np.percentile(v, 5)),
                         "p95": float(np.percentile(v, 95)), "std": float(v.std())})
        ref = args.ref if args.ref in vals else None
        if ref:
            for t, v in vals.items():
                if t == ref:
                    continue
                d = v - vals[ref]
                pairs.append({"subset": subset, "cell": t, "vs": ref,
                              "prob_better": float((d < 0).mean()),
                              "median_diff": float(np.median(d)),
                              "ci_low": float(np.percentile(d, 2.5)),
                              "ci_high": float(np.percentile(d, 97.5)),
                              "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})

    pd.DataFrame(rows).to_csv(out / "resampled_metric.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "resampled_pairs.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == "__main__":
    main()
