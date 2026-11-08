"""Engine-clustered aggregation of per-window predictions (read-only).

Usage
-----
  python summarize_full.py --base runs --subsets FD002 FD004 --seed 101 \
      --arms M0 OPS FULL NORM --ref M0 [--out summary.json]

Every prediction file is one (subset, fold, arm, seed) CSV with columns
unit,end_cycle,y_true,y_pred.  Nothing is fitted here; the script only
aggregates and bootstraps.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPS = 5000
RNG_SEED = 99


def nasa(err):
    err = np.asarray(err, dtype=np.float64)
    return np.where(err > 0, np.expm1(err / 10.0), np.expm1(-err / 13.0))


def load_arm(base, subset, arm, seed, folds=(0, 1, 2, 3, 4), pattern="{arm}_{seed}.csv"):
    blocks = []
    for fold in folds:
        path = Path(base) / subset / str(fold) / pattern.format(arm=arm, seed=seed)
        if not path.is_file():
            return None
        frame = pd.read_csv(path)
        frame["fold"] = fold
        blocks.append(frame)
    data = pd.concat(blocks, ignore_index=True)
    assert np.isfinite(data.y_pred).all()
    err = data.y_pred.to_numpy() - data.y_true.to_numpy()
    data["err"] = err
    return data


def engine_stats(data):
    g = data.groupby("unit")
    return pd.DataFrame({
        "mse": g.apply(lambda d: float((d.err ** 2).mean()), include_groups=False),
        "mae": g.apply(lambda d: float(d.err.abs().mean()), include_groups=False),
        "nasa": g.apply(lambda d: float(nasa(d.err).mean()), include_groups=False),
        "bias": g.err.mean(),
        "n": g.size(),
        "true_life": g.y_true.max() + g.end_cycle.max(),
    })


def pooled(stats):
    return {
        "rmse": float(np.sqrt(stats.mse.mean())),
        "mae": float(stats.mae.mean()),
        "nasa": float(stats.nasa.mean()),
        "bias": float(stats.bias.mean()),
        "nasa_sum": float(stats.nasa.sum()),
        "engines": int(len(stats)),
        "windows": int(stats.n.sum()),
    }


def paired_delta(a, b, reps=REPS):
    """Delta RMSE = RMSE_a - RMSE_b with engine-clustered bootstrap."""
    idx = a.index.intersection(b.index)
    av, bv = a.loc[idx, "mse"].to_numpy(), b.loc[idx, "mse"].to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(idx), (reps, len(idx)))
    diffs = np.sqrt(av[draws].mean(1)) - np.sqrt(bv[draws].mean(1))
    point = float(np.sqrt(av.mean()) - np.sqrt(bv.mean()))
    return {"point": point,
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "frac_boot_lt0": float((diffs < 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs")
    ap.add_argument("--pattern", default="{arm}_{seed}.csv")
    ap.add_argument("--subsets", nargs="+", default=["FD002", "FD004"])
    ap.add_argument("--arms", nargs="+", default=["M0", "OPS", "FULL", "NORM"])
    ap.add_argument("--ref", default="M0")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="summary.json")
    args = ap.parse_args()

    out = {"base": str(Path(args.base).resolve()), "seed": args.seed,
           "reference": args.ref, "subsets": {}}
    for subset in args.subsets:
        stats, data = {}, {}
        for arm in args.arms:
            frame = load_arm(args.base, subset, arm, args.seed, pattern=args.pattern)
            if frame is None:
                continue
            data[arm] = frame
            stats[arm] = engine_stats(frame)
        entry = {"metrics": {a: pooled(s) for a, s in stats.items()},
                 "contrasts": {}}
        if args.ref in stats:
            for arm in stats:
                if arm == args.ref:
                    continue
                entry["contrasts"][arm + "-" + args.ref] = paired_delta(stats[arm], stats[args.ref])
        out["subsets"][subset] = entry
    Path(args.out).write_text(json.dumps(out, indent=2))
    for subset, entry in out["subsets"].items():
        print("==", subset)
        for arm, m in entry["metrics"].items():
            c = entry["contrasts"].get(arm + "-" + args.ref)
            extra = "" if c is None else "  d=%+.3f [%+.3f,%+.3f]" % (c["point"], c["ci95"][0], c["ci95"][1])
            print("  %-11s RMSE %7.3f  MAE %6.3f  NASA %9.2f  Bias %+7.3f  n=%d%s"
                  % (arm, m["rmse"], m["mae"], m["nasa"], m["bias"], m["engines"], extra))


if __name__ == "__main__":
    main()
