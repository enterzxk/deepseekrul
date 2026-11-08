"""Build RESULTS.csv / RESULTS.json from per-window prediction files.

Long format: one row per (protocol, subset, arm, seed) with the four metrics and the
paired contrast against the locked reference of that subset.  Nothing is fitted here.

Usage: python build_results.py --spec spec.json --out-prefix RESULTS
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPS = 5000
RNG_SEED = 99


def nasa(err):
    err = np.asarray(err, float)
    return np.where(err > 0, np.expm1(err / 10.0), np.expm1(-err / 13.0))


def load_group(base, subset, arm, seed, pattern, folds=(0, 1, 2, 3, 4)):
    blocks = []
    for fold in folds:
        p = Path(base) / subset / str(fold) / pattern.format(arm=arm, seed=seed)
        if not p.is_file():
            return None
        d = pd.read_csv(p)
        d["fold"] = fold
        blocks.append(d)
    d = pd.concat(blocks, ignore_index=True)
    assert np.isfinite(d.y_pred).all()
    d["err"] = d.y_pred - d.y_true
    return d


def engine_table(d):
    g = d.groupby("unit")
    return pd.DataFrame({
        "mse": g.apply(lambda x: float((x.err ** 2).mean()), include_groups=False),
        "mae": g.apply(lambda x: float(x.err.abs().mean()), include_groups=False),
        "nasa": g.apply(lambda x: float(nasa(x.err).mean()), include_groups=False),
        "bias": g.err.mean(),
        "n": g.size(),
    })


def pooled(t):
    return dict(rmse=float(np.sqrt(t.mse.mean())), mae=float(t.mae.mean()),
                nasa=float(t.nasa.mean()), bias=float(t.bias.mean()),
                nasa_sum=float(t.nasa.sum()), engines=int(len(t)), windows=int(t.n.sum()))


def paired(a, b):
    idx = a.index.intersection(b.index)
    av, bv = a.loc[idx, "mse"].to_numpy(), b.loc[idx, "mse"].to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(idx), (REPS, len(idx)))
    diffs = np.sqrt(av[draws].mean(1)) - np.sqrt(bv[draws].mean(1))
    return dict(delta=float(np.sqrt(av.mean()) - np.sqrt(bv.mean())),
                ci_lo=float(np.percentile(diffs, 2.5)), ci_hi=float(np.percentile(diffs, 97.5)),
                p_boot=float((diffs < 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-prefix", default="RESULTS")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())

    rows, detail = [], {}
    for group in spec["groups"]:
        protocol, base, pattern = group["protocol"], group["base"], group["pattern"]
        for subset, ref_arm in group["reference"].items():
            tables = {}
            for arm in group["arms"]:
                for seed in group["seeds"]:
                    d = load_group(base, subset, arm, seed, pattern)
                    if d is None:
                        continue
                    t = engine_table(d)
                    tables[(arm, seed)] = t
                    m = pooled(t)
                    row = dict(protocol=protocol, subset=subset, arm=arm, seed=seed,
                               reference=ref_arm, model_type=group.get("model_type", "single"),
                               **m)
                    if ref_arm in group["arms"] and (ref_arm, seed) in tables:
                        c = paired(t, tables[(ref_arm, seed)])
                        row.update(delta_rmse=c["delta"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"],
                                   gain=1.0 - m["rmse"] / pooled(tables[(ref_arm, seed)])["rmse"])
                    rows.append(row)
            detail[f"{protocol}|{subset}"] = {f"{a}|{s}": pooled(t) for (a, s), t in tables.items()}

    frame = pd.DataFrame(rows)
    frame.to_csv(f"{args.out_prefix}.csv", index=False)
    Path(f"{args.out_prefix}.json").write_text(json.dumps(dict(spec=spec, detail=detail), indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
