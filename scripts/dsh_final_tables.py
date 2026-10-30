"""Official result tables for the RUL study (DSH).

Computes, per (cell, subset):
  - matched-cutoff RMSE (primary deployment metric)
  - Protocol B: pooled (samples merged) and macro (per-cutoff RMSE then mean)
  - Protocol C: pooled and macro
  - safety trio: NASA, late fraction, severe-late count
  - engine-clustered paired bootstrap vs a reference cell

Run: python scripts/dsh_final_tables.py --root experiments/order03 --out results/dsh
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

CUT_B = ["0.5", "0.7", "0.9"]


def rmse(e):
    return float(np.sqrt(np.mean(np.asarray(e, dtype=float) ** 2)))


def safety(e):
    e = np.asarray(e, dtype=float)
    return {
        "nasa": float(np.sum(np.where(e < 0, np.expm1(-e / 13), np.expm1(e / 10)))),
        "late_fraction": float((e > 0).mean()),
        "severe_late": int((e > 10).sum()),
        "mae": float(np.abs(e).mean()),
    }


def cell_dirs(root, tag, subset):
    return sorted(glob.glob(str(Path(root) / tag / subset / "*")))


def load_matched(root, tag, subset):
    frames = []
    for d in cell_dirs(root, tag, subset):
        p = Path(d) / "matched_cutoff_predictions.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames) if frames else None


def load_protocol(root, tag, subset, kind):
    frames = []
    for d in cell_dirs(root, tag, subset):
        p = Path(d) / ("pseudo_test_predictions.csv" if kind == "B" else "protocol_c_predictions.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    return pd.concat(frames) if frames else None


def summarise(df, group_col):
    if df is None or df.empty or group_col not in df.columns:
        return None, None
    pooled = rmse(df["signed_error"])
    macro = float(np.mean([rmse(g["signed_error"]) for _, g in df.groupby(group_col)]))
    return pooled, macro


def bootstrap(a, b, reps=20000, seed=20261022):
    """Engine-clustered paired bootstrap on per-engine mean squared error."""
    m = a.merge(b, on="unit_id", suffixes=("_a", "_b"))
    if m.empty:
        return None
    ea = m.groupby("unit_id")["signed_error_a"].apply(lambda s: (s ** 2).mean())
    eb = m.groupby("unit_id")["signed_error_b"].apply(lambda s: (s ** 2).mean())
    idx = ea.index.intersection(eb.index)
    ea, eb = ea.loc[idx].to_numpy(), eb.loc[idx].to_numpy()
    rng = np.random.default_rng(seed)
    dr = []
    for _ in range(reps):
        take = rng.integers(0, len(idx), len(idx))
        dr.append(np.sqrt(ea[take].mean()) - np.sqrt(eb[take].mean()))
    dr = np.asarray(dr)
    return {
        "engines": int(len(idx)),
        "mean_diff": float(dr.mean()),
        "ci_low": float(np.percentile(dr, 2.5)),
        "ci_high": float(np.percentile(dr, 97.5)),
        "prob_a_better": float((dr < 0).mean()),
        "excludes_zero": bool(np.percentile(dr, 2.5) > 0 or np.percentile(dr, 97.5) < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/order03")
    ap.add_argument("--out", default="results/dsh")
    ap.add_argument("--ref", default="M6_tfA")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows, boots = [], []
    cells = sorted({p.name for p in Path(args.root).glob("*") if p.is_dir()})
    for tag in cells:
        for subset in ("FD001", "FD002"):
            if not cell_dirs(args.root, tag, subset):
                continue
            m = load_matched(args.root, tag, subset)
            if m is None:
                continue
            b = load_protocol(args.root, tag, subset, "B")
            c = load_protocol(args.root, tag, subset, "C")
            row = {"cell": tag, "subset": subset, "n_seeds": m["model_seed"].nunique() if "model_seed" in m else 0,
                   "matched_rmse": rmse(m["signed_error"])}
            row.update({f"matched_{k}": v for k, v in safety(m["signed_error"]).items()})
            for name, df, col in (("B", b, "cutoff"), ("C", c, "abs_cut")):
                pooled, macro = summarise(df, col)
                row[f"{name}_pooled"] = pooled
                row[f"{name}_macro"] = macro
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(["subset", "cell"])
    table.to_csv(out / "final_tables.csv", index=False)
    print(table.to_string(index=False))

    ref = load_matched(args.root, args.ref, "FD002")
    if ref is not None:
        for tag in cells:
            if tag == args.ref:
                continue
            cur = load_matched(args.root, tag, "FD002")
            if cur is None:
                continue
            r = bootstrap(cur, ref)
            if r:
                boots.append({"cell": tag, "ref": args.ref, **r})
        if boots:
            pd.DataFrame(boots).to_csv(out / "bootstrap_vs_ref.csv", index=False)
            print()
            print(pd.DataFrame(boots).to_string(index=False))


if __name__ == "__main__":
    main()
