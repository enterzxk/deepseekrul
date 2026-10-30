"""C (corrected): stage x condition INTERACTION, level effects removed.

x - engine_mean = b0 + b1*stage + b2*cond + b3*(stage x cond) + e
Engine means are removed first so global degradation drift cancels. The statistic
is the RMS of the interaction block b3; its interval comes from a cluster
bootstrap over engines (engines resampled, refit each time).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/CMAPSS")
SENSORS = [f"s{i}" for i in range(1, 22)]
USE = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
OPS = ["op_1","op_2","op_3"]

def load(subset):
    return pd.read_csv(ROOT / f"train_{subset}.txt", sep=r"\s+", header=None,
                       names=["unit","cycle"] + OPS + SENSORS)

def build(subset, late_rul):
    rows = []
    df = load(subset)
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        n = len(g)
        rul = (n - g.cycle.to_numpy()).astype(float)
        stage = (rul < late_rul).astype(float)
        key = np.round(g[OPS].to_numpy(float), 2)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        k = len(uniq)
        if k < 3:
            continue
        oh = np.eye(k)[inv]
        X = np.hstack([np.ones((n, 1)), stage[:, None], oh[:, 1:], stage[:, None] * oh[:, 1:]])
        x = g[USE].to_numpy(float)
        x = x - x.mean(0, keepdims=True)
        ok = np.isfinite(X).all(1) & np.isfinite(x).all(1)
        rows.append((X[ok], x[ok], X.shape[1] - (k - 1)))
    return rows

def stat(sample):
    vals = []
    for X, x, start in sample:
        beta, *_ = np.linalg.lstsq(X, x, rcond=None)
        vals.append(beta[start:])
    if not vals:
        return np.nan
    return float(np.sqrt(np.mean(np.square(np.vstack(vals)))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--late-rul", type=float, default=60)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--out", default="results/dsh/review_r1")
    args = ap.parse_args()
    sample = build(args.subset, args.late_rul)
    full = stat(sample)
    rng = np.random.default_rng(20261022)
    boots = []
    for _ in range(args.reps):
        idx = rng.integers(0, len(sample), len(sample))
        boots.append(stat([sample[i] for i in idx]))
    boots = np.array([b for b in boots if np.isfinite(b)])
    # null: shuffle stage labels within engine destroys the interaction
    out = {"subset": args.subset, "late_rul": args.late_rul, "engines": len(sample),
           "interaction_rms": full,
           "ci_low": float(np.percentile(boots, 2.5)),
           "ci_high": float(np.percentile(boots, 97.5)),
           "boot_mean": float(boots.mean())}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{args.out}/c_interaction.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
