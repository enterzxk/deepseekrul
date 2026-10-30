"""C: interaction test with a permutation null.

Null: within each engine permute the stage labels (keeps condition structure,
destroys any genuine stage x condition association). If the observed interaction
RMS exceeds the null distribution, the state-dependent response is real.
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

def engine_blocks(subset, late_rul):
    blocks = []
    for u, g in load(subset).groupby("unit"):
        g = g.sort_values("cycle")
        n = len(g)
        rul = (n - g.cycle.to_numpy()).astype(float)
        stage = (rul < late_rul).astype(float)
        key = np.round(g[OPS].to_numpy(float), 2)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        k = len(uniq)
        if k < 3:
            continue
        x = g[USE].to_numpy(float)
        x = x - x.mean(0, keepdims=True)
        blocks.append((stage, np.eye(k)[inv], x, k))
    return blocks

def stat(blocks, rng=None, permute=False):
    vals = []
    for stage, oh, x, k in blocks:
        s = stage
        if permute and rng is not None:
            s = rng.permutation(stage)
        n = len(s)
        X = np.hstack([np.ones((n, 1)), s[:, None], oh[:, 1:], s[:, None] * oh[:, 1:]])
        beta, *_ = np.linalg.lstsq(X, x, rcond=None)
        vals.append(beta[X.shape[1] - (k - 1):])
    return float(np.sqrt(np.mean(np.square(np.vstack(vals)))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--late-rul", type=float, default=60)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--out", default="results/dsh/review_r1")
    args = ap.parse_args()
    blocks = engine_blocks(args.subset, args.late_rul)
    obs = stat(blocks)
    rng = np.random.default_rng(20261022)
    null = np.array([stat(blocks, rng, permute=True) for _ in range(args.perms)])
    p = float((null >= obs).mean())
    out = {"subset": args.subset, "late_rul": args.late_rul, "engines": len(blocks),
           "observed_interaction_rms": obs,
           "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
           "p_value": p, "above_null": bool(p < 0.05)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{args.out}/c_interaction_null.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
