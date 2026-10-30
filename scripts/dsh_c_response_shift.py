"""C (diagnostic): does the condition->sensor response change with life stage?

For each training engine that contains both an early segment (RUL > 100) and a
late segment (RUL < 30), fit x_s ~ one-hot(condition) separately in each segment
and compare coefficients. Engine-level paired test on the coefficient change.

RUL is used only for this *diagnostic* split; the predictive use (if any) would
have to rely on an observable proxy.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/CMAPSS")
SENSORS = [f"s{i}" for i in range(1, 22)]
USE = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
OPS = ["op_1","op_2","op_3"]

def load(subset="FD002"):
    return pd.read_csv(ROOT / f"train_{subset}.txt", sep=r"\s+", header=None,
                       names=["unit","cycle"] + OPS + SENSORS)

def fit_coef(x, cond_onehot, lam=1e-2):
    A = cond_onehot.T @ cond_onehot + lam * np.eye(cond_onehot.shape[1])
    return np.linalg.solve(A, cond_onehot.T @ x)  # (k, sensors)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--early", type=float, default=100)
    ap.add_argument("--late", type=float, default=30)
    ap.add_argument("--out", default="results/dsh/review_r1")
    args = ap.parse_args()
    df = load(args.subset)
    deltas, units = [], []
    skipped = 0
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        n = len(g)
        rul = (n - g.cycle.to_numpy()).astype(float)
        early = rul > args.early
        late = rul < args.late
        if early.sum() < 20 or late.sum() < 10:
            skipped += 1
            continue
        cl = g[OPS].to_numpy(float)
        # discretise condition by rounding to 2 decimals to get stable one-hot
        key = np.round(cl, 2)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        if len(uniq) < 2:
            skipped += 1
            continue
        onehot = np.eye(len(uniq))[inv]
        x = g[USE].to_numpy(float)
        be = fit_coef(x[early], onehot[early])
        bl = fit_coef(x[late], onehot[late])
        # align on shared conditions; if condition sets differ, use the first
        d = bl.mean(0) - be.mean(0)
        deltas.append(d); units.append(int(u))
    D = np.array(deltas)
    n = len(D)
    mean = D.mean(0)
    sd = D.std(0, ddof=1)
    se = sd / np.sqrt(n)
    t = mean / np.where(se > 0, se, np.inf)
    rng = np.random.default_rng(20261022)
    boot = np.array([D[rng.integers(0, n, n)].mean(0) for _ in range(2000)])
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    rows = [{"sensor": USE[i], "delta": float(mean[i]), "t": float(t[i]),
             "lo": float(lo[i]), "hi": float(hi[i]),
             "excludes_zero": bool(lo[i] > 0 or hi[i] < 0)} for i in range(len(USE))]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{args.out}/c_response_shift.csv", index=False)
    n_sig = sum(r["excludes_zero"] for r in rows)
    summary = {"engines_used": n, "engines_skipped": skipped,
               "sensors_with_ci_excluding_zero": n_sig, "of": len(USE),
               "mean_abs_delta": float(np.abs(mean).mean())}
    json.dump(summary, open(f"{args.out}/c_response_shift_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()
