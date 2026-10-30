"""C: is the condition->sensor response state-dependent?

Static calibration removes a per-condition healthy mean estimated from early
life. State calibration additionally conditions that mean on the *observable*
life-stage bin (current cycle), so it can absorb a response that drifts with
degradation.

Cross-fitting: means and the ridge are fitted on training engines only and
evaluated on held-out validation engines from split_seed=7. Online inputs are
{condition, current cycle, sensors}; no future life, no labels at test time.

If (b) beats (a) on held-out engines, the gap is real; otherwise route A dies.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path("data/CMAPSS")
SENSORS = [f"s{i}" for i in range(1, 22)]
USE = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
OPS = ["op_1","op_2","op_3"]
WINDOW, STRIDE, CAP = 50, 10, 125

def load():
    df = pd.read_csv(ROOT / "train_FD002.txt", sep=r"\s+", header=None,
                     names=["unit","cycle"] + OPS + SENSORS)
    return df

def rul_of(g, cap=CAP):
    n = g.cycle.max()
    return np.minimum(g.cycle.to_numpy(), n - g.cycle.to_numpy() + cap - cap).astype(float) if False else np.minimum(n - g.cycle.to_numpy(), cap).astype(float)

def build(df, units, kmeans, mu_static, mu_state, stage_edges):
    Xa, Xb, y = [], [], []
    for u in units:
        g = df[df.unit == u].sort_values("cycle")
        x = g[USE].to_numpy(float)
        cyc = g.cycle.to_numpy(int)
        cl = kmeans.predict(g[OPS].to_numpy(float))
        r = np.minimum(cyc.max() - cyc, CAP).astype(float)
        for end in range(WINDOW, len(g) + 1, STRIDE):
            t = end - 1
            w = x[end - WINDOW:end]
            m = cl[t]
            b = np.searchsorted(stage_edges, cyc[t], side="right")
            b = min(b, len(stage_edges))
            ra = (w - mu_static[m]).mean(0)
            rb = (w - mu_state[(m, b)]).mean(0)
            Xa.append(ra); Xb.append(rb); y.append(r[t])
    return np.array(Xa), np.array(Xb), np.array(y)

def ridge_fit(X, y, lam=1.0):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    return np.linalg.solve(A, X1.T @ y)

def rmse(p, y):
    return float(np.sqrt(((p - y) ** 2).mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--out", default="results/dsh/review_r1")
    args = ap.parse_args()
    df = load()
    units = np.sort(df.unit.unique())
    rng = np.random.default_rng(args.split_seed + 20261020)
    perm = rng.permutation(units)
    n_val = max(1, int(round(len(units) * 0.2)))
    val_u = sorted(int(x) for x in perm[:n_val]); tr_u = sorted(int(x) for x in perm[n_val:])

    kmeans = KMeans(n_clusters=6, random_state=args.split_seed, n_init=10).fit(df[df.unit.isin(tr_u)][OPS].to_numpy(float))
    tr = df[df.unit.isin(tr_u)].copy()
    tr["cl"] = kmeans.predict(tr[OPS].to_numpy(float))
    # static: per-condition healthy mean from early life of training engines only
    early = tr[tr.cycle <= 30]
    mu_static = {m: early[early.cl == m][USE].to_numpy(float).mean(0) for m in range(6)}
    # state: per (condition, observable cycle bin) mean from training engines
    allc = tr.cycle.to_numpy()
    stage_edges = np.quantile(allc, [0.2, 0.4, 0.6, 0.8])
    tr["b"] = np.searchsorted(stage_edges, allc, side="right")
    mu_state = {}
    for m in range(6):
        for b in range(len(stage_edges) + 1):
            sel = tr[(tr.cl == m) & (tr.b == b)]
            mu_state[(m, b)] = sel[USE].to_numpy(float).mean(0) if len(sel) else mu_static[m]
    Xa_tr, Xb_tr, y_tr = build(df, tr_u, kmeans, mu_static, mu_state, stage_edges)
    Xa_va, Xb_va, y_va = build(df, val_u, kmeans, mu_static, mu_state, stage_edges)
    wa, wb = ridge_fit(Xa_tr, y_tr), ridge_fit(Xb_tr, y_tr)
    pa = np.hstack([Xa_va, np.ones((len(Xa_va), 1))]) @ wa
    pb = np.hstack([Xb_va, np.ones((len(Xb_va), 1))]) @ wb
    ra, rb = rmse(pa, y_va), rmse(pb, y_va)
    # engine-level paired difference
    units_va = np.repeat(val_u, [len(range(WINDOW, len(df[df.unit == u]) + 1, STRIDE)) for u in val_u])
    d = []
    for u in val_u:
        m = units_va == u
        if m.sum():
            d.append(np.sqrt(((pa[m] - y_va[m]) ** 2).mean()) - np.sqrt(((pb[m] - y_va[m]) ** 2).mean()))
    d = np.array(d)
    out = {"split_seed": args.split_seed, "train_engines": len(tr_u), "val_engines": len(val_u),
           "windows_val": int(len(y_va)), "rmse_static": ra, "rmse_state": rb,
           "delta": ra - rb, "engine_mean_delta": float(d.mean()), "engine_sd": float(d.std(ddof=1)),
           "engines_state_better": int((d > 0).sum())}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{args.out}/crossfit_c.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
