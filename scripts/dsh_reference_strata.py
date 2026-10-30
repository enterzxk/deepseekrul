"""Stratified test: does the condition-reference disadvantage shrink with n?

A one-feature proxy stands in for the full model: predict RUL from the residual
magnitude under each reference, fit on training engines, evaluate on held-out
engines from split_seed=7, then stratify the RMSE difference by how many early
samples the engine had for its rarest operating condition.

B2 (offset contamination) predicts the disadvantage shrinks with n and vanishes
at n >= 5. B1 (signal compression) predicts it does not depend on n.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--out", default="results/dsh/review_r1")
    args = ap.parse_args()
    df = pd.read_csv(ROOT / f"train_{args.subset}.txt", sep=r"\s+", header=None,
                     names=["unit","cycle"] + OPS + SENSORS)
    units = np.sort(df.unit.unique())
    rng = np.random.default_rng(args.split_seed + 20261020)
    perm = rng.permutation(units)
    n_val = max(1, int(round(len(units) * 0.2)))
    val_u = set(int(x) for x in perm[:n_val])

    km = KMeans(n_clusters=6, random_state=args.split_seed, n_init=10).fit(
        df[df.unit.isin(perm[n_val:])][OPS].to_numpy(float))

    recs = []
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        x = g[USE].to_numpy(float); cyc = g.cycle.to_numpy(int)
        cl = km.predict(g[OPS].to_numpy(float))
        early = cyc <= 30
        g_ref = x[early].mean(0)
        refs, cnt = {}, {}
        for m in range(6):
            sel = early & (cl == m)
            cnt[m] = int(sel.sum())
            refs[m] = x[sel].mean(0) if sel.any() else g_ref
        n_min = min(cnt.values())
        rg = np.linalg.norm(x - g_ref, axis=1)
        rc = np.linalg.norm(x - np.stack([refs[m] for m in cl]), axis=1)
        rul = np.minimum(len(g) - cyc, 125).astype(float)
        for i in range(len(g)):
            recs.append({"unit": int(u), "n_min": n_min, "rul": rul[i],
                         "fg": rg[i], "fc": rc[i], "is_val": int(u in val_u)})
    d = pd.DataFrame(recs)
    tr = d[d.is_val == 0]; va = d[d.is_val == 1]

    def fit_eval(col):
        A = np.hstack([tr[col].to_numpy()[:, None], np.ones((len(tr), 1))])
        w = np.linalg.lstsq(A, tr.rul.to_numpy(), rcond=None)[0]
        B = np.hstack([va[col].to_numpy()[:, None], np.ones((len(va), 1))])
        return B @ w

    pg, pc = fit_eval("fg"), fit_eval("fc")
    va = va.assign(pg=pg, pc=pc, eg=(pg - va.rul.to_numpy()) ** 2, ec=(pc - va.rul.to_numpy()) ** 2)
    rows = []
    for lo, hi in [(0, 1), (2, 2), (3, 4), (5, 99)]:
        s = va[(va.n_min >= lo) & (va.n_min <= hi)]
        if len(s) < 20:
            continue
        rows.append({"n_min": f"{lo}-{hi}", "engines": int(s.unit.nunique()), "windows": len(s),
                     "rmse_global": float(np.sqrt(s.eg.mean())), "rmse_cond": float(np.sqrt(s.ec.mean())),
                     "cond_minus_global": float(np.sqrt(s.ec.mean()) - np.sqrt(s.eg.mean()))})
    out = {"subset": args.subset, "val_engines": len(val_u), "strata": rows}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"{args.out}/reference_strata.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
