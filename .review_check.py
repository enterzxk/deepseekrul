import glob, json, os
import numpy as np, pandas as pd
R = "/home/yndx/zxk/bearing-rul-pi"

# ---------- 1. fold composition ----------
raw = pd.read_csv(f"{R}/data/CMAPSS/train_FD002.txt", sep=r"\s+", header=None)
raw = raw.rename(columns={0: "unit", 1: "cycle", 2: "op1", 3: "op2", 4: "op3"})
life = raw.groupby("unit").cycle.max()
units = np.sort(raw.unit.unique())
rng = np.random.default_rng(7 + 20261020)
sh = rng.permutation(units)
n = max(1, int(round(units.size * 0.2)))
print("n units", units.size, "fold size", n)
# condition: modal op triple
cond = raw.groupby("unit")[["op1", "op2", "op3"]].mean().round(1)
cond_key = cond.apply(lambda r: f"{r.op1:.0f}/{r.op2:.2f}/{r.op3:.1f}", axis=1)
combo = cond_key.value_counts()
print("\ndistinct condition signatures:", len(combo))
# map to 6 clusters by rounding op1 and op2
cond["c"] = (cond.op1.round(0).astype(int).astype(str) + "_" + (cond.op2 * 100).round(0).astype(int).astype(str))
print("engine-level condition classes:")
print(cond["c"].value_counts().to_string())

rows = []
for k in range(5):
    v = [int(x) for x in sh[k * n:(k + 1) * n]]
    l = life.loc[v]
    cm = cond.loc[v, "c"].value_counts(normalize=True)
    rows.append(dict(fold=k, n=len(v), life_mean=float(l.mean()), life_med=float(l.median()),
                     life_min=int(l.min()), life_max=int(l.max()),
                     short=int((l <= 140).sum()), long=int((l > 200).sum()),
                     **{f"c_{c}": float(cm.get(c, 0.0)) for c in sorted(cond["c"].unique())}))
tab = pd.DataFrame(rows)
print("\nfold composition (validation engines):")
print(tab.round(3).to_string(index=False))
print("\noverall life mean %.1f  sd %.1f" % (life.mean(), life.std()))
tab.to_csv(f"{R}/results/dsh/e16_fold_composition.csv", index=False)
print("wrote results/dsh/e16_fold_composition.csv")
