import json, re
from pathlib import Path
import pandas as pd

root = Path("experiments/punet_matrix")
rows = []
for sub in ["FD001","FD002","FD003","FD004"]:
    for d in sorted(root.glob(f"{sub}/f*_s*")):
        m = re.match(r"f(\d+)_s(\d+)", d.name)
        if not m: continue
        fold, seed = int(m.group(1)), int(m.group(2))
        rows.append(dict(subset=sub, fold=fold, seed=seed,
                         cqr=(d/"preds_cqr.csv").exists(),
                         point=(d/"preds_point.csv").exists(),
                         mcd=(d/"preds_mcd.csv").exists(),
                         js=(d/"e2e_cqr_none.json").exists()))
df = pd.DataFrame(rows)
print("training cells (any):", len(df))
print("  with quantile head :", int(df.cqr.sum()))
print("  with point head    :", int(df.point.sum()))
print("  with BOTH          :", int((df.cqr & df.point).sum()))
print("  with MC-dropout    :", int(df.mcd.sum()))
print("seeds present:", sorted(df.seed.unique()))
print("\nby seed (cells with both heads):")
both = df[df.cqr & df.point]
print(both.groupby("seed").size().to_string())
print("\nby subset x seed:")
print(both.pivot_table(index="subset", columns="seed", values="fold", aggfunc="count").fillna(0).astype(int).to_string())
print("\nmatched-coverage comparisons available: ", int(both.groupby(['subset','fold']).size().ge(1).sum() * 1), "cells (one per subset-fold-seed)")

# what punet_e2e (the 20-fold nominal-level study) has
e2e = Path("experiments/punet_e2e")
n = 0
for sub in ["FD001","FD002","FD003","FD004"]:
    for f in range(5):
        d = e2e/sub/f"f{f}_s42"
        if (d/"e2e_cqr_none.json").exists() and (d/"e2e_point_none.json").exists():
            n += 1
print("\npunet_e2e (nominal-level study) complete folds:", n)
