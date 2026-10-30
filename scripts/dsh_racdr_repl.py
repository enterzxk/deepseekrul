"""Does the RACDR vs Transformer gap replicate on the other splits?

The levelb result is post hoc, so the same comparison is checked on order03,
split8 and split9. Replication across independent engine splits is the minimum
bar before treating it as real.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

PAIRS = [("order03", "M4_racdr", "M0_raw_tf"), ("split8", "M4_racdr", "M0_raw_tf"),
         ("split9", "M4_racdr", "M0_raw_tf")]

def per_engine(root, tag):
    fs = glob.glob(f"experiments/{root}/{tag}/FD002/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    per = d.groupby(["unit_id", "seed"]).sq.mean().reset_index()
    return per.groupby("unit_id").sq.mean().pow(0.5)

for root, a_tag, b_tag in PAIRS:
    a, b = per_engine(root, a_tag), per_engine(root, b_tag)
    if a is None or b is None:
        print(f"{root}: missing ({a_tag}={a is not None}, {b_tag}={b is not None})")
        continue
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    rng = np.random.default_rng(20261022); dd = []
    for _ in range(3000):
        t = rng.integers(0, len(x), len(x))
        dd.append(np.sqrt((x[t]**2).mean()) - np.sqrt((y[t]**2).mean()))
    dd = np.asarray(dd)
    print(f"{root}: n={len(idx)} {a_tag}={np.sqrt((x**2).mean()):.2f} {b_tag}={np.sqrt((y**2).mean()):.2f} "
          f"diff={np.median(dd):+.2f} CI=[{np.percentile(dd,2.5):+.2f},{np.percentile(dd,97.5):+.2f}] P={float((dd<0).mean()):.3f}")
