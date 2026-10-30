"""Power analysis (corrected): paired per-engine RMSE differences.

The earlier version used per-engine MSE, whose heavy tail (sd ~150) makes the
normal-approximation power formula meaningless. This version uses per-engine RMSE
(sqrt of mean squared error across evaluated cutoffs), a bounded, better-behaved
paired statistic.
"""
import glob, itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT, SUBSET = "experiments/order03", "FD002"
Z = (norm.ppf(0.975) + norm.ppf(0.80)) ** 2

def per_engine_rmse(tag):
    fs = glob.glob(f"{ROOT}/{tag}/{SUBSET}/*/protocol_c_predictions.csv")
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return d.groupby("unit_id").signed_error.apply(lambda s: float(np.sqrt((s ** 2).mean())))

tags = sorted({p.split("/")[2] for p in glob.glob(f"{ROOT}/*/{SUBSET}/*/protocol_c_predictions.csv")})
r = {t: per_engine_rmse(t) for t in tags}
r = {t: v for t, v in r.items() if v is not None}

rows = []
for a, b in itertools.combinations(sorted(r), 2):
    idx = r[a].index.intersection(r[b].index)
    d = (r[a].loc[idx] - r[b].loc[idx]).to_numpy()
    mean, sd, n = float(d.mean()), float(d.std(ddof=1)), len(d)
    need = (Z * sd / abs(mean)) ** 2 if abs(mean) > 1e-9 else np.inf
    rows.append({"a": a, "b": b, "gap": mean, "sd": sd, "n_now": n,
                 "n_needed": need, "resolvable": n >= need})
df = pd.DataFrame(rows)
df.to_csv("results/dsh/power_analysis.csv", index=False)
fin = df[np.isfinite(df.n_needed)]
print(f"pairs={len(df)}  resolvable now={int(df.resolvable.sum())}")
print(f"median sd of paired per-engine RMSE diff = {df.sd.median():.2f}")
print(f"median engines needed = {fin.n_needed.median():.0f}")
print(f"pairs needing <=52: {int((fin.n_needed<=52).sum())} / {len(fin)}")
print()
print(df.reindex(df.gap.abs().sort_values(ascending=False).index).head(8)[["a","b","gap","sd","n_needed"]].round(2).to_string(index=False))
