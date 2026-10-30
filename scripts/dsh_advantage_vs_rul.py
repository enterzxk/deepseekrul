"""Does the per-engine method advantage depend on engine lifespan?

The method x split interaction accounts for ~48% of the variance, i.e. the ranking
depends on which engines are in the validation set. A concrete mechanism: the
advantage of one method over another varies systematically with engine lifespan,
so any split that over-samples short or long engines will rank differently.

Per engine and per split: RMSE under each method, plus the engine's lifespan.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOTS = {"split7": "experiments/order03", "split8": "experiments/split8", "split9": "experiments/split9"}
OUT = Path("results/dsh/review_r1")

def per_engine(root, tag, subset="FD002"):
    fs = glob.glob(f"{root}/{tag}/{subset}/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    per = d.groupby(["unit_id", "model_seed"]).sq.mean().reset_index()
    r = per.groupby("unit_id").sq.mean().pow(0.5)
    life = d.groupby("unit_id").apply(lambda g: g["cutoff_true"].max() if "cutoff_true" in g else np.nan) if False else None
    return r.to_frame("rmse")

rows = []
for name, root in ROOTS.items():
    a, b = per_engine(root, "M1_global"), per_engine(root, "M6_tfA")
    if a is None or b is None:
        continue
    m = a.join(b, lsuffix="_m1", rsuffix="_m6", how="inner")
    # lifespan proxy: number of evaluations is constant; use target at the matched cutoff
    fs = glob.glob(f"{root}/M1_global/FD002/*/matched_cutoff_predictions.csv")
    d = pd.concat([pd.read_csv(f) for f in fs])
    proxy = d.groupby("unit_id").y_true.max()          # RUL at the sampled cutoff
    span = d.groupby("unit_id").y_true.count()
    m["split"] = name
    m["rul_at_cutoff"] = proxy.reindex(m.index)
    rows.append(m.reset_index())
df = pd.concat(rows, ignore_index=True)
df["diff"] = df.rmse_m1 - df.rmse_m6      # negative = M1 better
df.to_csv(OUT / "per_engine_advantage.csv", index=False)

x = df.rul_at_cutoff.to_numpy(float); y = df["diff"].to_numpy(float)
ok = np.isfinite(x) & np.isfinite(y)
slope, intercept = np.polyfit(x[ok], y[ok], 1)
corr = float(np.corrcoef(x[ok], y[ok])[0, 1])
# stratified by RUL at cutoff
bands = [(0, 40), (40, 80), (80, 126)]
strat = []
for lo, hi in bands:
    s = df[(df.rul_at_cutoff >= lo) & (df.rul_at_cutoff < hi)]
    if len(s) >= 5:
        strat.append({"band": f"{lo}-{hi}", "n": int(len(s)),
                      "mean_diff": float(s["diff"].mean()),
                      "pct_M1_better": float((s["diff"] < 0).mean())})
out = {"n": int(ok.sum()), "slope": float(slope), "corr": corr,
       "mean_diff_overall": float(np.nanmean(y)), "strata": strat}
json.dump(out, open(OUT / "advantage_vs_rul.json", "w"), indent=2)
print(json.dumps(out, indent=2))
