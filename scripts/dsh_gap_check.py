"""Is the M3 vs M1 gap (cond vs global reference) real under correct clustering?

All four attempted explanations were aimed at a 3.77 RMSE difference produced by
the same evaluation code that later turned out to have seed-overwrite bugs. This
checks the difference itself: engine-clustered paired bootstrap on per-engine
RMSE, using matched-cutoff predictions, keyed by (subset, unit, seed).
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = "experiments/order03"

def per_engine(tag, subset="FD002"):
    fs = glob.glob(f"{ROOT}/{tag}/{subset}/*/matched_cutoff_predictions.csv")
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f).assign(model_seed=int(Path(f).parts[-2].split("_s")[-1])) for f in fs])
    d["sq"] = d.signed_error.astype(float) ** 2
    # average squared error across seeds first, then per-engine RMSE
    per = d.groupby(["unit_id", "model_seed"]).sq.mean().reset_index()
    per = per.groupby("unit_id").sq.mean().pow(0.5)
    return per

def boot(a, b, reps=5000, seed=20261022):
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(reps):
        t = rng.integers(0, len(idx), len(idx))
        diffs.append(np.sqrt((x[t] ** 2).mean()) - np.sqrt((y[t] ** 2).mean()))
    d = np.asarray(diffs)
    return {"engines": int(len(idx)),
            "rmse_a": float(np.sqrt((x ** 2).mean())), "rmse_b": float(np.sqrt((y ** 2).mean())),
            "median_diff": float(np.median(d)), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "prob_a_better": float((d < 0).mean()),
            "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

out = {}
for sub in ("FD002", "FD001"):
    a, b = per_engine("M3_cond", sub), per_engine("M1_global", sub)
    if a is not None and b is not None:
        r = boot(a, b); r["subset"] = sub; r["pair"] = "M3_cond vs M1_global"
        out[f"M3vsM1_{sub}"] = r
        print(json.dumps(r, indent=2))
Path("results/dsh/review_r1").mkdir(parents=True, exist_ok=True)
json.dump(out, open("results/dsh/review_r1/gap_check.json", "w"), indent=2)
