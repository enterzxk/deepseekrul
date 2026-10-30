"""Uncertainty of the resolution limit: bootstrap the sd of paired differences.

The single general quantity is sd of paired per-engine RMSE differences; MDE(N)
scales with it. Report a bootstrap CI over engines (resampling which engines are
in the study) and over pairs (resampling which method comparisons are considered).
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
cells = sorted(r)
units_all = sorted(set.intersection(*[set(v.index) for v in r.values()]))

rng = np.random.default_rng(20261022)
sd_engine, sd_pair = [], []
for _ in range(2000):
    take = list(rng.choice(units_all, len(units_all), replace=True))
    sds = []
    for a, b in itertools.combinations(cells, 2):
        d = (r[a].loc[take].to_numpy() - r[b].loc[take].to_numpy())
        sds.append(d.std(ddof=1))
    sd_engine.append(np.median(sds))
    sd_pair.append(np.median(rng.choice(sds, len(sds), replace=True)))

sd_engine, sd_pair = np.array(sd_engine), np.array(sd_pair)
n = len(units_all)
def mde(sd): return sd * np.sqrt(Z / n)
print(f"engines={n} pairs={len(cells)*(len(cells)-1)//2}")
print(f"sd median = {np.median(sd_engine):.2f}  95% CI over engines = [{np.percentile(sd_engine,2.5):.2f}, {np.percentile(sd_engine,97.5):.2f}]")
print(f"sd 95% CI over pairs   = [{np.percentile(sd_pair,2.5):.2f}, {np.percentile(sd_pair,97.5):.2f}]")
print(f"MDE at N={n}: {mde(np.median(sd_engine)):.2f}  CI=[{mde(np.percentile(sd_engine,2.5)):.2f}, {mde(np.percentile(sd_engine,97.5)):.2f}]")
for N in (100, 200, 500):
    s = np.median(sd_engine) * np.sqrt(Z / N)
    lo = np.percentile(sd_engine, 2.5) * np.sqrt(Z / N)
    hi = np.percentile(sd_engine, 97.5) * np.sqrt(Z / N)
    print(f"MDE at N={N}: {s:.2f}  CI=[{lo:.2f}, {hi:.2f}]")
pd.DataFrame({"sd_engine": sd_engine, "sd_pair": sd_pair}).to_csv("results/dsh/sd_bootstrap.csv", index=False)
