"""Resolution limits of the C-MAPSS evaluation as used here.

Minimum detectable effect (MDE) at 80% power, alpha=0.05, paired design:
    MDE(N) = sd * sqrt((z_{0.975}+z_{0.80})^2 / N)
Using the empirical sd of paired per-engine RMSE differences from this study.
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
sds = []
for a, b in itertools.combinations(sorted(r), 2):
    idx = r[a].index.intersection(r[b].index)
    sds.append(float((r[a].loc[idx] - r[b].loc[idx]).std(ddof=1)))
sds = np.array(sds)
sd_med, sd_q1, sd_q3 = np.median(sds), np.percentile(sds, 25), np.percentile(sds, 75)
print(f"paired per-engine RMSE diff sd: median={sd_med:.2f} q1={sd_q1:.2f} q3={sd_q3:.2f}")

rows = []
for N in (20, 52, 100, 200, 500, 1000, 2000):
    rows.append({"engines": N,
                 "MDE_sd_median": sd_med * np.sqrt(Z / N),
                 "MDE_sd_q1": sd_q1 * np.sqrt(Z / N),
                 "MDE_sd_q3": sd_q3 * np.sqrt(Z / N)})
df = pd.DataFrame(rows)
print()
print(df.round(2).to_string(index=False))
df.to_csv("results/dsh/resolution_limits.csv", index=False)
