"""How often is a method ranking determined by the data?

For every pair of cells evaluated on the same engines, compute the probability
that A beats B under random single-cutoff resampling. Pairs with P near 0.5 have
rankings that the evaluation cannot resolve; this quantifies how fragile the
"method A beats method B by X RMSE" statements typical of C-MAPSS papers are.
"""
import glob, itertools
import numpy as np
import pandas as pd

ROOT = "experiments/order03"
SUBSET = "FD002"

def load(tag):
    fs = glob.glob(f"{ROOT}/{tag}/{SUBSET}/*/protocol_c_predictions.csv")
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    return {u: g.set_index("abs_cut").signed_error.to_dict() for u, g in d.groupby("unit_id")}

tags = sorted({p.split("/")[2] for p in glob.glob(f"{ROOT}/*/{SUBSET}/*/protocol_c_predictions.csv")})
data = {t: load(t) for t in tags}
data = {t: v for t, v in data.items() if v}
units_all = set.intersection(*[set(v) for v in data.values()])
print(f"cells={len(data)} engines={len(units_all)}")

rng = np.random.default_rng(20261022)
B = 1000
# common draws so comparisons are paired
cuts = {u: list(sorted(next(iter(data.values()))[u].keys())) for u in units_all}
draws = [{u: cuts[u][rng.integers(0, len(cuts[u]))] for u in units_all} for _ in range(B)]
vals = {}
for t, d in data.items():
    arr = np.empty((B, len(units_all)))
    ul = sorted(units_all)
    for b, dr in enumerate(draws):
        arr[b] = [d[u].get(dr[u], np.nan) for u in ul]
    vals[t] = arr

rows = []
for a, b in itertools.combinations(sorted(vals), 2):
    ea, eb = vals[a], vals[b]
    m = ~(np.isnan(ea) | np.isnan(eb))
    ra = np.sqrt(np.nansum(np.where(m, ea**2, 0), axis=1) / m.sum(axis=1))
    rb = np.sqrt(np.nansum(np.where(m, eb**2, 0), axis=1) / m.sum(axis=1))
    d = ra - rb
    rows.append({"a": a, "b": b, "p_a_better": float((d < 0).mean()),
                 "median_diff": float(np.median(d))})
df = pd.DataFrame(rows)
df.to_csv("results/dsh/ranking_resolvability.csv", index=False)
und = df[(df.p_a_better > 0.25) & (df.p_a_better < 0.75)]
print(f"pairs={len(df)}  undetermined (0.25<P<0.75)={len(und)}  ({100*len(und)/len(df):.1f}%)")
print(f"pairs with P in [0.4,0.6] (coin flip)={len(df[(df.p_a_better>=0.4)&(df.p_a_better<=0.6)])}")
print()
print(df.sort_values("p_a_better").head(8).to_string(index=False))
