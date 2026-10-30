"""Validate the resolution theory: fewer engines -> more unresolvable rankings.

Prediction from the MDE table: at 20 engines the minimum detectable effect is
~2.66 RMSE, so a larger fraction of method pairs must be undetermined compared
with the 52-engine case (~45%).
"""
import glob, itertools
import numpy as np
import pandas as pd

def run(ROOT, SUBSET, label):
    def load(tag):
        fs = glob.glob(f"{ROOT}/{tag}/{SUBSET}/*/protocol_c_predictions.csv")
        if not fs: return None
        d = pd.concat([pd.read_csv(f) for f in fs])
        return {u: g.set_index("abs_cut").signed_error.to_dict() for u, g in d.groupby("unit_id")}
    tags = sorted({p.split("/")[2] for p in glob.glob(f"{ROOT}/*/{SUBSET}/*/protocol_c_predictions.csv")})
    data = {t: load(t) for t in tags}
    data = {t: v for t, v in data.items() if v}
    if len(data) < 3: 
        print(f"{label}: only {len(data)} cells, skip"); return
    units = sorted(set.intersection(*[set(v) for v in data.values()]))
    rng = np.random.default_rng(20261022); B = 600
    ref = next(iter(data.values()))
    cuts = {u: sorted(ref[u].keys()) for u in units}
    draws = [{u: cuts[u][rng.integers(0, len(cuts[u]))] for u in units} for _ in range(B)]
    vals = {}
    for t, d in data.items():
        arr = np.empty((B, len(units)))
        for b, dr in enumerate(draws):
            arr[b] = [d[u].get(dr[u], np.nan) for u in units]
        vals[t] = arr
    ps = []
    for a, b in itertools.combinations(sorted(vals), 2):
        ea, eb = vals[a], vals[b]; m = ~(np.isnan(ea) | np.isnan(eb))
        ra = np.sqrt(np.nansum(np.where(m, ea**2, 0), 1) / m.sum(1))
        rb = np.sqrt(np.nansum(np.where(m, eb**2, 0), 1) / m.sum(1))
        ps.append(float(((ra - rb) < 0).mean()))
    ps = np.array(ps)
    und = ((ps > 0.25) & (ps < 0.75)).mean() * 100
    print(f"{label}: cells={len(data)} engines={len(units)} pairs={len(ps)} undetermined={und:.1f}% coin_flip={(((ps>=0.4)&(ps<=0.6)).mean()*100):.1f}%")

run("experiments/order03", "FD002", "FD002 (52 engines)")
run("experiments/order03", "FD001", "FD001 (20 engines)")
