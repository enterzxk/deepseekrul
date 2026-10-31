"""Paired fold decomposition for FD001 (completes E19 for the second subset)."""
import glob, json
import numpy as np, pandas as pd
R = "/home/yndx/zxk/bearing-rul-pi/experiments"
REPS = 20000
def load(arm, s, f):
    sub = "e3_fd001" if f == 0 else "e18"
    pat = f"{R}/{sub}/{arm}_s{s}" + (f"/FD001/{arm}_s{s}" if f == 0 else f"_f{f}/FD001/{arm}_s{s}") + "/matched_cutoff_predictions.csv"
    g = glob.glob(pat)
    return pd.read_csv(g[0]).sort_values("unit_id").reset_index(drop=True) if g else None
rng = np.random.default_rng(9001)
out = {}
for a, b in (("BE","A"),("PE","A")):
    per, within = [], []
    for f in range(5):
        da, db = load(a, 2027, f), load(b, 2027, f)
        if da is None or db is None: continue
        ea = da.signed_error.to_numpy(float); eb = db.signed_error.to_numpy(float)
        per.append(float(np.sqrt((ea**2).mean()) - np.sqrt((eb**2).mean())))
        acc = np.empty(REPS)
        for r in range(REPS):
            i = rng.integers(0, len(ea), len(ea))
            acc[r] = np.sqrt((ea[i]**2).mean()) - np.sqrt((eb[i]**2).mean())
        within.append(float(np.std(acc, ddof=1)))
    between = float(np.std(per, ddof=1))
    out[f"{a}_minus_{b}"] = dict(per_fold=per, between_fold_sd=between,
                                 mean_within_se=float(np.mean(within)),
                                 ratio=between/float(np.mean(within)))
    print(f"{a}-{b}: per-fold {['%+.2f'%x for x in per]}")
    print(f"   between SD {between:.3f}  within SE {np.mean(within):.3f}  ratio {between/np.mean(within):.2f}")
json.dump(out, open("/home/yndx/zxk/bearing-rul-pi/results/dsh/e19/fd001_fold_decomposition.json","w"), indent=2)
print("wrote results/dsh/e19/fd001_fold_decomposition.json")
