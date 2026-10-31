import glob, json
import numpy as np, pandas as pd
R = "/home/yndx/zxk/bearing-rul-pi/experiments"
ARMS = {"A": ("e10", True), "tf": ("e11", False), "gru": ("e11", False),
        "bilstm": ("e11", False), "tcn": ("e11", False), "cnn1d": ("e11", False)}
FOLDS = list(range(5)); SEED = 2027

def load(arm, f):
    exp, f0 = ARMS[arm]
    pat = (f"{R}/e3/{arm}_s{SEED}/FD002/{arm}_s{SEED}/matched_cutoff_predictions.csv" if f == 0 and f0
           else f"{R}/{exp}/{arm}_s{SEED}_f{f}/FD002/{arm}_s{SEED}/matched_cutoff_predictions.csv")
    g = glob.glob(pat)
    return pd.read_csv(g[0]).sort_values("unit_id").reset_index(drop=True) if g else None

err = {}
for a in ARMS:
    per = [load(a, f) for f in FOLDS]
    if all(d is not None for d in per):
        err[a] = [d.signed_error.to_numpy(float) for d in per]
print("arms with all 5 folds:", list(err))

def rmse_concat(fold_errs, picks):
    return np.sqrt(np.mean(np.concatenate([e[p] for e, p in zip(fold_errs, picks)]) ** 2))

def boot(a, b, reps=100000, rng_seed=9000):
    ea, eb = err[a], err[b]
    pt = rmse_concat(eb, [slice(None)] * 5) - rmse_concat(ea, [slice(None)] * 5)
    rng = np.random.default_rng(rng_seed)
    acc = np.empty(reps)
    for r in range(reps):
        picks = [rng.integers(0, len(e), len(e)) for e in ea]
        acc[r] = rmse_concat(eb, picks) - rmse_concat(ea, picks)
    return float(pt), np.percentile(acc, [5, 95]), np.percentile(acc, [2.5, 97.5])

DELTA = 1.0
PAIRS = [("A", "tf"), ("tf", "gru"), ("tf", "bilstm"), ("gru", "bilstm"), ("tcn", "cnn1d"), ("A", "gru")]
out = {"margin": DELTA, "reps": 100000, "rng": 9000, "seed": SEED,
       "note": "clustered bootstrap resampling engines within each fold; delta = RMSE(first) - RMSE(second)",
       "rows": []}
print(f"\n{'contrast':14s} {'delta':>8s} {'90% CI':>19s} {'95% CI':>19s}  TOST(margin={DELTA})")
for a, b in PAIRS:
    pt, ci90, ci95 = boot(a, b)
    tost = bool(ci90[0] > -DELTA and ci90[1] < DELTA)
    out["rows"].append(dict(a=a, b=b, delta=pt, ci90=[float(x) for x in ci90],
                            ci95=[float(x) for x in ci95], tost_equivalent=tost))
    print(f"{a+' - '+b:14s} {pt:+8.3f}  [{ci90[0]:+7.3f},{ci90[1]:+7.3f}]  "
          f"[{ci95[0]:+7.3f},{ci95[1]:+7.3f}]  {'EQUIVALENT' if tost else 'not established'}")
json.dump(out, open(f"{R}/../results/dsh/e16_tost.json", "w"), indent=2)
print("\nwrote results/dsh/e16_tost.json")
