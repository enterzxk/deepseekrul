import glob, numpy as np, pandas as pd, collections
R = "/home/yndx/zxk/bearing-rul-pi/experiments"
def load(arm, s, f):
    sub = "e3_fd001" if f == 0 else "e18"
    pat = f"{R}/{sub}/{arm}_s{s}" + (f"/FD001/{arm}_s{s}" if f == 0 else f"_f{f}/FD001/{arm}_s{s}") + "/matched_cutoff_predictions.csv"
    g = glob.glob(pat)
    return pd.read_csv(g[0]).sort_values("unit_id").reset_index(drop=True) if g else None
print("per-fold delta (BE-A then PE-A), rows = fold, cols = seed")
for a, b in (("BE","A"),("PE","A")):
    print(f"--- {a} - {b}")
    for f in range(5):
        vals = []
        for s in (2027,2028,2029):
            da, db = load(a,s,f), load(b,s,f)
            if da is None or db is None: vals.append("   -- "); continue
            ea = da.signed_error.to_numpy(float); eb = db.signed_error.to_numpy(float)
            vals.append("%+6.2f" % (np.sqrt((ea**2).mean()) - np.sqrt((eb**2).mean())))
        print(f"  fold{f}: " + " ".join(vals))
print("\narms completed per fold:")
for arm in ("A","BE","PE"):
    for f in range(5):
        n = sum(load(arm,s,f) is not None for s in (2027,2028,2029))
        print(f"  {arm} fold{f}: {n}/3")
