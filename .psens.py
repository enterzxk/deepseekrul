import glob, json, os
import numpy as np, pandas as pd
R = "/home/yndx/zxk/bearing-rul-pi/experiments"
out = {"question": "how much does the evaluation protocol alone move the reported number?",
       "model": "arm A (candidate), 3 seeds, fold 1, FD002", "rows": []}
rows = []
for s in [2027, 2028, 2029]:
    g = glob.glob(f"{R}/e10/A_s{s}_f1/FD002/A_s{s}/protocol_c_predictions.csv")
    if g: rows.append(pd.read_csv(g[0]))
d = pd.concat(rows)
for cut in sorted(d.abs_cut.unique()):
    sub = d[d.abs_cut == cut]; e = sub.signed_error.to_numpy(float)
    out["rows"].append(dict(protocol=f"abs_cut={int(cut)}", n_obs=int(len(e)),
        rmse=float(np.sqrt((e**2).mean())), mae=float(np.abs(e).mean()),
        nasa=float(np.mean(np.where(e<0, np.exp(-e/13)-1, np.exp(e/10)-1))),
        engines=int(sub.unit_id.nunique())))
g = glob.glob(f"{R}/e10/A_s*_f1/FD002/A_s*/matched_cutoff_predictions.csv")
m = pd.concat([pd.read_csv(x) for x in g]); e = m.signed_error.to_numpy(float)
out["rows"].append(dict(protocol="matched (paper main table)", n_obs=int(len(e)),
    rmse=float(np.sqrt((e**2).mean())), mae=float(np.abs(e).mean()),
    nasa=float(np.mean(np.where(e<0, np.exp(-e/13)-1, np.exp(e/10)-1))),
    engines=int(m.unit_id.nunique())))
rs = [r["rmse"] for r in out["rows"]]
out["spread_rmse"] = [min(rs), max(rs)]
out["spread_ratio"] = float(max(rs)/min(rs))
out["conclusion"] = ("The same trained model spans %.2f-%.2f RMSE (x%.2f) purely as a function of "
                     "the cutoff rule, with the engine count changing from %d to %d. Absolute RMSE is "
                     "therefore not comparable across papers that use different cutoff or cap "
                     "conventions." % (min(rs), max(rs), max(rs)/min(rs),
                     min(r["engines"] for r in out["rows"]), max(r["engines"] for r in out["rows"])))
os.makedirs(f"{R}/../results/dsh/e14", exist_ok=True)
json.dump(out, open(f"{R}/../results/dsh/e14/protocol_sensitivity.json", "w"), indent=2)
print(out["conclusion"])
print("wrote results/dsh/e14/protocol_sensitivity.json")
