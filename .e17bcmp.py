import glob, json
import numpy as np, pandas as pd
R = "/home/yndx/zxk/bearing-rul-pi/experiments"
def find(path_glob):
    g = sorted(glob.glob(path_glob))
    return pd.read_csv(g[0]).sort_values("unit_id").reset_index(drop=True) if g else None
rows = []
for arm in ("tf", "cnn1d"):
    for base, sub, tag in (("e11", f"{arm}_s2027_f{{f}}", "d=88"),
                           ("e17b", f"{arm}w128_s2027_f{{f}}", "d=128")):
        errs, mp = [], None
        for f in range(5):
            d = find(f"{R}/{base}/" + sub.format(f=f) + f"/FD002/{arm}_s2027/matched_cutoff_predictions.csv")
            if d is None: break
            errs.append(d.signed_error.to_numpy(float))
            if f == 0:
                m = find_meta = json.load(open(f"{R}/{base}/" + sub.format(f=0) + f"/FD002/{arm}_s2027/metrics.json"))
                mp = m.get("num_params")
        if len(errs) < 5:
            print(f"{arm} {tag}: incomplete {len(errs)}/5"); continue
        e = np.concatenate(errs)
        rows.append(dict(arm=arm, width=tag, rmse=float(np.sqrt((e ** 2).mean())), params=mp,
                         per_fold=[round(float(np.sqrt((x ** 2).mean())), 2) for x in errs]))
for r in rows:
    print(f"{r['arm']:6s} {r['width']:6s} RMSE {r['rmse']:6.3f}  params {r['params']}  per-fold {r['per_fold']}")
json.dump(rows, open("/home/yndx/zxk/bearing-rul-pi/results/dsh/e16/e17b_capacity.json", "w"), indent=2)
