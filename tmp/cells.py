import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from evaluate_matrix import cqr_pts, loc_cqr_pts, const_pts, at, engine_split
from pathlib import Path
root = Path("experiments/punet_matrix")
rows = []
for sub in ["FD001","FD002","FD003","FD004"]:
    for seed in [42,7,2024]:
        for fold in range(5):
            d = root / sub / ("f%d_s%d" % (fold, seed))
            pq, pp = d/"preds_cqr.csv", d/"preds_point.csv"
            if not (pq.exists() and pp.exists()):
                continue
            c = np.genfromtxt(pq, delimiter=",", names=True)
            p = np.genfromtxt(pp, delimiter=",", names=True)
            cal, tst = engine_split(c["unit"])
            y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
            mu = (p["y_lo"] + p["y_hi"]) / 2.0
            rows.append((sub, seed, fold,
                         at(cqr_pts(y, lo, hi, cal, tst)),
                         at(loc_cqr_pts(y, lo, hi, cal, tst)),
                         at(const_pts(p["y_true"], mu, cal, tst))))
df = pd.DataFrame(rows, columns=["subset","seed","fold","cqr","loc","const"])
print(df.round(1).to_string(index=False))
print()
print(df.groupby("subset")[["cqr","loc","const"]].agg(["mean","std"]).round(1).to_string())
