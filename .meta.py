import json
import numpy as np
# inverse-variance combination of the two subset estimates of BE - A
# FD002: E10b, 5 folds x 52 engines, 100k clustered bootstrap
# FD001: E18,  5 folds x 20 engines, 100k clustered bootstrap
ests = {"FD002": dict(d=1.133, lo=0.245, hi=2.023, n=260),
        "FD001": dict(d=2.052, lo=0.505, hi=3.733, n=100)}
for k, v in ests.items():
    v["se"] = (v["hi"] - v["lo"]) / (2 * 1.959964)
w = {k: 1 / v["se"] ** 2 for k, v in ests.items()}
pooled = sum(ests[k]["d"] * w[k] for k in ests) / sum(w.values())
se = 1 / np.sqrt(sum(w.values()))
out = dict(method="inverse-variance weighted combination of the two subset estimates",
           arms="BE - A (time channel removed)", **{k: v for k, v in ests.items()},
           pooled_delta=float(pooled), pooled_se=float(se),
           pooled_ci95=[float(pooled - 1.959964 * se), float(pooled + 1.959964 * se)],
           total_engines=sum(v["n"] for v in ests.values()))
print(f"pooled BE - A = {pooled:.3f} [{out['pooled_ci95'][0]:.3f}, {out['pooled_ci95'][1]:.3f}] "
      f"over {out['total_engines']} engines (both subsets significant individually)")
json.dump(out, open("/home/yndx/zxk/bearing-rul-pi/results/dsh/e18/pooled_BE_A.json", "w"), indent=2)
