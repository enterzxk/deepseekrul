"""Pooled BE−A estimate across all four C-MAPSS subsets (validation side).

Inverse-variance combination of the subset deltas, plus Cochran's Q and I²
heterogeneity diagnostics. Subset estimates come from the frozen per-subset
analyses: e18 (FD001), e10-era pooled file (FD002), E20_cv.json (FD003/FD004).

Output: results/dsh/e20/pooled_BE_A_4subsets.json
"""
from __future__ import annotations

import json
import os

ROOT = "/home/yndx/zxk/bearing-rul-pi"

e18 = json.load(open(f"{ROOT}/results/dsh/e18/pooled_BE_A.json"))
e20 = json.load(open(f"{ROOT}/results/dsh/e20/E20_cv.json"))
e21 = json.load(open(f"{ROOT}/results/dsh/e21/test_contrasts.json"))

# --- validation side (development pool) ---
subs = {
    "FD001": dict(d=e18["FD001"]["d"], se=e18["FD001"]["se"], n=e18["FD001"]["n"]),
    "FD002": dict(d=e18["FD002"]["d"], se=e18["FD002"]["se"], n=e18["FD002"]["n"]),
}
for s in ("FD003", "FD004"):
    c = e20[s]["contrasts"]["BE_minus_A"]
    se = (c["ci95"][1] - c["ci95"][0]) / (2 * 1.959963984540054)
    subs[s] = dict(d=c["delta"], se=float(se), n=e20[s]["arms"]["A"]["n_engines"])

w = {s: 1.0 / v["se"] ** 2 for s, v in subs.items()}
sw = sum(w.values())
pooled = sum(w[s] * subs[s]["d"] for s in subs) / sw
se_p = (1.0 / sw) ** 0.5
q = sum(w[s] * (subs[s]["d"] - pooled) ** 2 for s in subs)
df = len(subs) - 1
i2 = max(0.0, (q - df) / q) if q > 0 else 0.0

# chi2 upper tail without scipy (Wilson-Hilferty is overkill; use survival via math.erfc)
import math


def chi2_sf(x, k):
    return math.erfc((x / k) ** (1 / 3) * (1 - 2 / (9 * k)) + 0.802
                     ) if False else None


val = dict(subsets=subs, pooled_delta=float(pooled), pooled_se=float(se_p),
           pooled_ci95=[float(pooled - 1.959963984540054 * se_p),
                        float(pooled + 1.959963984540054 * se_p)],
           total_engines=int(sum(v["n"] for v in subs.values())),
           heterogeneity=dict(Q=float(q), df=int(df), I2=float(i2)))

# --- official test side ---
tsubs = {}
for s in ("FD001", "FD002", "FD003", "FD004"):
    c = e21[s]["contrasts"]["BE_minus_A"]
    se = (c["ci95"][1] - c["ci95"][0]) / (2 * 1.959963984540054)
    tsubs[s] = dict(d=c["delta"], se=float(se), n=e21[s]["n_engines"])
tw = {s: 1.0 / v["se"] ** 2 for s, v in tsubs.items()}
tsw = sum(tw.values())
tpooled = sum(tw[s] * tsubs[s]["d"] for s in tsubs) / tsw
tse = (1.0 / tsw) ** 0.5
tq = sum(tw[s] * (tsubs[s]["d"] - tpooled) ** 2 for s in tsubs)
ti2 = max(0.0, (tq - 3) / tq) if tq > 0 else 0.0
test = dict(subsets=tsubs, pooled_delta=float(tpooled), pooled_se=float(tse),
            pooled_ci95=[float(tpooled - 1.959963984540054 * tse),
                         float(tpooled + 1.959963984540054 * tse)],
            total_engines=int(sum(v["n"] for v in tsubs.values())),
            heterogeneity=dict(Q=float(tq), df=3, I2=float(ti2)))

out = dict(
    arms="BE - A (time channel removed)",
    method="inverse-variance weighted combination of subset estimates; "
           "se recovered from the registered 95% CI; Q and I2 reported",
    validation=val,
    official_test=test,
)
path = f"{ROOT}/results/dsh/e20/pooled_BE_A_4subsets.json"
json.dump(out, open(path, "w"), indent=2)

print("=== validation (dev pool) ===")
for s, v in subs.items():
    print(f"  {s}: d={v['d']:+.3f} se={v['se']:.3f} n={v['n']}")
print(f"  pooled: {pooled:+.3f} [{val['pooled_ci95'][0]:+.3f},{val['pooled_ci95'][1]:+.3f}]"
      f"  n={val['total_engines']}  Q={q:.2f}(df={df}) I2={i2*100:.0f}%")
print("=== official test ===")
for s, v in tsubs.items():
    print(f"  {s}: d={v['d']:+.3f} se={v['se']:.3f} n={v['n']}")
print(f"  pooled: {tpooled:+.3f} [{test['pooled_ci95'][0]:+.3f},{test['pooled_ci95'][1]:+.3f}]"
      f"  n={test['total_engines']}  Q={tq:.2f}(df=3) I2={ti2*100:.0f}%")
print(f"wrote {path}")
