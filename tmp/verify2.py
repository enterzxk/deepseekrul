import json, statistics as st
from pathlib import Path
import numpy as np, pandas as pd
ok = fail = 0
def check(name, got, want, tol=0.06):
    global ok, fail
    good = abs(got - want) <= tol * max(1.0, abs(want))
    print(("PASS " if good else "FAIL ") + name + ": got %.3f want %.3f" % (got, want))
    ok += good; fail += (not good)

E = Path("experiments/punet_e2e"); subs = ["FD001","FD002","FD003","FD004"]
def load(s,f,n):
    p = E / s / ("f%d_s42" % f) / (n + ".json")
    return json.loads(p.read_text()) if p.exists() else None

d = [(load(s,f,"e2e_cqr_none")["winkler"]-load(s,f,"e2e_point_none")["winkler"])/load(s,f,"e2e_point_none")["winkler"]*100
     for s in subs for f in range(5) if load(s,f,"e2e_cqr_none") and load(s,f,"e2e_point_none")]
check("paired wins", sum(1 for x in d if x<0), 19, 0)
check("paired median", st.median(d), -8.7, 0.03)

for s, want_min, want_b5 in [("FD002",0.529,0.873),("FD003",0.714,0.893),("FD004",0.501,0.864)]:
    js = [load(s,f,"e2e_cqr_none") for f in range(5)]
    js = [j for j in js if j]
    check(s+" engine_min", st.mean(j["engine_min"] for j in js), want_min, 0.02)
    b5 = [j["diagnostics"]["stage_bins"][4]["coverage"] for j in js if len(j["diagnostics"]["stage_bins"])==5]
    check(s+" stage5", st.mean(b5), want_b5, 0.02)

# FEMTO: recompute from dumps using the evaluator logic
import sys; sys.path.insert(0, "scripts")
from evaluate_femto import pts_cqr, pts_loc, pts_scaled, at
F = Path("experiments/femto"); rel = []
for d0 in sorted(F.glob("Bearing*")):
    q, p = d0/"preds_q.csv", d0/"preds_p.csv"
    if not (q.exists() and p.exists()): continue
    qq, pp = pd.read_csv(q), pd.read_csv(p)
    tq, tp = qq.split=="test", pp.split=="test"
    y, lo, hi = qq.y.values, qq.lo.values, qq.hi.values
    yp, mu = pp.y.values, pp.mu.values
    lc = at(pts_loc(y[tq], lo[tq], hi[tq])); cc = at(pts_scaled(yp[tp], mu[tp], np.ones(tp.sum())))
    rel.append((lc-cc)/cc*100)
check("femto bearings", len(rel), 6, 0)
check("femto mean improvement", st.mean(rel), -23.3, 0.05)
check("femto wins", sum(1 for x in rel if x<0), 6, 0)
print("\nRESULT: %d passed, %d failed" % (ok, fail))
