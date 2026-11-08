import sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from pathlib import Path
from evaluate_femto import pts_cqr, pts_scaled, at
F = Path("experiments/femto"); rows = []
for d in sorted(F.glob("Bearing*")):
    q, p = d/"preds_q.csv", d/"preds_p.csv"
    if not (q.exists() and p.exists()): continue
    qq, pp = pd.read_csv(q), pd.read_csv(p)
    tq, tp = qq.split=="test", pp.split=="test"
    y, lo, hi = qq.y.values, qq.lo.values, qq.hi.values
    yp, mu = pp.y.values, pp.mu.values
    rows.append((d.name, at(pts_cqr(y[tq], lo[tq], hi[tq])), at(pts_scaled(yp[tp], mu[tp], np.ones(tp.sum())))))
df = pd.DataFrame(rows, columns=["bearing","cqr","const"])
rel = (df.cqr - df.const)/df.const*100
print(df.round(2).to_string(index=False))
print("cqr mean %.2f const mean %.2f | wins %d/%d | mean %+.1f%%" % (
    df.cqr.mean(), df.const.mean(), int((rel<0).sum()), len(rel), rel.mean()))
