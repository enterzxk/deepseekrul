"""One-sided conformal lower bound as an operating estimate, and its late-risk.

Symmetric intervals answer "how wide", but the objective cares about late
replacement. A one-sided lower bound L satisfies P(RUL >= L) >= 1 - alpha, so
acting on L should cut late decisions at the cost of earlier ones. Evaluated with
engine-level folds, 200 repeats.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("results/dsh/review_r1")
fs = glob.glob("experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv")
d = pd.concat([pd.read_csv(f) for f in fs])
g = d.groupby("unit_id").agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
yt = g.y_true.to_numpy(); yp = g.y_pred.to_numpy()
r = yt - yp                      # positive when the model under-predicts
units = g.index.to_numpy()

def one_sided(alpha, reps=200, seed=20261022):
    rng = np.random.default_rng(seed)
    viol, late_pt, late_b, early_b, widths = [], [], [], [], []
    for _ in range(reps):
        idx = rng.permutation(len(units)); h = len(units) // 2
        cal, ev = idx[:h], idx[h:]
        n = len(cal); k = int(np.ceil((n + 1) * alpha)); k = max(1, min(k, n))
        q = np.sort(r[cal])[k - 1]         # alpha-quantile of residuals
        L = yp[ev] + q
        viol.append((yt[ev] < L).mean())           # P(RUL < L), should be <= alpha
        late_pt.append((yp[ev] > yt[ev]).mean())   # baseline late fraction
        late_b.append((L > yt[ev]).mean())         # acting on L: still late?
        early_b.append((L + 10 < yt[ev]).mean())   # early by more than 10
        widths.append(q)
    return dict(alpha=alpha, violation=float(np.mean(viol)),
                late_point=float(np.mean(late_pt)), late_bound=float(np.mean(late_b)),
                early_gt10=float(np.mean(early_b)), shift=float(np.mean(widths)))

rows = [one_sided(a) for a in (0.1, 0.2, 0.3)]
out = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "conformal_onesided.csv", index=False)
print(out.round(3).to_string(index=False))
