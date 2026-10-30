"""How often does the change-point cap actually bind on FD002?

The change-point label caps RUL at (T - t_cp). It differs from the standard
cap-125 label only when T - t_cp < 125, i.e. when the detected onset is within
125 cycles of failure. This matters because it decides whether the method can do
anything at all on this benchmark.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from dataset import load_cmapss_train

df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]

def onsets(rule):
    out = []
    for u, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        t = g.cycle.to_numpy(float); x = g[cols].to_numpy(float); n = len(g)
        w = max(5, n // 20)
        mu = x[:w].mean(0); sd = x[:w].std(0) + 1e-9
        hi = ((x - mu) / sd).mean(1)
        k = max(5, n // 20)
        sm = np.convolve(hi, np.ones(k) / k, mode="same")
        if rule == "3sigma_early":
            base = sm[: max(5, n // 10)]
            thr = base.mean() + 3.0 * (base.std() + 1e-9)
        elif rule == "2sigma_early":
            base = sm[: max(5, n // 10)]
            thr = base.mean() + 2.0 * (base.std() + 1e-9)
        else:  # 1 sigma
            base = sm[: max(5, n // 10)]
            thr = base.mean() + 1.0 * (base.std() + 1e-9)
        ab = np.where(sm > thr)[0]
        cp = int(ab[0]) if ab.size else n - 1
        out.append(t[-1] - t[min(max(cp, 5), n - 1)])
    return np.array(out)

for rule in ("3sigma_early", "2sigma_early", "1sigma_early"):
    span = onsets(rule)
    binds = (span < 125).mean()
    print(f"{rule:14s} span mean={span.mean():6.1f} sd={span.std():5.1f} "
          f"median={np.median(span):5.1f} | cap binds in {binds*100:5.1f}% of engines")
