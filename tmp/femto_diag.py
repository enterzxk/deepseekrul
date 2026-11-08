import pandas as pd, numpy as np
from pathlib import Path
for b in ["Bearing1_1", "Bearing1_2", "Bearing2_1"]:
    d = Path("experiments/femto") / b
    if not (d / "preds_q.csv").exists():
        continue
    q = pd.read_csv(d / "preds_q.csv")
    p = pd.read_csv(d / "preds_p.csv")
    t = q[q.split == "test"]
    pt = p[p.split == "test"]
    print(b)
    print("  y:     mean %.1f std %.1f min %.1f max %.1f" % (t.y.mean(), t.y.std(), t.y.min(), t.y.max()))
    print("  q_lo:  mean %.1f std %.1f" % (t.lo.mean(), t.lo.std()))
    print("  q_hi:  mean %.1f std %.1f  width %.1f" % (t.hi.mean(), t.hi.std(), (t.hi - t.lo).mean()))
    print("  mu:    mean %.1f std %.1f  sd %.1f" % (pt.mu.mean(), pt.mu.std(), pt.sd.mean()))
