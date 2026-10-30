import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from dataset import load_cmapss_train

df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
for u, g in list(df.groupby("unit"))[:4]:
    g = g.sort_values("cycle")
    t = g.cycle.to_numpy(float); x = g[cols].to_numpy(float); n = len(g)
    w = max(5, n // 20)
    mu = x[:w].mean(0); sd = x[:w].std(0) + 1e-9
    hi = ((x - mu) / sd).mean(1)
    k = max(5, n // 20)
    sm = np.convolve(hi, np.ones(k) / k, mode="same")
    base = sm[: max(5, n // 10)]
    print(f"unit {u} n={n} | sm range [{sm.min():.2f},{sm.max():.2f}] | base mean={base.mean():.2f} sd={base.std():.3f} "
          f"| thr(3sd)={base.mean()+3*base.std():.2f} | final sm={sm[-1]:.2f}")
