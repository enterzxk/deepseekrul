"""Does the onset detector ever fire? Mirrors the dataset.py rule exactly.

If the threshold is never exceeded, every engine falls back to 0.6*T, and the
"change-point" labels are really just a fixed fraction-of-life cap.
"""
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from dataset import load_cmapss_train

df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
fired, span_fired, span_fallback = 0, [], []
for u, g in df.groupby("unit"):
    g = g.sort_values("cycle")
    t = g.cycle.to_numpy(float); x = g[cols].to_numpy(float); n = len(g)
    w = max(5, n // 20)
    mu = x[:w].mean(0); sd = x[:w].std(0) + 1e-9
    hi = ((x - mu) / sd).mean(1)
    k = max(5, n // 20)
    sm = np.convolve(hi, np.ones(k) / k, mode="same")
    base = sm[: max(5, n // 10)]
    thr = base.mean() + 3.0 * (base.std() + 1e-9)
    ab = np.where(sm > thr)[0]
    if ab.size:
        fired += 1
        cp = int(ab[0]); span_fired.append(t[-1] - t[min(max(cp, 10), n - 1)])
    else:
        cp = max(0, int(0.6 * n)); span_fallback.append(t[-1] - t[min(max(cp, 10), n - 1)])
print(f"engines={fired+len(span_fallback)}  detector fired in {fired} ({100*fired/(fired+len(span_fallback)):.1f}%)")
if span_fired: print(f"  fired span: mean={np.mean(span_fired):.1f} median={np.median(span_fired):.1f}")
if span_fallback: print(f"  fallback span: mean={np.mean(span_fallback):.1f} median={np.median(span_fallback):.1f}")
allspan = np.array(span_fired + span_fallback)
print(f"cap (125) binds in {100*(allspan < 125).mean():.1f}% of engines")
