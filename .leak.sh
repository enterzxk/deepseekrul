#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "src")
from pathlib import Path
import dataset
from dataset import load_cmapss_train
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
cols = [c for c in df.columns if c.startswith("s") and c[1:].isdigit()]
rows = []
for u, g in df.groupby("unit"):
    g = g.sort_values("cycle"); t = g.cycle.to_numpy(float); x = g[cols].to_numpy(float); n = len(g)
    mu = x[: max(5, n // 20)].mean(0); sd = x[: max(5, n // 20)].std(0) + 1e-9
    hi = ((x - mu) / sd).mean(1); k = max(5, n // 20)
    sm = np.convolve(hi, np.ones(k) / k, mode="same")
    base = sm[: max(5, n // 10)]
    thr = base.mean() + 3.0 * (base.std() + 1e-9)
    ab = np.where(sm > thr)[0]
    cp = int(ab[0]) if ab.size else max(0, int(0.6 * n))
    cp = min(max(cp, 10), n - 1)
    T = t[-1] - t[0] + 1
    t_cp = t[cp] - t[0] + 1
    rows.append((T, t_cp, T - t_cp, 125.0 / max(T - t_cp, 1.0)))
A = np.array(rows)
print(f"engines={len(A)}")
print(f"total life T:      mean={A[:,0].mean():.0f} sd={A[:,0].std():.0f} range=[{A[:,0].min():.0f},{A[:,0].max():.0f}]")
print(f"degradation span:  mean={A[:,2].mean():.0f} sd={A[:,2].std():.0f} range=[{A[:,2].min():.0f},{A[:,2].max():.0f}]")
print(f"label scale s=125/span: mean={A[:,3].mean():.2f} sd={A[:,3].std():.2f} range=[{A[:,3].min():.2f},{A[:,3].max():.2f}]")
print(f"corr(span, T) = {np.corrcoef(A[:,2], A[:,0])[0,1]:.3f}")
EOF
