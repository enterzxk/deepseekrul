#!/bin/bash
cd /home/yndx/zxk/bearing-rul-pi || exit 1
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH
/home/yndx/miniconda3/envs/a40_env/bin/python - <<'EOF'
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
df = load_cmapss_train(Path("data/CMAPSS"), "FD002")
cols = list(SELECTED_SENSOR_COLUMNS)
spans, fired = [], 0
for u, g in df.groupby("unit"):
    g = g.sort_values("cycle"); t = g.cycle.to_numpy(float); x = g[cols].to_numpy(float); n = len(g)
    w = max(5, n // 20); mu = x[:w].mean(0); sd = x[:w].std(0) + 1e-9
    hi = ((x - mu) / sd).mean(1); k = max(5, n // 20)
    smv = np.convolve(hi, np.ones(k)/k, mode="valid"); pad = n - len(smv)
    sm = np.concatenate([np.full(pad//2, smv[0]), smv, np.full(pad-pad//2, smv[-1])])
    base = sm[: max(5, n//10)]; thr = base.mean() + 3.0*(base.std()+1e-9)
    ab = np.where(sm > thr)[0]
    if ab.size:
        fired += 1
        cp = int(min(max(ab[0], 5), n-1)); spans.append(t[-1] - t[cp])
spans = np.array(spans)
print(f"fired={fired}/260")
print(f"span: mean={spans.mean():.1f} median={np.median(spans):.1f} min={spans.min():.1f} max={spans.max():.1f}")
print(f"fired engines with span<125: {int((spans<125).sum())} ({100*(spans<125).mean():.1f}% of fired)")
EOF
