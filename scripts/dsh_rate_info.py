"""Does the degradation RATE at mid-life carry lifespan information?"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
cols = list(SELECTED_SENSOR_COLUMNS)
life = df.groupby('unit')['cycle'].max()
rows = []
for u, g in df.groupby('unit'):
    g = g.sort_values('cycle'); T = int(life[u])
    end = max(50, int(round(T * 0.5)))
    w = g.iloc[end-50:end][cols].to_numpy(float)
    # normalise each sensor by its own early-life std, then sum absolute slopes
    early = g.iloc[:20][cols].to_numpy(float)
    sd = early.std(0) + 1e-6
    z = (w - early.mean(0)) / sd
    # per-sensor linear slope over the window, averaged
    t = np.arange(len(z), dtype=float)
    slopes = np.array([np.polyfit(t, z[:, j], 1)[0] for j in range(z.shape[1])])
    level = z[-1].mean()
    rows.append({'unit': u, 'life': T, 'rate': float(np.abs(slopes).mean()), 'level': float(level)})
r = pd.DataFrame(rows)
print(f'n_engines={len(r)}')
print(f'  corr(|rate|, life)  = {r.rate.corr(r.life):+.3f}')
print(f'  corr(level, life)   = {r.level.corr(r.life):+.3f}')
print(f'  R^2 of life ~ rate  = {r.rate.corr(r.life)**2:.3f}')
print(f'  R^2 of life ~ level = {r.level.corr(r.life)**2:.3f}')
