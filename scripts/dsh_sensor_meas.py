"""Per-sensor measurement characterisation on FD002."""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
cols = list(SELECTED_SENSOR_COLUMNS)
rows = []
for c in cols:
    cvs, rhos = [], []
    for u, g in df.groupby('unit'):
        g = g.sort_values('cycle')
        early = g[c].to_numpy(float)[:20]
        allv = g[c].to_numpy(float)
        rul = np.arange(len(allv))[::-1].astype(float)
        cvs.append(early.std()/(abs(early.mean())+1e-9))
        if allv.std() > 0:
            rhos.append(spearmanr(allv, rul).statistic)
    rows.append({'sensor': c, 'healthy_cv': float(np.mean(cvs)), 'rho_with_rul': float(np.mean(rhos))})
t = pd.DataFrame(rows).sort_values('rho_with_rul')
print(f"{'sensor':10s} {'healthy CV':>10s} {'|rho| with RUL':>14s}")
for _, r in t.iterrows():
    print(f"{r.sensor:10s} {r.healthy_cv:10.4f} {abs(r.rho_with_rul):14.3f}")
print()
print(f'median healthy CV: {t.healthy_cv.median():.4f} | median |rho|: {t.rho_with_rul.abs().median():.3f}')
