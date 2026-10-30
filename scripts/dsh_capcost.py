"""How much RMSE does the arbitrary cap-125 convention cost?"""
import glob, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train
tr = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
life = tr.groupby('unit')['cycle'].max()
fs = glob.glob('experiments/order03/M1_global/FD002/*/protocol_c_predictions.csv')
if not fs: print('missing'); raise SystemExit
d = pd.concat([pd.read_csv(f) for f in fs])
print('cols:', list(d.columns)[:8])
cols = d.columns.tolist()
cyc = 'abs_cut' if 'abs_cut' in cols else ('cutoff' if 'cutoff' in cols else None)
if cyc is None: print('no cycle column; cannot recover uncapped RUL'); raise SystemExit
d['T'] = d.unit_id.map(life)
d['rul_uncapped'] = (d['T'] - d[cyc]).clip(lower=0)
print(f'rows={len(d)} | uncapped RUL range [{d.rul_uncapped.min():.0f},{d.rul_uncapped.max():.0f}]')
print()
print(' cap    RMSE     MAE    note')
for cap in (80, 100, 125, 150, 200, 10000):
    y = np.minimum(d.rul_uncapped, cap)
    e = d.y_pred - y
    label = 'no cap' if cap > 1000 else ''
    print(f'{cap:5d} {np.sqrt((e**2).mean()):7.2f} {np.abs(e).mean():7.2f}   {label}')
