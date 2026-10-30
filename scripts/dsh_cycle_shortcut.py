"""Does cycle number predict bias after controlling for true RUL?"""
import glob, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train
life = load_cmapss_train(Path('data/CMAPSS'), 'FD002').groupby('unit')['cycle'].max()
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['p'] = np.clip(d.y_pred, 0, 125); d['e'] = d.p - d.y_true
d['life'] = d.unit_id.map(life)
d['cycle'] = d.life - d.y_true
u = d[d.y_true < 124.5].copy()
X = np.column_stack([np.ones(len(u)), u.cycle.to_numpy(), u.y_true.to_numpy()])
beta, *_ = np.linalg.lstsq(X, u.e.to_numpy(), rcond=None)
print(f'n={len(u)}')
print(f'  bias = {beta[0]:+.3f} + ({beta[1]:+.4f})*cycle + ({beta[2]:+.4f})*RUL')
print(f'  -> per +10 cycles at fixed RUL, bias changes {10*beta[1]:+.3f}')
print()
print('same model with lifespan instead of cycle:')
X2 = np.column_stack([np.ones(len(u)), u.life.to_numpy(), u.y_true.to_numpy()])
b2, *_ = np.linalg.lstsq(X2, u.e.to_numpy(), rcond=None)
print(f'  bias = {b2[0]:+.3f} + ({b2[1]:+.4f})*life + ({b2[2]:+.4f})*RUL')
print(f'  -> per +10 lifespan at fixed RUL, bias changes {10*b2[1]:+.3f}')
