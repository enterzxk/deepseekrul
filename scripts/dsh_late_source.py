import glob, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train
life = load_cmapss_train(Path('data/CMAPSS'), 'FD002').groupby('unit')['cycle'].max()
fs = glob.glob('experiments/order03/M1_global/FD002/*/matched_cutoff_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['e'] = np.clip(d.y_pred,0,125) - d.y_true
d['late'] = d.e > 0
print('=== by true RUL band ===')
bands = [(0,30),(30,60),(60,90),(90,126)]
for lo, hi in bands:
    s = d[(d.y_true>=lo)&(d.y_true<hi)]
    if len(s): print(f'  RUL {lo:3d}-{hi:3d}: n={len(s):4d} late={s.late.mean():.3f} mean_err={s.e.mean():+6.2f}')
print()
print('=== by engine lifespan tertile ===')
d['life'] = d.unit_id.map(life)
q1, q2 = d.life.quantile([0.33, 0.67])
for name, m in (('short', d.life<=q1), ('mid', (d.life>q1)&(d.life<=q2)), ('long', d.life>q2)):
    s = d[m]
    if len(s): print(f'  {name:6s} n={len(s):4d} late={s.late.mean():.3f} mean_err={s.e.mean():+6.2f}')
