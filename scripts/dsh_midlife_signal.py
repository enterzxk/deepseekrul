"""Does mid-life optimism track signal strength (lifespan as proxy)?"""
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
u = d[d.y_true < 124.5]
mid = u[(u.y_true>=60)&(u.y_true<120)]
q1, q2 = mid.life.quantile([0.33, 0.67])
print('mid-life band (RUL 60-120), split by engine lifespan:')
print(f"{'group':8s} {'n':>4s} {'life~':>6s} {'bias':>8s} {'late':>6s}")
for name, m in (('short', mid.life<=q1), ('mid', (mid.life>q1)&(mid.life<=q2)), ('long', mid.life>q2)):
    s = mid[m]
    if len(s): print(f'{name:8s} {len(s):4d} {s.life.median():6.0f} {s.e.mean():+8.2f} {(s.e>0).mean():6.3f}')
print()
print('correlation across engines between lifespan and mid-life bias:')
per = mid.groupby('unit_id').agg(life=('life','first'), bias=('e','mean'))
print(f'  n_engines={len(per)}  corr(life, bias) = {per.life.corr(per.bias):+.3f}')
