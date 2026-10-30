"""Is the model predicting the population-average RUL at mid-life?"""
import glob, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train
life = load_cmapss_train(Path('data/CMAPSS'), 'FD002').groupby('unit')['cycle'].max()
fs = glob.glob('experiments/order03/M1_global/FD002/*/pseudo_test_predictions.csv')
d = pd.concat([pd.read_csv(f) for f in fs])
d['p'] = np.clip(d.y_pred, 0, 125); d['e'] = d.p - d.y_true
d['life'] = d.unit_id.map(life)
q1, q2 = d.life.quantile([0.33, 0.67])
s = d[np.isclose(d.cutoff, 0.5)]
print('fraction 0.5:')
print(f"{'group':8s} {'n':>4s} {'life':>6s} {'true RUL':>9s} {'pred':>7s} {'bias':>7s}")
for name, m in (('short', s.life<=q1), ('mid', (s.life>q1)&(s.life<=q2)), ('long', s.life>q2)):
    t = s[m]
    print(f'{name:8s} {len(t):4d} {t.life.median():6.0f} {t.y_true.mean():9.1f} {t.p.mean():7.1f} {t.e.mean():+7.2f}')
print(f'{"all":8s} {len(s):4d} {s.life.median():6.0f} {s.y_true.mean():9.1f} {s.p.mean():7.1f} {s.e.mean():+7.2f}')
print()
print('spread of predictions across groups:', round(s.groupby(pd.cut(s.life, [0,q1,q2,1e9])).p.mean().std(), 2))
print('spread of true RUL across groups  :', round(s.groupby(pd.cut(s.life, [0,q1,q2,1e9])).y_true.mean().std(), 2))
