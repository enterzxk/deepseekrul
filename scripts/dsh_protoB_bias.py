"""Life-fraction protocol: does the bias pattern persist when cutoffs are proportional?"""
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
print('Protocol B (life fractions) bias by fraction x lifespan tertile:')
print(f"{'frac':>5s} {'n':>4s} {'bias':>8s} {'late':>6s}   by lifespan:")
q1, q2 = d.life.quantile([0.33, 0.67])
for f in sorted(d.cutoff.unique()):
    s = d[np.isclose(d.cutoff, f)]
    parts = []
    for name, m in (('short', s.life<=q1), ('mid', (s.life>q1)&(s.life<=q2)), ('long', s.life>q2)):
        t = s[m]
        parts.append(f'{name}={t.e.mean():+6.2f}' if len(t) else f'{name}=n/a')
    print(f'{f:5.1f} {len(s):4d} {s.e.mean():+8.2f} {(s.e>0).mean():6.3f}   ' + '  '.join(parts))
