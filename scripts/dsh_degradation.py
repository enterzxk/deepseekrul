"""Different degradation scenarios: fast (short-life) vs slow (long-life) engines."""
import glob, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
life = df.groupby('unit')['cycle'].max()
def per_engine(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean()
a, b = per_engine('M1_global'), per_engine('M6_tfA')
j = pd.concat([a.rename('m1'), b.rename('m6')], axis=1).dropna()
j['life'] = [life[u] for u in j.index]
j['short'] = j.life <= j.life.quantile(0.33)
j['long'] = j.life >= j.life.quantile(0.67)
print(f"{'group':10s} {'n':>3s} {'M1':>6s} {'M6':>6s} {'M1-M6':>7s} {'life med':>9s}")
for name, m in (('short (fast)', j.short), ('long (slow)', j.long), ('all', pd.Series(True, index=j.index))):
    s = j[m]
    r1, r6 = np.sqrt(s.m1.mean()), np.sqrt(s.m6.mean())
    print(f'{name:10s} {len(s):3d} {r1:6.2f} {r6:6.2f} {r1-r6:+7.2f} {s.life.median():9.0f}')
