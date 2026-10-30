import glob, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0,'src')
from dataset import load_cmapss_train
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
life = df.groupby('unit')['cycle'].max()
def pe(tag):
    fs = glob.glob(f'experiments/order03/{tag}/FD002/*/matched_cutoff_predictions.csv')
    d = pd.concat([pd.read_csv(f) for f in fs]); d['sq'] = d.signed_error.astype(float)**2
    return d.groupby('unit_id').sq.mean()
j = pd.concat([pe('M1_global').rename('m1'), pe('M6_tfA').rename('m6')], axis=1).dropna()
j['life'] = [life[u] for u in j.index]
j = j.sort_values('life')
j['tert'] = pd.qcut(j.life, 3, labels=['short','mid','long'])
print(f"{'tert':6s} {'n':>3s} {'life~':>6s} {'M1':>6s} {'M6':>6s} {'diff':>7s} {'MSE share':>9s}")
tot = (j.m1.sum()+j.m6.sum())/2
for t, s in j.groupby('tert', observed=True):
    r1, r6 = np.sqrt(s.m1.mean()), np.sqrt(s.m6.mean())
    print(f'{t:6s} {len(s):3d} {s.life.median():6.0f} {r1:6.2f} {r6:6.2f} {r1-r6:+7.2f} {100*s.m1.sum()/tot:8.1f}%')
print()
print(f'pooled: M1={np.sqrt(j.m1.mean()):.2f} M6={np.sqrt(j.m6.mean()):.2f} diff={np.sqrt(j.m1.mean())-np.sqrt(j.m6.mean()):+.2f}')
