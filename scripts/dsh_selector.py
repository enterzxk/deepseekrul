"""Is per-engine method preference predictable from observable features?"""
import glob, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS

df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
cols = list(SELECTED_SENSOR_COLUMNS)
feat = {}
for u, g in df.groupby('unit'):
    g = g.sort_values('cycle'); x = g[cols].to_numpy(float); n = len(g)
    w = max(5, n // 20)
    early = x[:w]
    feat[u] = {'life': n, 'early_std': float(early.std(0).mean()),
               'early_mean': float(early.mean()), 'drift': float(np.abs(x[-w:].mean(0) - early.mean(0)).mean())}
F = pd.DataFrame(feat).T

def per_engine(root, tag):
    fs = glob.glob(f'experiments/{root}/{tag}/FD002/*/matched_cutoff_predictions.csv')
    if not fs: return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    d['sq'] = d.signed_error.astype(float) ** 2
    return d.groupby('unit_id').sq.mean().pow(0.5)

a, b = per_engine('order03','M1_global'), per_engine('order03','M6_tfA')
j = pd.concat([a.rename('m1'), b.rename('m6'), F], axis=1).dropna()
j['d'] = j.m1 - j.m6                # negative: M1 better
print('engines:', len(j), '| M1 better in', float((j.d<0).mean()), 'of cases')
for c in ['life','early_std','drift']:
    r = np.corrcoef(j[c], j.d)[0,1]
    print(f'  corr({c}, delta) = {r:+.3f}')
# AUC of 'M1 better' using each feature
def auc(score, label):
    pos, neg = score[label==1], score[label==0]
    if len(pos)==0 or len(neg)==0: return float('nan')
    return float((pos[:,None] > neg[None,:]).mean() + 0.5*(pos[:,None]==neg[None,:]).mean())
lab = (j.d < 0).astype(int).to_numpy()
for c in ['life','early_std','drift']:
    s = j[c].to_numpy(float)
    print(f'  AUC({c}) = {max(auc(s,lab), auc(-s,lab)):.3f}')
