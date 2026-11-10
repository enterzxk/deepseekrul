"""HI with per-operating-condition normalization."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path('/home/user/rulb')
sys.path.insert(0, str(ROOT))
from sklearn.cluster import KMeans
SENSORS = ['s_2','s_3','s_4','s_7','s_8','s_9','s_11','s_12','s_13','s_14','s_15','s_17','s_20','s_21']
OPS = ['op_1','op_2','op_3']
COLS = ['unit','cycle'] + OPS + [f's_{i}' for i in range(1,22)]
out = ROOT/'data'/'HI'; out.mkdir(parents=True, exist_ok=True)
for sub in ['FD001','FD002','FD003','FD004']:
    d = pd.read_csv(ROOT/'data'/'CMAPSS'/f'train_{sub}.txt', sep=r'\s+', header=None, names=COLS)
    d = d.sort_values(['unit','cycle']).reset_index(drop=True)
    nreg = len(d[OPS].drop_duplicates())
    k = min(6, max(1, nreg))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(d[OPS].to_numpy(float))
    lab = km.labels_
    X = d[SENSORS].to_numpy(np.float64).copy()
    # z-score WITHIN each operating cluster
    for c in range(k):
        m = lab == c
        if m.sum() < 2: continue
        mu = X[m].mean(0); sd = X[m].std(0) + 1e-9
        X[m] = (X[m] - mu) / sd
    C = np.cov(X.T); w, V = np.linalg.eigh(C)
    pc1 = X @ V[:, -1]
    cyc = d['cycle'].to_numpy(float)
    if np.corrcoef(pc1, cyc)[0,1] > 0: pc1 = -pc1
    lo, hi = np.percentile(pc1, 1), np.percentile(pc1, 99)
    hi_val = np.clip((pc1 - lo)/(hi - lo + 1e-9), 0.0, 1.0)
    # enforce monotone via per-engine isotonic-like cumulative min
    res = pd.DataFrame({'unit': d['unit'].to_numpy(), 'cycle': d['cycle'].to_numpy(), 'hi_raw': hi_val})
    mono = res.groupby('unit')['hi_raw'].transform(lambda s: np.minimum.accumulate(s.to_numpy()))
    res['hi'] = mono
    res.to_csv(out/f'hi_{sub}.csv', index=False)
    m = 0; t = 0
    for u, g in res.groupby('unit'):
        v = g['hi_raw'].to_numpy()
        if len(v) > 1: m += (np.diff(v) <= 0).sum(); t += len(v)-1
    print('%-6s n=%5d k=%d  corr(hi,cycle)=%+.3f  mono_frac=%.3f  hi %.2f-%.2f' % (
        sub, len(res), k, np.corrcoef(res['hi'], res['cycle'])[0,1], m/max(t,1), res['hi'].min(), res['hi'].max()))