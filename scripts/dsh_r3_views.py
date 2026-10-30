"""R3-1/2: correct feature registry (op_1..3 + cluster) and time-shortcut arms."""
import sys, warnings, json, hashlib
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
from sklearn.cluster import KMeans

CUTS = (60, 100, 140, 180); WIN = 50; CAP = 125.0
cols = list(SELECTED_SENSOR_COLUMNS); OPS = ['op_1','op_2','op_3']
frames = {ds: load_cmapss_train(Path('data/CMAPSS'), ds) for ds in ('FD001','FD002')}

# cluster on settings, fit on TRAIN engines only (FD002 multi-condition; FD001 single)
km = None
allops = pd.concat([f[OPS] for f in frames.values()], ignore_index=True)
n_clusters = 6 if allops.std().max() > 0.05 else 1
if n_clusters > 1:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=2026).fit(allops.to_numpy(float))
    print('kmeans clusters:', n_clusters)
else:
    print('single condition detected -> 1 cluster')

def slope(a):
    t = np.arange(len(a), dtype=float); return np.polyfit(t, a, 1)[0]

rows = []
for ds, df in frames.items():
    life = df.groupby('unit')['cycle'].max()
    for u, g in df.groupby('unit'):
        T = int(life[u]); g = g.sort_values('cycle').reset_index(drop=True)
        early = g.iloc[:20][cols].to_numpy(float)
        m0, s0 = early.mean(0), early.std(0); s0[s0 < 1e-9] = 1.0
        ops_all = g[OPS].to_numpy(float)
        cl = km.predict(ops_all) if km is not None else np.zeros(len(g), int)
        for c in CUTS:
            if T <= c or c < WIN: continue
            w = g.iloc[c-WIN:c]; z = (w[cols].to_numpy(float)-m0)/s0
            ops_w = ops_all[c-WIN:c]
            cl_w = cl[c-WIN:c]
            # V0: window only
            v0 = np.concatenate([z.mean(0), z.std(0), z[-1], [slope(z[:,j]) for j in range(len(cols))]])
            # V1: + observable cycle + settings + cluster one-hot + cluster fractions
            oh = np.bincount(cl_w, minlength=n_clusters)/len(cl_w)
            v1 = np.concatenate([v0, [float(c)], ops_w[-1], oh])
            # V2: + history strictly up to c (longer lookback summaries)
            hist = g.iloc[max(0, c-100):c][cols].to_numpy(float)
            zh = (hist-m0)/s0
            s100 = np.array([slope(zh[:,j]) for j in range(len(cols))]) if len(hist) >= 20 else np.zeros(len(cols))
            dstd = np.array([np.diff(zh[:,j]).std() if len(hist) > 2 else 0.0 for j in range(len(cols))])
            # past exposure u_t: cluster residency share over full prefix (not just window)
            cl_pre = cl[:c]; oh_pre = np.bincount(cl_pre, minlength=n_clusters)/len(cl_pre)
            v2 = np.concatenate([v1, s100, dstd, oh_pre])
            rows.append(dict(dataset=ds, engine=int(u), cutoff=c, life=T, has_hist100=(c>=100),
                             v0=v0, v1=v1, v2=v2, rul=min(float(T-c), CAP)))
print('observations:', len(rows), '| engines:', len({(r['dataset'],r['engine']) for r in rows}))
np.save('results/dsh/review_r3/views_meta.npy', np.array([0]))
import pickle
pickle.dump(rows, open('results/dsh/review_r3/views_rows.pkl','wb'))
print('per-cutoff counts:', {c: sum(1 for r in rows if r['cutoff']==c) for c in CUTS})
print('dims: v0=%d v1=%d v2=%d' % (len(rows[0]['v0']), len(rows[0]['v1']), len(rows[0]['v2'])))
print('script sha256:', hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16])
