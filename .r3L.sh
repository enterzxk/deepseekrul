#!/bin/bash
cat > /home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_Lstrat.py <<'PYEOF'
"""Does stratifying the failure level L shrink T_hat bias for short-lived engines?"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

COLS = list(SELECTED_SENSOR_COLUMNS); OPS = ['op_1','op_2','op_3']
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
life = df.groupby('unit')['cycle'].max(); units = np.sort(df.unit.unique())
tr = units[:208]; va = units[208:]
km = KMeans(n_clusters=6, n_init=10, random_state=2026).fit(df[OPS].to_numpy(float))
df['cl'] = km.predict(df[OPS].to_numpy(float))
stats = {c: (g[COLS].mean().to_numpy(), g[COLS].std().to_numpy()+1e-9) for c, g in df[df.unit.isin(tr)].groupby('cl')}
def zc(frame):
    f = frame.reset_index(drop=True); out = np.empty((len(f), len(COLS)))
    for c, g in f.groupby('cl'):
        mu, sd = stats.get(c, (np.zeros(len(COLS)), np.ones(len(COLS))))
        out[g.index.to_numpy()] = (g[COLS].to_numpy(float)-mu)/sd
    return out
pca = PCA(n_components=1, random_state=0).fit(zc(df[df.unit.isin(tr)])); w = pca.components_[0]

end_h, rate, ends_cl = {}, {}, {}
for u in tr:
    g = df[df.unit == u]; h = zc(g) @ w
    end_h[u] = float(h[-1])
    t = np.arange(len(h), dtype=float); rate[u] = float(np.polyfit(t, h, 1)[0])
    ends_cl[u] = int(g['cl'].iloc[-1])
L_glob = float(np.mean(list(end_h.values())))
L_cl = {c: float(np.mean([end_h[u] for u in tr if ends_cl[u] == c])) for c in set(ends_cl.values())}
rq1 = np.quantile(list(rate.values()), 1/3)
def rate_bin(u): return 'slow' if rate[u] <= rq1 else 'fast'
L_rate = {b: float(np.mean([end_h[u] for u in tr if rate_bin(u) == b])) for b in ('slow','fast')}
print('L_global=%.3f  L_by_cluster=%s  L_by_rate=%s' % (L_glob, {k: round(v,2) for k,v in L_cl.items()}, {k: round(v,2) for k,v in L_rate.items()}))

def that(u, c, L):
    g = df[df.unit == u]; h = (zc(g) @ w)[:c]; t = np.arange(len(h), dtype=float)
    sl = float(np.polyfit(t, h, 1)[0])
    return float(np.clip(c + (L - h[-1])/sl, 1.2*c, 400.0)) if sl > 1e-9 else np.nan
rows = []
tl = np.array([float(life[u]) for u in va]); q1, q2 = np.quantile(tl, [1/3, 2/3])
for u in va:
    T = float(life[u]); band = 'short' if T <= q1 else ('mid' if T <= q2 else 'long')
    c = 140
    if T <= c: continue
    g_end_cl = int(df[df.unit == u]['cl'].iloc[-1])
    rows.append(dict(unit=int(u), T=T, band=band,
                     g=that(u, c, L_glob),
                     cl=that(u, c, L_cl.get(g_end_cl, L_glob)),
                     rt=that(u, c, L_rate['slow'] if np.isfinite(that(u,c,L_glob)) else L_glob)))
r = pd.DataFrame(rows)
print()
print(f"{'band':>6s} {'n':>3s} {'median T_hat/T (global)':>24s} {'(cluster)':>11s} {'(rate)':>9s}")
for b, g in r.groupby('band'):
    print(f'{b:>6s} {len(g):3d} {np.nanmedian(g.g/g.T):24.2f} {np.nanmedian(g.cl/g.T):11.2f} {np.nanmedian(g.rt/g.T):9.2f}')
print()
print('all: global %.2f cluster %.2f rate %.2f' % (np.nanmedian(r.g/r.T), np.nanmedian(r.cl/r.T), np.nanmedian(r.rt/r.T)))
print('corr with T at c=140: global %.3f cluster %.3f rate %.3f' % tuple(np.corrcoef(r[k], r.T)[0,1] for k in ('g','cl','rt')))
r.to_csv('/home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/L_strat.csv', index=False)
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_Lstrat.py 2>&1 | grep -v Warning | tail -12
