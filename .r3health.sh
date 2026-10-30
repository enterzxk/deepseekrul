#!/bin/bash
cat > /home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_health.py <<'PYEOF'
"""Build a condition-corrected monotone health index and test prefix extrapolation."""
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
tr_units = set(units[:208]); va_units = list(units[208:])

km = KMeans(n_clusters=6, n_init=10, random_state=2026).fit(df[OPS].to_numpy(float))
df['cl'] = km.predict(df[OPS].to_numpy(float))

# per-cluster sensor mean/sd from TRAIN engines only
tr_mask = df.unit.isin(tr_units)
stats = {c: (g[COLS].mean().to_numpy(), g[COLS].std().to_numpy() + 1e-9)
         for c, g in df[tr_mask].groupby('cl')}

def corrected(frame):
    out = np.empty((len(frame), len(COLS)))
    for c, g in frame.groupby('cl'):
        mu, sd = stats.get(c, (np.zeros(len(COLS)), np.ones(len(COLS))))
        out[g.index.to_numpy()] = (g[COLS].to_numpy(float) - mu) / sd
    return out

Ztr = corrected(df[tr_mask]); pca = PCA(n_components=1, random_state=0).fit(Ztr)
print('PC1 explained variance ratio: %.3f' % pca.explained_variance_ratio_[0])
w = pca.components_[0]
if np.dot(w, Ztr.mean(0)) < 0 or np.corrcoef(Ztr @ w, np.arange(len(Ztr)))[0,1] < 0:
    w = -w

def mono(frame):
    z = corrected(frame); h = z @ w; d = np.diff(h)
    return float(np.mean(d > 0)), h

print()
print('monotonicity (fraction of increasing steps) on the FULL trajectory:')
rows = []
for label, us in (('train', sorted(tr_units)), ('val', va_units)):
    ms = [mono(df[df.unit == u])[0] for u in us]
    rows.append((label, len(ms), float(np.median(ms)), float(np.mean(np.array(ms) > 0.6))))
    print(f'  {label}: n={len(ms)} median={np.median(ms):.3f} frac>0.6={np.mean(np.array(ms)>0.6):.3f}')

L = float(np.mean([mono(df[df.unit == u])[1][-1] for u in sorted(tr_units)]))
print(); print('failure level L (PC1 at train end) = %.3f' % L)
print()
print(f"{'cut':>4s} {'n':>4s} {'nan':>4s} {'lower':>6s} {'free':>5s} {'corr':>7s} {'R2':>6s} {'medT_hat/T':>10s}")
rec = []
for c in (60, 100, 140):
    vals = []
    for u in va_units:
        g = df[df.unit == u]; T = float(life[u])
        if T <= c: continue
        h = mono(g)[1][:c]; t = np.arange(len(h), dtype=float)
        sl = float(np.polyfit(t, h, 1)[0])
        raw = c + (L - h[-1]) / sl if sl > 1e-9 else np.nan
        th = float(np.clip(raw, 1.2*c, 400.0)) if np.isfinite(raw) else np.nan
        vals.append(dict(unit=int(u), life=T, that=th, raw=raw, lower=bool(np.isfinite(th) and np.isclose(th, 1.2*c))))
    v = pd.DataFrame(vals); free = v[v.that.notna() & ~v.lower]
    if len(free) > 3:
        cc = float(np.corrcoef(free.that, free.life)[0,1])
        print(f'{c:4d} {len(v):4d} {int(v.that.isna().sum()):4d} {int(v.lower.sum()):6d} {len(free):5d} {cc:7.3f} {cc*cc:6.3f} {float(np.median(free.that/free.life)):10.2f}')
    else:
        print(f'{c:4d} {len(v):4d} {int(v.that.isna().sum()):4d} {int(v.lower.sum()):6d} {len(free):5d}   n/a')
    rec.append(v.assign(cutoff=c))
pd.concat(rec).to_csv('/home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/health_that.csv', index=False)
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_health.py 2>&1 | grep -v Warning
