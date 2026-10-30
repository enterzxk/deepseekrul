#!/bin/bash
cat > /home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_that.py <<'PYEOF'
"""Is a deployable prefix-based lifespan estimate T_hat informative? (no training)"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS, SensorMinMaxScaler

COLS = list(SELECTED_SENSOR_COLUMNS)
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
life = df.groupby('unit')['cycle'].max()
units = np.sort(df.unit.unique())
tr_units, va_units = units[:208], units[208:]
tr_df = df[df.unit.isin(tr_units)].reset_index(drop=True)
sc = SensorMinMaxScaler(COLS).fit(tr_df[COLS])

# failure level from TRAIN engines only: mean z at their final cycle
fail = []
for u in tr_units:
    z = sc.transform(df[df.unit == u][COLS])
    fail.append(float((z[-1] - z[:5].mean(0)).mean()))
L = float(np.mean(fail))
print('failure level L = %.4f (train engines)' % L)

def that_for(u, upto):
    g = df[df.unit == u]
    z = sc.transform(g[COLS])[:upto]
    zz = z.mean(axis=1)
    if len(zz) < 5: return np.nan, np.nan
    t = np.arange(len(zz), dtype=float)
    sl = np.polyfit(t, zz, 1)[0]
    if sl <= 1e-6: return np.nan, float(sl)
    th = upto + (L - zz[-1]) / sl
    return float(np.clip(th, 1.2*upto, 400.0)), float(sl)

rows = []
for u in va_units:
    T = float(life[u])
    for frac in (0.3, 0.5, 0.7):
        upto = max(10, int(round(T*frac)))
        th, sl = that_for(u, upto)
        rows.append(dict(unit=int(u), life=T, frac=frac, upto=upto, that=th, slope=sl))
r = pd.DataFrame(rows)
print()
print(f"{'frac':>5s} {'n':>4s} {'corr(T_hat,T)':>14s} {'R2':>7s} {'med T_hat/T':>12s} {'nan':>4s}")
for frac, g in r.groupby('frac'):
    ok = g.dropna(subset=['that'])
    c = np.corrcoef(ok.that, ok.life)[0,1] if len(ok) > 2 else np.nan
    ratio = float(np.median(ok.that/ok.life)) if len(ok) else np.nan
    print(f'{frac:5.1f} {len(g):4d} {c:14.3f} {c*c:7.3f} {ratio:12.2f} {int(g.that.isna().sum()):4d}')
r.to_csv('/home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/that_diagnostic.csv', index=False)
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_that.py
