#!/bin/bash
cat > /home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_that2.py <<'PYEOF'
"""Deployable T_hat at ABSOLUTE cutoffs (no dependence on true lifespan)."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS, SensorMinMaxScaler

COLS = list(SELECTED_SENSOR_COLUMNS)
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
life = df.groupby('unit')['cycle'].max()
units = np.sort(df.unit.unique()); tr_units, va_units = units[:208], units[208:]
tr_df = df[df.unit.isin(tr_units)].reset_index(drop=True)
sc = SensorMinMaxScaler(COLS).fit(tr_df[COLS])
fail = []
for u in tr_units:
    z = sc.transform(df[df.unit == u][COLS]); fail.append(float((z[-1] - z[:5].mean(0)).mean()))
L = float(np.mean(fail)); print('L = %.4f' % L)

rows = []
for u in va_units:
    g = df[df.unit == u]; T = float(life[u])
    for c in (60, 100, 140):
        if T <= c: continue
        z = sc.transform(g[COLS])[:c].mean(axis=1)
        t = np.arange(len(z), dtype=float); sl = float(np.polyfit(t, z, 1)[0])
        raw = c + (L - z[-1])/sl if sl > 1e-6 else np.nan
        th = float(np.clip(raw, 1.2*c, 400.0)) if np.isfinite(raw) else np.nan
        rows.append(dict(unit=int(u), life=T, cutoff=c, slope=sl, that_raw=raw, that=th,
                         clipped=bool(np.isfinite(raw) and (raw < 1.2*c or raw > 400)), ok=np.isfinite(th)))
r = pd.DataFrame(rows)
print(f"{'cut':>4s} {'n':>4s} {'ok':>3s} {'corr':>7s} {'R2':>6s} {'medT_hat/T':>10s} {'clipped%':>9s}")
for c, g in r.groupby('cutoff'):
    ok = g[g.ok]
    if len(ok) > 2:
        cc = float(np.corrcoef(ok.that, ok.life)[0,1])
        print(f'{c:4d} {len(g):4d} {len(ok):3d} {cc:7.3f} {cc*cc:6.3f} {float(np.median(ok.that/ok.life)):10.2f} {100*g.clipped.mean():8.0f}%')
    else:
        print(f'{c:4d} {len(g):4d} {len(ok):3d}   n/a')
r.to_csv('/home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/that_absolute.csv', index=False)
print()
print('if R2 is near 0, an oracle-style scale normalisation is NOT reachable online')
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_that2.py
