#!/bin/bash
cat > /home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_four_summary.py <<'PYEOF'
import json, glob
import numpy as np, pandas as pd
rows = [json.load(open(f)) for f in sorted(glob.glob('/home/yndx/zxk/bearing-rul-pi/experiments/r3_four/*/metrics.json'))]
t = pd.DataFrame(rows)
print('runs:', len(t), '| train windows:', int(t.n_train.median()))
agg = t.groupby('arm').agg({'rmse':['mean','std'],'mae':'mean','nasa_mean':['mean','std'],'late':['mean','std'],'severe_e10':'mean'}).round(3)
print(agg.to_string())
print()
p = t.pivot(index='seed', columns='arm', values='rmse')
print(p.round(3).to_string())
for a in ('none','oracle','est'):
    if a in p.columns:
        d = p[a]-p['cycle']
        print(f'{a:7s} - cycle: {[round(x,2) for x in d]} mean {d.mean():+.3f}')
print()
print('by-lifespan RMSE / bias (mean over seeds):')
for arm in ('cycle','none','oracle','est'):
    s = t[t.arm==arm]
    if not len(s): continue
    out = {}
    for b in ('short','mid','long'):
        vals = [r['by_lifespan'][b] for r in s if b in r['by_lifespan']]
        out[b] = f"{np.mean([v['rmse'] for v in vals]):.1f}/{np.mean([v['bias'] for v in vals]):+.1f}"
    print(' ', arm.ljust(7), out)
t.to_csv('/home/yndx/zxk/bearing-rul-pi/results/dsh/review_r3/four_arms.csv', index=False)
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_summary.py
