import json, glob
import numpy as np, pandas as pd
rows = [json.load(open(f)) for f in sorted(glob.glob('experiments/r3_pe/*/metrics.json'))]
t = pd.DataFrame(rows)
assert t.epochs_ran.min() >= 30, t[['arm','seed','epochs_ran']]
print('runs:', len(t))
s = t.groupby('arm').agg({'rmse':['mean','std'],'mae':'mean','nasa_mean':['mean','std'],'late':['mean','std'],'severe_e10':['mean','std']}).round(3)
print(s.to_string())
t.to_csv('results/dsh/review_r3/neural_time_arms.csv', index=False)
# paired per-seed deltas
p = t.pivot(index='seed', columns='arm', values='rmse')
print()
print(p.round(3).to_string())
for a in ('zero','bounded'):
    d = (p[a] - p['cycle'])
    print(f'{a} - cycle: per-seed {[round(x,2) for x in d]}  mean {d.mean():+.3f}')
s.to_csv('results/dsh/review_r3/neural_time_arms_summary.csv')
