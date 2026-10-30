import json, glob
import numpy as np, pandas as pd
rows = [json.load(open(f)) for f in sorted(glob.glob('experiments/r3_scale/*/metrics.json'))]
t = pd.DataFrame(rows)
print('runs:', len(t), '| train windows:', t.n_train.iloc[0])
agg = t.groupby('arm').agg({'rmse':['mean','std'],'late':['mean','std'],'nasa_mean':'mean','severe_e10':'mean'}).round(3)
print(agg.to_string())
print()
p = t.pivot(index='seed', columns='arm', values='rmse')
print(p.round(3).to_string())
for a in ('none','oracle'):
    if a in p: d = p[a]-p['cycle']; print(f'{a} - cycle: {[round(x,2) for x in d]} mean {d.mean():+.3f}')
print()
print('by lifespan RMSE (mean over seeds):')
bands = ['short','mid','long']
for arm in t.arm.unique():
    s = t[t.arm==arm]
    print(' ', arm.ljust(7), {b: round(float(np.mean([r['by_lifespan'][b]['rmse'] for r in s if b in r['by_lifespan']])),2) for b in bands})
print()
print('by lifespan BIAS (mean over seeds):')
for arm in t.arm.unique():
    s = t[t.arm==arm]
    print(' ', arm.ljust(7), {b: round(float(np.mean([r['by_lifespan'][b]['bias'] for r in s if b in r['by_lifespan']])),2) for b in bands})
t.to_csv('results/dsh/review_r3/scale_arms.csv', index=False)
