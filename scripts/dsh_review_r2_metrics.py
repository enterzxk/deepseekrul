import numpy as np, pandas as pd, json
d = pd.read_csv('results/dsh/review_r2/C_predictions.csv')
print(f'rows={len(d)} engines={d.groupby(["dataset","engine"]).ngroups}')
def met(g):
    e = g.y_pred.to_numpy() - g.y_true.to_numpy()
    per = pd.DataFrame({'k': g.dataset + '_' + g.engine.astype(str), 'se': e**2, 'e': e, 'y': g.y_true.to_numpy()})
    macro = np.sqrt(per.groupby('k').se.mean()).mean()
    nasa = np.where(e < 0, np.exp(-e/13)-1, np.exp(e/10)-1).mean()
    return dict(n=len(g), rmse=float(np.sqrt((e**2).mean())), macro=float(macro),
                nasa=float(nasa), late=float((e>0).mean()),
                severe=float(((e>0) & (g.y_true.to_numpy() <= 30)).mean()))
rows = []
for (v, m), g in d.groupby(['view','model']):
    r = met(g); r.update(view=v, model=m); rows.append(r)
t = pd.DataFrame(rows)[['view','model','n','rmse','macro','nasa','late','severe']]
print(t.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
t.to_csv('results/dsh/review_r2/C_metrics.csv', index=False)

print(); print('paired V2 - V1 (engine-clustered bootstrap, 2000 resamples):')
out = {}
for m in ('ridge','rbf'):
    a = d[d.model==m].pivot_table(index=['dataset','engine','cutoff'], columns='view', values='y_pred')
    y = d[d.model==m].drop_duplicates(['dataset','engine','cutoff']).set_index(['dataset','engine','cutoff']).y_true
    a = a.join(y)
    keys = list(a.index)
    rng = np.random.default_rng(11); diffs = []
    for _ in range(2000):
        pick = rng.choice(len(keys), len(keys), replace=True)
        s = a.iloc[pick]
        r1 = np.sqrt(((s.v1 - s.y_true)**2).mean()); r2 = np.sqrt(((s.v2 - s.y_true)**2).mean())
        diffs.append(r2 - r1)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(np.sqrt(((a.v2-a.y_true)**2).mean()) - np.sqrt(((a.v1-a.y_true)**2).mean()))
    out[m] = dict(point=point, lo=float(lo), hi=float(hi))
    print(f'  {m:6s}: dRMSE={point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  excludes0={bool(hi<0 or lo>0)}')
json.dump(out, open('results/dsh/review_r2/C_paired_ci.json','w'), indent=2)
