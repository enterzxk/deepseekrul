"""A5: frozen severe-late (e>10) from existing predictions; rename the old proxy."""
import numpy as np, pandas as pd, json
out = {}
# C predictions
c = pd.read_csv('results/dsh/review_r2/C_predictions.csv')
c['e'] = c.y_pred - c.y_true
rows = []
for (v, m), g in c.groupby(['view','model']):
    e = g.e.to_numpy(); y = g.y_true.to_numpy()
    rows.append(dict(view=v, model=m, n=len(g),
        rmse=float(np.sqrt((e**2).mean())),
        late=float((e>0).mean()),
        severe_e_gt_10_count=int((e>10).sum()),
        severe_e_gt_10_fraction=float((e>10).mean()),
        nasa_mean=float(np.where(e<0, np.exp(-e/13)-1, np.exp(e/10)-1).mean()),
        nasa_sum=float(np.where(e<0, np.exp(-e/13)-1, np.exp(e/10)-1).sum()),
        old_proxy_e_gt0_and_rul_le_30=float(((e>0)&(y<=30)).mean())))
t = pd.DataFrame(rows)
t.to_csv('results/dsh/review_r3/A5_metrics_frozen_severe.csv', index=False)
print(t[['view','model','n','rmse','late','severe_e_gt_10_fraction','old_proxy_e_gt0_and_rul_le_30']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# band-wise late rate with explicit denominators (C, v1/ridge)
g = c[(c.view=='v1')&(c.model=='ridge')]
bands = [(0,30),(30,60),(60,90),(90,125)]
print()
print(f"{'band':>9s} {'n':>4s} {'late':>6s} {'severe>10':>10s}")
for lo,hi in bands:
    s = g[(g.y_true>=lo)&(g.y_true<hi)]
    if len(s)==0: continue
    print(f'{str(lo)+"-"+str(hi):>9s} {len(s):4d} {(s.e>0).mean():6.3f} {(s.e>10).mean():10.3f}')
out['C_v1_ridge'] = t[(t.view=='v1')&(t.model=='ridge')].to_dict('records')

# E replay
e = pd.read_csv('results/dsh/review_r3/E_predictions.csv'); e['err'] = e.y_pred - e.y_true
for spec in ('S1','S2'):
    s = e[e.spec==spec]; er = s.err.to_numpy()
    print(f'E {spec}: rmse={np.sqrt((er**2).mean()):.4f} late={(er>0).mean():.3f} severe_gt10={(er>10).mean():.3f}')
    out['E_'+spec] = dict(rmse=float(np.sqrt((er**2).mean())), late=float((er>0).mean()), severe_gt10=float((er>10).mean()))
json.dump(out, open('results/dsh/review_r3/A5_summary.json','w'), indent=2, default=str)
