"""R2-D: does route 1's precondition hold? partial effects of drift mu and exposure E on health, controlling RUL."""
import sys, warnings, json
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS

CUTS = (60, 100, 140); WIN = 50; CAP = 125.0
cols = list(SELECTED_SENSOR_COLUMNS)
rows = []
for ds in ('FD001', 'FD002'):
    df = load_cmapss_train(Path('data/CMAPSS'), ds)
    life = df.groupby('unit')['cycle'].max()
    for u, g in df.groupby('unit'):
        T = int(life[u]); g = g.sort_values('cycle').reset_index(drop=True)
        for c in CUTS:
            if T <= c or c < WIN: continue
            w = g.iloc[c-WIN:c][cols].to_numpy(float)
            # condition-corrected health level: z-score each sensor by its first 20 cycles, then mean
            early = g.iloc[:20][cols].to_numpy(float)
            sd = early.std(0); sd[sd < 1e-9] = 1.0
            z = (w - early.mean(0)) / sd
            h = float(z[-1].mean())
            mu = float(np.polyfit(np.arange(WIN, dtype=float), z.mean(1), 1)[0])
            cyc = g.iloc[max(0, c-100):c]
            ent = 0.0
            rows.append(dict(dataset=ds, engine=int(u), cutoff=c, h=h, mu=mu, E=ent,
                             rul=min(float(T - c), CAP)))
d = pd.DataFrame(rows)
print(f'observations={len(d)} engines={d.groupby(["dataset","engine"]).ngroups}')
eng = d[['dataset','engine']].drop_duplicates().to_numpy()
rng = np.random.default_rng(2026); perm = rng.permutation(len(eng)); folds = np.array_split(perm, 5)
fd = {}
for k, f in enumerate(folds):
    for i in f: fd[(eng[i][0], eng[i][1])] = k
d['fold'] = [fd[(a, b)] for a, b in zip(d.dataset, d.engine)]

res = {'with_rul': [], 'without_rul': []}
nfit = 0
for spec in ('with_rul', 'without_rul'):
    for k in range(5):
        tr = d[d.fold != k]
        X = tr[['mu','E'] + (['rul'] if spec == 'with_rul' else [])].to_numpy(float)
        X = np.column_stack([np.ones(len(X)), X])
        y = tr.h.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        nfit += 1
        res[spec].append({'fold': k, 'mu': float(beta[1]), 'E': float(beta[2])})
print('outer fits:', nfit); assert nfit == 10
json.dump(res, open('results/dsh/review_r2/D_precondition.json','w'), indent=2)
for spec in res:
    mus = np.array([r['mu'] for r in res[spec]])
    print(f'  {spec:12s} mu coef mean={mus.mean():+.4f} sd={mus.std():.4f} all-positive={bool((mus>0).all())}')
print()
mu_with = np.array([r['mu'] for r in res['with_rul']])
print('precondition (mu partial effect, controlling RUL): positive folds =', int((mu_with>0).sum()), '/5')
print('note: E was not constructed (constant 0) - see report; the E part of the test is VOID')
