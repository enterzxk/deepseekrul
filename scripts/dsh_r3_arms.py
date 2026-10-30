"""R3-2/3: time-shortcut arms + view attribution, ridge, engine-level 5-fold, clustered CIs."""
import sys, warnings, json, pickle, hashlib
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
rows = pickle.load(open('results/dsh/review_r3/views_rows.pkl','rb'))
CAP = 125.0; WIN = 50
def pe(c, T1=200.0, T2=60.0):
    return np.array([np.sin(2*np.pi*c/T1), np.cos(2*np.pi*c/T1), np.sin(2*np.pi*c/T2), np.cos(2*np.pi*c/T2)])
# build arms: cycle index in v1 is at position 56 (after 56-dim v0)
for r in rows:
    base_v1 = r['v1']
    cyc = base_v1[56]
    rest = np.concatenate([base_v1[:56], base_v1[57:]])
    r['arm_cycle'] = base_v1
    r['arm_nocycle'] = rest
    r['arm_pe'] = np.concatenate([rest[:56], pe(cyc), rest[56:]])
keys = sorted({(r['dataset'], r['engine']) for r in rows})
rng = np.random.default_rng(2026); perm = rng.permutation(len(keys)); folds = np.array_split(perm, 5)
fold_of = {}
for k, f in enumerate(folds):
    for i in f: fold_of[keys[i]] = k
for r in rows: r['fold'] = fold_of[(r['dataset'], r['engine'])]
def ridge_fit(A, y, lam=10.0):
    return np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@(y-y.mean()))
ARMS = ['arm_cycle','arm_nocycle','arm_pe','v0','v1','v2']
recs = []; nfit = 0
for arm in ARMS:
    for k in range(5):
        tr = [r for r in rows if r['fold'] != k]; te = [r for r in rows if r['fold'] == k]
        Xtr = np.array([r[arm] for r in tr], float); Xte = np.array([r[arm] for r in te], float)
        ytr = np.array([r['rul'] for r in tr], float)
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1.0
        w = ridge_fit((Xtr-mu)/sd, ytr)
        p = ((Xte-mu)/sd)@w + ytr.mean(); nfit += 1
        for r, v in zip(te, p):
            recs.append(dict(arm=arm, dataset=r['dataset'], engine=r['engine'], cutoff=r['cutoff'],
                             fold=k, y_true=r['rul'], y_pred=float(np.clip(v, 0, CAP))))
print('fits:', nfit)
d = pd.DataFrame(recs); d.to_csv('results/dsh/review_r3/R3_preds.csv', index=False)
def metrics(g):
    e = g.y_pred.to_numpy() - g.y_true.to_numpy(); y = g.y_true.to_numpy()
    nz = np.where(e < 0, np.exp(-e/13)-1, np.exp(e/10)-1)
    return dict(n=len(g), rmse=float(np.sqrt((e**2).mean())), mae=float(np.abs(e).mean()),
                nasa_mean=float(nz.mean()), late=float((e>0).mean()), severe_e10=float((e>10).mean()),
                near_fail_late=float((e[y<=30]>0).mean()) if (y<=30).any() else float('nan'))
tab = []
for arm in ARMS:
    for scope in ('pooled','FD001','FD002'):
        g = d[d.arm==arm] if scope=='pooled' else d[(d.arm==arm)&(d.dataset==scope)]
        m = metrics(g); m.update(arm=arm, scope=scope); tab.append(m)
t = pd.DataFrame(tab)
t.to_csv('results/dsh/review_r3/R3_metrics.csv', index=False)
print(t[t.scope=='pooled'][['arm','n','rmse','mae','nasa_mean','late','severe_e10','near_fail_late']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print()
print(t[t.scope=='FD002'][['arm','rmse','late','severe_e10']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
# clustered CI for key contrasts
def contrast(a, b, scope='FD002', reps=2000):
    sub = d[(d.dataset==scope)] if scope!='pooled' else d
    A = sub[sub.arm==a].set_index(['dataset','engine','cutoff'])
    B = sub[sub.arm==b].set_index(['dataset','engine','cutoff'])
    idx = A.index.intersection(B.index)
    A = A.loc[idx]; B = B.loc[idx]
    keys = sorted({(i[0], i[1]) for i in idx}); rng2 = np.random.default_rng(7)
    def r(df): return float(np.sqrt(((df.y_pred-df.y_true)**2).mean()))
    diffs = []
    for _ in range(reps):
        pick = [keys[i] for i in rng2.integers(0, len(keys), len(keys))]
        sel = [i for i in idx if (i[0], i[1]) in set(pick)]
        diffs.append(r(B.loc[sel]) - r(A.loc[sel]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(point=float(r(B)-r(A)), lo=float(lo), hi=float(hi), engines=len(keys), excl0=bool(hi<0 or lo>0))
out = {}
for name, (a, b) in {'cycle_minus_nocycle': ('arm_cycle','arm_nocycle'),
                     'pe_minus_nocycle': ('arm_pe','arm_nocycle'),
                     'v2_minus_v1': ('v2','v1'),
                     'v1_minus_v0': ('v1','v0')}.items():
    for scope in ('pooled','FD002'):
        out[f'{name}|{scope}'] = contrast(a, b, scope)
        o = out[f'{name}|{scope}']
        print(f'{name:22s} {scope:7s} engines={o["engines"]:3d} dRMSE={o["point"]:+.4f} CI[{o["lo"]:+.4f},{o["hi"]:+.4f}] excl0={o["excl0"]}')
json.dump(out, open('results/dsh/review_r3/R3_contrasts.json','w'), indent=2)
print('script sha256:', hashlib.sha256(open(__file__,'rb').read()).hexdigest()[:16])
