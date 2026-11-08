import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
import run_ida as ida
from scipy.stats import spearmanr
subset=sys.argv[1]
folds,df=ida.outer_folds(subset); df=df.sort_values(["unit","cycle"]).reset_index(drop=True)
def load(arm):
    b=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/{arm}_101.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True) if b else None
m=load('M0'); o=load('OPS'); f=load('FULL')
rows=[]
for k in range(5):
    if (m.fold==k).sum()==0: continue
    train,valid=folds[k]
    tr,va=ida.prep_fold(df,subset,k,train,valid)
    xva=va[0]; ev=va[2]; meta=va[3]
    t=np.arange(xva.shape[1],dtype=np.float64); t-=t.mean()
    s=np.einsum('ntf,t->nf',xva,t)/(t@t)
    resid=xva-np.einsum('nf,t->ntf',s,t)-xva.mean(1)[:,None,:]
    noise=np.sqrt((resid**2).mean(1)).mean(1)
    rows.append(pd.DataFrame(dict(unit=meta[:,0],end_cycle=meta[:,1],dev=-np.log(np.clip(ev[:,3],1e-12,None)),
        jitter=ev[:,5],denr=np.abs(s).mean(1)/(noise+1e-6))))
E=pd.concat(rows,ignore_index=True).groupby('unit').mean()
def eng(d):
    e=d.y_pred-d.y_true
    return np.sqrt(pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').e2.mean())
em=eng(m); units=em.index.to_numpy()
out=pd.DataFrame({'unit':units,'rmse_M0':em.to_numpy()})
if o is not None: out['rmse_OPS']=eng(o).reindex(units).to_numpy()
if f is not None: out['rmse_FULL']=eng(f).reindex(units).to_numpy()
out=out.merge(E.reset_index(),on='unit',how='left')
out['life']=m.groupby('unit').apply(lambda x: x.y_true.max()+x.end_cycle.max(), include_groups=False).reindex(units).to_numpy()
print('==',subset,'engines',len(out))
for c in ['rmse_M0','rmse_OPS','rmse_FULL']:
    if c in out: print('  mean per-engine %s %.3f'%(c,out[c].mean()))
if 'rmse_OPS' in out:
    out['d']=out.rmse_M0-out.rmse_OPS
    for col in ['dev','jitter','denr','life']:
        print('  %-7s spearman(d) %+.3f  pearson %+.3f'%(col,spearmanr(out[col],out.d).statistic,np.corrcoef(out[col],out.d)[0,1]))
