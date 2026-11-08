"""Does an observable evidence statistic predict which estimator is better?"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
import run_ida as ida
from scipy.stats import spearmanr

subset=sys.argv[1]
folds,df=ida.outer_folds(subset)
df=df.sort_values(["unit","cycle"]).reset_index(drop=True)
def load(arm):
    b=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/{arm}_101.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True) if b else None
m=load('M0'); o=load('OPS')
if m is None or o is None or len(m)!=len(o): 
    print('need both arms'); sys.exit()
rows=[]
for k in range(5):
    mk=m[m.fold==k]; ok=o[o.fold==k]
    if len(mk)==0: continue
    train,valid=folds[k]
    tr,va=ida.prep_fold(df,subset,k,train,valid)
    ev=va[2]; meta=va[3]
    assert np.array_equal(meta[:,0],mk.unit.to_numpy()) and np.array_equal(meta[:,1],mk.end_cycle.to_numpy())
    xva=va[0]
    slope=np.abs(np.polyfit(np.arange(xva.shape[1]),xva.mean(2),1)[0]) if False else None
    # per-window trend-to-noise: mean over sensors of |linear slope| / std
    t=np.arange(xva.shape[1],dtype=np.float64); t-=t.mean()
    s=np.einsum('ntf,t->nf',xva,t)/(t@t)              # (n,F) slope per sensor
    resid=xva-np.einsum('nf,t->ntf',s,t)-xva.mean(1)[:,None,:]
    noise=np.sqrt((resid**2).mean(1)).mean(1)
    trend=np.abs(s).mean(1)
    denr=trend/(noise+1e-6)
    rows.append(pd.DataFrame(dict(unit=meta[:,0],end_cycle=meta[:,1],
        deviation=-np.log(np.clip(ev[:,3],1e-12,None)), jitter=ev[:,5],
        spread=ev[:,4], denr=denr, fold=k)))
E=pd.concat(rows,ignore_index=True)
m=m.merge(E,on=['unit','end_cycle','fold']); o=o.merge(E,on=['unit','end_cycle','fold'])
y=m.y_true.to_numpy(); em=m.y_pred.to_numpy()-y; eo=o.y_pred.to_numpy()-y
better=(np.abs(em)<np.abs(eo)).astype(float)
print('==',subset,'n',len(m),'frac M0 better %.3f'%better.mean())
for col in ['deviation','jitter','spread','denr','end_cycle']:
    v=m[col].to_numpy()
    print('  %-10s spearman(with M0-better) %+.3f   corr(|eM0|-|eOPS|) %+.3f'%(
        col,spearmanr(v,better).statistic,np.corrcoef(v,np.abs(em)-np.abs(eo))[0,1]))
def pooled(err):
    g=pd.DataFrame({'u':m.unit,'e2':err**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
print('  M0 %.3f OPS %.3f fixed0.5 %.3f'%(pooled(em),pooled(eo),pooled(0.5*em+0.5*eo)))
for col in ['deviation','denr','jitter']:
    v=m[col].to_numpy(); q=np.quantile(v,[0.25,0.5,0.75])
    w=np.where(v<=q[0],1.0,np.where(v<=q[1],0.75,np.where(v<=q[2],0.5,0.25)))  # low evidence -> more OPS
    print('  quantile gate on %-9s -> %.3f'%(col,pooled(w*eo+(1-w)*em)))
