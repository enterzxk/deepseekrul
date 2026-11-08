"""Pre-registered tail guard (PROTOCOL 8.8): if |pred_w100 - pred_w50| > tau, keep w50."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
SEEDS=[101,202]; TAUS=[None,30,50,75]
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def load(subset,fold,arm,seed):
    a=H/'runs'/subset/str(fold)/f'{arm}_{seed}.csv'
    b=H/'runs3_w100_w100'/subset/str(fold)/f'{arm}_{seed}.csv'
    if not (a.is_file() and b.is_file()): return None,None
    return pd.read_csv(a), pd.read_csv(b)
def apply(w50,w100,tau):
    d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
    late=(d.end_cycle>=100)&d.y_pred_t.notna()
    d['y_pred_sched']=np.where(late,d.y_pred_t,d.y_pred)
    if tau is None:
        d['y_pred_g']=d.y_pred_sched
    else:
        dis=(d.y_pred_t-d.y_pred).abs()>tau
        d['y_pred_g']=np.where(late&dis,d.y_pred,d.y_pred_sched)
    return d
def met(d,col):
    e=d[col]-d.y_true
    g=pd.DataFrame({'u':d.unit.astype(str)+'_'+d.seed.astype(str),'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
    return dict(rmse=float(np.sqrt(g.e2.mean())),mae=float(g.ae.mean()),nasa=float(g.s.mean()),bias=float(g.b.mean())), g
rows=[]
for subset in ['FD002','FD004']:
    for tau in TAUS:
        S=[];R=[]
        for seed in SEEDS:
            for k in range(5):
                a,b=load(subset,k,'FULL',seed)
                if a is None: continue
                S.append(apply(a,b,tau).assign(fold=k,seed=seed))
                R.append(a.assign(fold=k,seed=seed))
        S=pd.concat(S,ignore_index=True); R=pd.concat(R,ignore_index=True)
        ms,_=met(S,'y_pred_g'); mr,_=met(R,'y_pred')
        rows.append(dict(subset=subset,tau=('none' if tau is None else tau),rmse=ms['rmse'],mae=ms['mae'],nasa=ms['nasa'],bias=ms['bias'],
                         ref_rmse=mr['rmse'],ref_mae=mr['mae'],ref_nasa=mr['nasa'],ref_bias=mr['bias'],
                         rel=100*(ms['rmse']/mr['rmse']-1)))
out=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(out.to_string(index=False,float_format=lambda x:'%.3f'%x))
out.to_csv(H/'GUARD.csv',index=False)