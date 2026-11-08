"""Append the final candidate (mixup + span + guard) to RESULTS.csv once both seeds are in."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
SEEDS=[101,202]; TAU=50
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def eng(d,col):
    e=d[col]-d.y_true
    return pd.DataFrame({'u':d.unit.astype(str)+'_'+d.seed.astype(str),'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('u').mean()
rows=[]
for subset in ['FD002','FD004']:
    S=[];R=[];ns=set()
    for seed in SEEDS:
        tag='' if seed==101 else str(seed)
        for k in range(5):
            a=H/'runs'/subset/str(k)/f'FULL_{seed}.csv'
            b=H/f'runs_v3_DM{tag}_w50'/subset/str(k)/f'M0_{seed}.csv'
            c=H/f'runs_v3_DM{tag}_w100'/subset/str(k)/f'M0_{seed}.csv'
            if not (a.is_file() and b.is_file() and c.is_file()): continue
            w50=pd.read_csv(b); w100=pd.read_csv(c)
            d=w50.merge(w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],how='left',suffixes=('','_t'))
            late=(d.end_cycle>=100)&d.y_pred_t.notna()
            s=np.where(late,d.y_pred_t,d.y_pred)
            dis=(d.y_pred_t-d.y_pred).abs()>TAU
            d['y']=np.where(late&dis,d.y_pred,s)
            S.append(d.assign(fold=k,seed=seed)); R.append(pd.read_csv(a).assign(fold=k,seed=seed)); ns.add(seed)
    if not S: print(subset,'incomplete'); continue
    gs=eng(pd.concat(S,ignore_index=True),'y'); gr=eng(pd.concat(R,ignore_index=True),'y_pred')
    rows.append(dict(subset=subset,arm='MIXUP_SPAN',n_seeds=len(ns),rmse_mean=round(float(np.sqrt(gs.e2.mean())),4),rmse_sd=0.0,
        mae=round(float(gs.ae.mean()),4),nasa=round(float(gs.s.mean()),4),bias=round(float(gs.b.mean()),4),
        engines=int(len(gs)),windows=0,reference='FULL',delta_rmse=round(float(np.sqrt(gs.e2.mean())-np.sqrt(gr.e2.mean())),4),
        ci_lo='',ci_hi='',gain=round(float(np.sqrt(gs.e2.mean())/np.sqrt(gr.e2.mean())-1),4)))
    print('%s seeds %s: RMSE %.3f ref %.3f  %+.2f%%  MAE %.3f/%.3f  NASA %.2f/%.2f  bias %+.3f'%(
        subset,sorted(ns),np.sqrt(gs.e2.mean()),np.sqrt(gr.e2.mean()),100*(np.sqrt(gs.e2.mean())/np.sqrt(gr.e2.mean())-1),
        gs.ae.mean(),gr.ae.mean(),gs.s.mean(),gr.s.mean(),gs.b.mean()))
if rows:
    r=pd.read_csv(H/'RESULTS.csv')
    r=r[~r.arm.isin(['MIXUP_SPAN'])]
    out=pd.concat([r,pd.DataFrame(rows)],ignore_index=True)
    out.to_csv(H/'RESULTS.csv',index=False); print('RESULTS rows',len(out))