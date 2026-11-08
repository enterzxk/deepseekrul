"""Corrected bootstrap: cluster by ENGINE, carrying all seeds of a resampled engine."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
SEEDS=[101,202]; TAU=50; REPS=5000
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
def build(subset,seed):
    tag='' if seed==101 else str(seed)
    S=[];R=[]
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
        S.append(d.assign(fold=k,seed=seed)); R.append(pd.read_csv(a).assign(fold=k,seed=seed))
    return S,R
parts={}
for subset in ['FD002','FD004']:
    S=[];R=[]
    for seed in SEEDS:
        s,r=build(subset,seed)
        if s: S+=s; R+=r
    if not S: print(subset,'incomplete'); continue
    S=pd.concat(S,ignore_index=True); R=pd.concat(R,ignore_index=True)
    # per-engine per-seed squared errors, then aggregate over seeds inside the engine
    def eng(df,col):
        e=df[col]-df.y_true
        return pd.DataFrame({'unit':df.unit,'e2':e**2,'ae':e.abs(),'s':nasa(e),'b':e}).groupby('unit').mean()
    gs=eng(S,'y'); gr=eng(R,'y_pred')
    idx=gr.index.intersection(gs.index); gs=gs.loc[idx]; gr=gr.loc[idx]
    parts[subset]=(gs,gr)
    print('%s: %d engines  RMSE %.3f vs %.3f  %+.2f%%  MAE %.3f/%.3f  NASA %.2f/%.2f'%(
        subset,len(idx),np.sqrt(gs.e2.mean()),np.sqrt(gr.e2.mean()),100*(np.sqrt(gs.e2.mean())/np.sqrt(gr.e2.mean())-1),
        gs.ae.mean(),gr.ae.mean(),gs.s.mean(),gr.s.mean()))
if len(parts)==2:
    rng=np.random.default_rng(41); diffs=[]; rels=[]
    for _ in range(REPS):
        num=den=0.0; n=0
        for gs,gr in parts.values():
            pick=rng.integers(0,len(gs),(len(gs),))
            num+=gs.e2.to_numpy()[pick].sum(); den+=gr.e2.to_numpy()[pick].sum(); n+=len(pick)
        diffs.append(np.sqrt(num/n)-np.sqrt(den/n)); rels.append(100*(np.sqrt(num/n)/np.sqrt(den/n)-1))
    sa=np.sqrt(sum(g.e2.sum() for g,_ in parts.values())/sum(len(g) for g,_ in parts.values()))
    ra=np.sqrt(sum(r.e2.sum() for _,r in parts.values())/sum(len(r) for _,r in parts.values()))
    print()
    print('ENGINE-clustered stratified: %.3f vs %.3f delta %+.3f  CI [%+.3f, %+.3f]  rel %+.2f%%  CI [%+.2f%%, %+.2f%%]'%(
        sa,ra,sa-ra,np.percentile(diffs,2.5),np.percentile(diffs,97.5),100*(sa/ra-1),np.percentile(rels,2.5),np.percentile(rels,97.5)))