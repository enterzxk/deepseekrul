"""Observability of the individual lifetime scale vs observation cutoff (all folds)."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
import run_fold as base, run_ida as ida
rows=[]
for subset in ['FD002','FD004']:
    folds,df=ida.outer_folds(subset); df=df.sort_values(['unit','cycle']).reset_index(drop=True)
    lives=df.groupby('unit').cycle.max()
    for k in range(5):
        train,valid=folds[k]
        tables=base.prior_tables(df,subset,train,valid,k)
        for (u,c),v in tables['FULL'].items():
            if u in valid and 50<=c<=200 and c%10==0:
                rows.append((subset,k,int(u),int(c),float(v),float(lives[u])))
d=pd.DataFrame(rows,columns=['subset','fold','unit','cutoff','mu','life'])
d.to_csv('observability.csv',index=False)
out={}
for subset,g in d.groupby('subset'):
    out[subset]={}
    for c,h in g.groupby('cutoff'):
        if len(h)<10: continue
        err=h.mu-h.life
        out[subset][int(c)]=dict(n=int(len(h)),mae=float(err.abs().mean()),rmse=float(np.sqrt((err**2).mean())),
            corr=float(np.corrcoef(h.mu,h.life)[0,1]),sd_mu=float(h.mu.std()),sd_life=float(h.life.std()))
    print('==',subset)
    for c,v in sorted(out[subset].items()):
        print('  cutoff %3d n=%4d MAE %5.2f RMSE %5.2f corr %.3f sd_mu %.1f (true sd %.1f)'%(
            c,v['n'],v['mae'],v['rmse'],v['corr'],v['sd_mu'],v['sd_life']))
Path('observability.json').write_text(json.dumps(out,indent=2))
