"""Prior accuracy vs FULL gain across the four subsets."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
import run_fold as base, run_ida as ida
out={}
for subset in ['FD001','FD003']:
    folds,df=ida.outer_folds(subset); df=df.sort_values(['unit','cycle']).reset_index(drop=True)
    lives=df.groupby('unit').cycle.max(); rows=[]
    for k in range(5):
        train,valid=folds[k]
        t=base.prior_tables(df,subset,train,valid,k)
        for (u,c),v in t['FULL'].items():
            if u in valid and c in (100,150):
                rows.append((int(u),int(c),float(v),float(lives[u])))
    d=pd.DataFrame(rows,columns=['unit','cutoff','mu','life'])
    out[subset]={}
    for c,g in d.groupby('cutoff'):
        err=g.mu-g.life
        out[subset][int(c)]=dict(n=int(len(g)),mae=float(err.abs().mean()),
            corr=float(np.corrcoef(g.mu,g.life)[0,1]))
    print(subset, {k:{kk:round(vv,3) for kk,vv in v.items()} for k,v in out[subset].items()})
Path('prior_quality_single.json').write_text(json.dumps(out,indent=2))
