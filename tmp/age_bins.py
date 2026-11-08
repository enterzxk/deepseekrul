"""Age-binned w50 vs w100 comparison: the falsifiable routing test."""
import numpy as np, pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
SEEDS=[101,202]
def nasa(e): return np.where(e>0,np.expm1(e/10.0),np.expm1(-e/13.0))
rows=[]
for subset in ['FD002','FD004']:
    for seed in SEEDS:
        tag='' if seed==101 else str(seed)
        for k in range(5):
            a=H/'runs'/subset/str(k)/f'M0_{seed}.csv'
            b=H/f'runs_v3_DM{tag}_w50'/subset/str(k)/f'M0_{seed}.csv'
            c=H/f'runs_v3_DM{tag}_w100'/subset/str(k)/f'M0_{seed}.csv'
            if not (a.is_file() and b.is_file() and c.is_file()): continue
            w50=pd.read_csv(b); w100=pd.read_csv(c)
            m=w50[['unit','end_cycle','y_true','y_pred']].merge(
                w100[['unit','end_cycle','y_pred']],on=['unit','end_cycle'],suffixes=('','_100'))
            m['subset']=subset; m['seed']=seed; m['fold']=k
            rows.append(m)
d=pd.concat(rows,ignore_index=True)
bins=[(0,50),(50,100),(100,150),(150,200),(200,400)]
print('%-6s %-12s %7s %8s %8s %8s'%('subset','age bin','n','w50','w100','w100-w50'))
for subset in ['FD002','FD004']:
    s=d[d.subset==subset]
    for lo,hi in bins:
        g=s[(s.end_cycle>=lo)&(s.end_cycle<hi)]
        if len(g)<20: continue
        e50=np.sqrt(np.mean((g.y_pred-g.y_true)**2)); e100=np.sqrt(np.mean((g.y_pred_100-g.y_true)**2))
        print('%-6s [%3d,%3d) %7d %8.3f %8.3f %+8.3f'%(subset,lo,hi,len(g),e50,e100,e100-e50))