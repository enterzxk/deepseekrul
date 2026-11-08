"""Compare observation windows on the common window set (end_cycle >= 100)."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
def load(base,subset,arm,seed=101):
    b=[]
    for k in range(5):
        p=Path(f'{base}/{subset}/{k}/{arm}_{seed}.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True) if b else None
def pooled(d):
    g=pd.DataFrame({'u':d.unit,'e2':d.err**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean()))
def report(d,name,minc):
    d=d[d.end_cycle>=minc].copy(); d['err']=d.y_pred-d.y_true
    lives=d.groupby('unit').apply(lambda x: x.y_true.max()+x.end_cycle.max(), include_groups=False)
    q=pd.qcut(lives,4,labels=['Q1','Q2','Q3','Q4'])
    out=[]
    for lab in ['Q1','Q2','Q3','Q4']:
        u=lives.index[q==lab]
        sub=d[d.unit.isin(u)]
        g=pd.DataFrame({'u':sub.unit,'e2':sub.err**2,'e':sub.err}).groupby('u').mean()
        out.append('%s %.2f/%+.1f'%(lab,np.sqrt(g.e2.mean()),g.e.mean()))
    print('  %-18s RMSE %.3f  n_eng %d  %s'%(name,pooled(d),d.unit.nunique(),' '.join(out)))
for subset in sys.argv[1].split(','):
    for minc in [100]:
        print('==',subset,'windows with end_cycle>=%d'%minc)
        for base,arm,name in [('runs','M0','M0 w50'),('runs3_w100','M0','M0 w100'),('runs','OPS','OPS w50'),('runs','FULL','FULL w50')]:
            d=load(base,subset,arm)
            if d is None: continue
            report(d,name,minc)
