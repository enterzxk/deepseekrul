import numpy as np, pandas as pd
from pathlib import Path
def load(subset,arm):
    b=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/{arm}_101.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True) if b else None
def pooled(d,err):
    g=pd.DataFrame({'u':d.unit,'e2':err**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
for subset in ['FD002']:
    m=load(subset,'M0'); f=load(subset,'FULL')
    n=min(len(m),len(f)); m=m.iloc[:n]; f=f.iloc[:n]
    y=m.y_true.to_numpy(); em=m.y_pred.to_numpy()-y; ef=f.y_pred.to_numpy()-y
    c=m.end_cycle.to_numpy()
    print('==',subset,'n',n,'folds M0',sorted(m.fold.unique()),'FULL',sorted(f.fold.unique()))
    print('  M0 %.3f FULL %.3f'%(pooled(m,em),pooled(m,ef)))
    for cut in [80,100,120,150]:
        a=np.where(c<cut,ef,em)   # FULL early, M0 late
        b=np.where(c<cut,em,ef)   # M0 early, FULL late
        print('  FULL if c<%3d -> %.3f   |  M0 if c<%3d -> %.3f'%(cut,pooled(m,a),cut,pooled(m,b)))
    # age-resolved comparison
    for lo,hi in [(50,100),(100,150),(150,400)]:
        s=(c>=lo)&(c<hi)
        print('  age [%d,%d): M0 %.2f FULL %.2f  n=%d'%(lo,hi,pooled(m[s],em[s]),pooled(m[s],ef[s]),s.sum()))
