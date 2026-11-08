import pandas as pd, numpy as np
from pathlib import Path
def load(subset,arm,seed=101):
    b=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/{arm}_{seed}.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True) if b else None
def pooled(d,err):
    g=pd.DataFrame({'u':d.unit,'e2':err**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
for subset in ['FD002','FD004']:
    m=load(subset,'M0'); o=load(subset,'OPS')
    if m is None or o is None: continue
    assert len(m)==len(o) and np.array_equal(m.unit,o.unit) and np.array_equal(m.end_cycle,o.end_cycle)
    y=m.y_true.to_numpy(); em=m.y_pred.to_numpy()-y; eo=o.y_pred.to_numpy()-y
    print('==',subset,'n',len(m))
    print('  M0 %.3f  OPS %.3f'%(pooled(m,em),pooled(o,eo)))
    for w in [0.2,0.4,0.5,0.6,0.8]:
        print('   fixed w=%.1f OPS -> %.3f'%(w,pooled(m,w*eo+(1-w)*em)))
    best=np.where(np.abs(em)<=np.abs(eo),em,eo)
    print('  window-level oracle %.3f'%pooled(m,best))
    # age-gated: use OPS below/above cutoffs
    for cut in [75,100,125,150]:
        g=np.where(m.end_cycle.to_numpy()>=cut,eo,em)
        print('   gate OPS when c>=%d -> %.3f'%(cut,pooled(m,g)))
