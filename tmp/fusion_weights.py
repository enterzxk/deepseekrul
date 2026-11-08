"""Optimal fixed fusion weights (constrained LS, sum to one) and fold stability."""
import numpy as np, pandas as pd
from pathlib import Path
def load(subset,arm):
    b=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/{arm}_101.csv')
        if not p.is_file(): return None
        b.append(pd.read_csv(p).assign(fold=k))
    return pd.concat(b,ignore_index=True)
def pooled(d,err):
    g=pd.DataFrame({'u':d.unit,'e2':err**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
def solve(P,y):
    G=P.T@P; h=P.T@y
    A=np.block([[G, np.ones((3,1))],[np.ones((1,3)), np.zeros((1,1))]])
    b=np.r_[h,1.0]
    return np.linalg.lstsq(A,b,rcond=None)[0][:3]
for subset in ['FD002','FD004']:
    arms={a:load(subset,a) for a in ['M0','OPS','FULL']}
    if any(v is None for v in arms.values()):
        print(subset,'incomplete'); continue
    n=min(len(v) for v in arms.values())
    d=arms['M0'].iloc[:n]; y=d.y_true.to_numpy()
    P=np.stack([arms[a].y_pred.to_numpy()[:n] for a in ['M0','OPS','FULL']],axis=1)
    w=solve(P,y)
    print('==',subset,'optimal weights',np.round(w,3))
    for name,ww in [('equal',np.ones(3)/3),('optimal',w)]:
        print('   %-8s RMSE %.3f'%(name,pooled(d,P@ww-y)))
    for k in sorted(d.fold.unique()):
        s=(d.fold==k).to_numpy()
        wk=solve(P[s],y[s])
        print('   fold %d weights %s  rmse %.3f'%(k,np.round(wk,3),pooled(d[s],P[s]@wk-y[s])))
