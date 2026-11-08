"""Leave-one-fold-out inverse-variance fusion (honest, no evaluation-fold fitting)."""
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
    return np.linalg.lstsq(A,np.r_[h,1.0],rcond=None)[0][:3]
for subset in ['FD002','FD004']:
    arms={a:load(subset,a) for a in ['M0','OPS','FULL']}
    if any(v is None for v in arms.values()):
        print(subset,'incomplete'); continue
    n=min(len(v) for v in arms.values())
    d=arms['M0'].iloc[:n].reset_index(drop=True); y=d.y_true.to_numpy()
    P=np.stack([arms[a].y_pred.to_numpy()[:n] for a in ['M0','OPS','FULL']],axis=1)
    pred=np.zeros(n)
    for k in sorted(d.fold.unique()):
        tr=(d.fold!=k).to_numpy(); va=(d.fold==k).to_numpy()
        w=solve(P[tr],y[tr]); pred[va]=P[va]@w
        print('   fold %d weights %s'%(k,np.round(w,3)))
    print('==',subset,'LOFO fusion RMSE %.3f   FULL %.3f   equal %.3f'%(
        pooled(d,pred-y),pooled(d,P[:,2]-y),pooled(d,P.mean(1)-y)))
    print('   gain vs FULL %.2f%%'%(100*(1-pooled(d,pred-y)/pooled(d,P[:,2]-y))))
