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
    m=load(subset,'M0'); o=load(subset,'OPS'); f=load(subset,'FULL')
    n=min(len(m),len(o),len(f)); m=m.iloc[:n]; o=o.iloc[:n]; f=f.iloc[:n]
    y=m.y_true.to_numpy(); e0=m.y_pred.to_numpy()-y; e1=o.y_pred.to_numpy()-y; e2=f.y_pred.to_numpy()-y
    print('==',subset,'n',n)
    for name,g in [('|M0-OPS|',np.abs(m.y_pred-o.y_pred)),('|M0-FULL|',np.abs(m.y_pred-f.y_pred)),('|OPS-FULL|',np.abs(o.y_pred-f.y_pred))]:
        d=np.abs(e0)-np.abs(e1)
        print('  %-11s corr with (|eM0|-|eOPS|) %+.3f'%(name,np.corrcoef(g,d)[0,1]))
    # gate: where M0 and OPS agree, trust M0; where disagree, trust OPS
    for th in [5,10,15,20]:
        g=np.abs(m.y_pred-o.y_pred)
        w=np.where(g>th,1.0,0.0)
        print('  disagreement gate th=%2d (w=1 -> OPS) -> %.3f  frac %.2f'%(th,pooled(m,w*e1+(1-w)*e0),w.mean()))
    print('  M0 %.3f OPS %.3f FULL %.3f blend0.5(M0,OPS) %.3f'%(pooled(m,e0),pooled(m,e1),pooled(m,e2),pooled(m,0.5*e0+0.5*e1)))
