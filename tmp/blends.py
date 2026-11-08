import pandas as pd, numpy as np, itertools
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
    arms={a:load(subset,a) for a in ['M0','OPS','FULL']}
    arms={a:d for a,d in arms.items() if d is not None}
    if len(arms)<2: continue
    ks=list(arms)
    n=min(len(arms[a]) for a in ks)
    y=arms[ks[0]].y_true.to_numpy()[:n]; d0=arms[ks[0]].iloc[:n]
    print('==',subset,'arms',ks,'n',n)
    for a in ks: print('   %-5s %.3f'%(a,pooled(d0,arms[a].y_pred.to_numpy()[:n]-y)))
    for r in range(2,len(ks)+1):
        for combo in itertools.combinations(ks,r):
            pred=np.mean([arms[a].y_pred.to_numpy()[:n] for a in combo],axis=0)
            print('   blend %-14s %.3f'%('+'.join(combo),pooled(d0,pred-y)))
