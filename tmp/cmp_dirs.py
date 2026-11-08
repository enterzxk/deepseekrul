"""Compare two run directories arm-by-arm, overall and by lifetime quartile."""
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
    g=pd.DataFrame({'u':d.unit,'e2':d.err**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
def paired(a,b,reps=5000):
    ga=pd.DataFrame({'u':a.unit,'e2':a.err**2}).groupby('u').e2.mean()
    gb=pd.DataFrame({'u':b.unit,'e2':b.err**2}).groupby('u').e2.mean()
    idx=ga.index.intersection(gb.index); av=ga.loc[idx].to_numpy(); bv=gb.loc[idx].to_numpy()
    rng=np.random.default_rng(99); d=rng.integers(0,len(idx),(reps,len(idx)))
    diffs=np.sqrt(av[d].mean(1))-np.sqrt(bv[d].mean(1))
    return float(np.sqrt(av.mean())-np.sqrt(bv.mean())),np.percentile(diffs,[2.5,97.5])
base_new, base_ref = sys.argv[1], sys.argv[2]
subset=sys.argv[3] if len(sys.argv)>3 else 'FD002'
arm_new=sys.argv[4] if len(sys.argv)>4 else 'M0'
arm_ref=sys.argv[5] if len(sys.argv)>5 else 'M0'
a=load(base_new,subset,arm_new); b=load(base_ref,subset,arm_ref)
if a is None or b is None:
    print('missing', a is None, b is None); raise SystemExit
a['err']=a.y_pred-a.y_true; b['err']=b.y_pred-b.y_true
print('==',subset,'new',base_new,arm_new,'n_folds',a.fold.nunique(),'| ref',base_ref,arm_ref)
print('  RMSE new %.3f  ref %.3f  delta %+.3f  CI %s'%(pooled(a),pooled(b),*[None]) if False else '')
d,ci=paired(a,b)
print('  RMSE new %.3f  ref %.3f  delta %+.3f [%+.3f,%+.3f]'%(pooled(a),pooled(b),d,ci[0],ci[1]))
lives=a.groupby('unit').apply(lambda x: x.y_true.max()+x.end_cycle.max(), include_groups=False)
q=pd.qcut(lives,4,labels=['Q1 short','Q2','Q3','Q4 long'])
for lab in q.cat.categories:
    u=lives.index[q==lab]
    sa=a[a.unit.isin(u)]; sb=b[b.unit.isin(u)]
    ea=sa.y_pred-sa.y_true; eb=sb.y_pred-sb.y_true
    print('  %-9s new %.2f (bias %+.1f)  ref %.2f (bias %+.1f)'%(lab,
        np.sqrt((ea**2).mean()),ea.mean(),np.sqrt((eb**2).mean()),eb.mean()))
