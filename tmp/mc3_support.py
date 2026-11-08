"""MC3: window-level error stratified by the training-fold support of the terminal condition."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
import run_ida as ida
from sklearn.cluster import KMeans
subset=sys.argv[1] if len(sys.argv)>1 else 'FD002'
folds,df=ida.outer_folds(subset); df=df.sort_values(['unit','cycle']).reset_index(drop=True)
keys=[]
for k in range(5):
    train,valid=folds[k]
    tr=df[df.unit.isin(train)]
    km=KMeans(n_clusters=6,random_state=7,n_init=10).fit(tr[['op_1','op_2','op_3']].to_numpy(np.float64))
    counts=np.bincount(km.labels_,minlength=6)
    va=df[df.unit.isin(valid)].copy()
    va['lab']=km.predict(va[['op_1','op_2','op_3']].to_numpy(np.float64))
    va['support']=counts[va.lab.to_numpy()]
    va['fold']=k
    keys.append(va[['unit','cycle','fold','lab','support']].rename(columns={'cycle':'end_cycle'}))
K=pd.concat(keys,ignore_index=True)
def pooled(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
for arm in ['M0','OPS','FULL']:
    blocks=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/{arm}_101.csv')
        if not p.is_file(): blocks=None; break
        blocks.append(pd.read_csv(p).assign(fold=k))
    if blocks is None: continue
    d=pd.concat(blocks,ignore_index=True).merge(K,on=['unit','end_cycle','fold'],how='inner')
    if len(d)==0: print(arm,'no match'); continue
    q=pd.qcut(d.support,3,labels=['low','mid','high'],duplicates='drop')
    out=[]
    for lab in q.cat.categories:
        s=d[q==lab]
        out.append('%s(n_win=%d) %.2f'%(lab,len(s),pooled(s)))
    print('  %-5s by condition support: %s'%(arm,'  '.join(out)))
