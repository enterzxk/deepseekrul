import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def load(arm,subset='FD002'):
    b=[]
    for k in range(5):
        p=HERE/'runs'/subset/str(k)/f'{arm}_101.csv'
        if not p.is_file(): return None
        b.append(pd.read_csv(p))
    return pd.concat(b,ignore_index=True)
m=load('M0'); o=load('OPS'); f=load('FULL')
n=min(len(m),len(o),len(f)); y=m.y_true.to_numpy()[:n]
E={a:(v.y_pred.to_numpy()[:n]-y) for a,v in [('M0',m),('OPS',o),('FULL',f)]}
print('FD002 window-level error correlations (n=%d)'%n)
for a in E:
    for b in E:
        if a<b: print('  corr(%s,%s) = %+.3f'%(a,b,np.corrcoef(E[a],E[b])[0,1]))
# engine-level
df=pd.DataFrame({'u':m.unit.to_numpy()[:n], **{a:E[a] for a in E}})
g=df.groupby('u').mean()
print('engine-level:')
for a in E:
    for b in E:
        if a<b: print('  corr(%s,%s) = %+.3f'%(a,b,np.corrcoef(g[a],g[b])[0,1]))
