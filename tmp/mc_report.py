import sys
from pathlib import Path
import pandas as pd, numpy as np
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rep(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return np.sqrt(g.e2.mean()), np.abs(e).mean(), e.mean()
for subset in ['FD002','FD004']:
    for fold in range(5):
        base=HERE/'measurement'/subset/str(fold)
        if not base.is_dir(): continue
        for arm in ['M0','FULL','NORM']:
            out=[]
            for c in ['clean','mc1','mc2']:
                p=base/f'{arm}_101_{c}.csv'
                if not p.is_file(): continue
                d=pd.read_csv(p); r=rep(d)
                out.append('%s %.3f/%.3f/%+.2f'%(c,*r))
            if out: print(subset,'f%d'%fold,arm,' | '.join(out))
