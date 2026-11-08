import pandas as pd, numpy as np
from pathlib import Path
def rmse(d):
    e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean()
    return float(np.sqrt(g.e2.mean())), len(g)
for base in ['runs','runs3_w50','runs2']:
    for subset in ['FD002','FD004','FD001']:
        for fold in range(5):
            files=sorted(Path(f'{base}/{subset}/{fold}').glob('*_101.csv'))
            if not files: continue
            out=[]
            n=0
            for f in files:
                d=pd.read_csv(f); r,n=rmse(d); arm=f.stem.rsplit('_',1)[0]
                out.append('%s %.2f'%(arm,r))
            print('%s %s f%d (n=%d): %s'%(base,subset,fold,n,'  '.join(out)))
