import pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
for subset in ['FD002','FD004']:
    for k in range(5):
        a=HERE/'runs'/subset/str(k)/'FULL_101.csv'; b=HERE/'runs3_w100_w100'/subset/str(k)/'FULL_101.csv'
        if not b.is_file(): continue
        A=pd.read_csv(a); B=pd.read_csv(b)
        late=A[A.end_cycle>=100]
        cov=late.merge(B[['unit','end_cycle']],on=['unit','end_cycle']).shape[0]/max(1,len(late))
        print('%s f%d: total %d, late %d, w100 preds %d, coverage %.3f, engines %d/%d'%(
            subset,k,len(A),len(late),len(B),cov,B.unit.nunique(),A.unit.nunique()))