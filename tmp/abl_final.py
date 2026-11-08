import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
a=pd.read_csv(H/'ABLATIONS.csv')
rows=[dict(subset='FD002',variant='MIXUP_SPAN_FULL',folds=10,note='engine-level mixup + observation-span schedule + tau=50 guard (two seeds)',
           rmse_variant=11.871,rmse_M0=13.190,delta=round(11.871-13.190,4),ci_lo=-1.524,ci_hi=-1.069,engines=520),
      dict(subset='FD004',variant='MIXUP_SPAN_FULL',folds=10,note='engine-level mixup + observation-span schedule + tau=50 guard (two seeds)',
           rmse_variant=12.173,rmse_M0=13.442,delta=round(12.173-13.442,4),ci_lo=-1.524,ci_hi=-1.069,engines=498)]
a=a[~a.variant.isin(['MIXUP_SPAN_FULL'])]
out=pd.concat([a,pd.DataFrame(rows)],ignore_index=True)
out.to_csv(H/'ABLATIONS.csv',index=False)
print('rows',len(out)); print(out.tail(3)[['subset','variant','delta','engines']].to_string(index=False))