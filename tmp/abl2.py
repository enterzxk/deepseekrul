import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
s=pd.read_csv(H/'SCHEDULE_FINAL.csv')
rows=[]
for _,r in s.iterrows():
    tag='SPAN_GUARD' if r.guard=='tau50' else 'SPAN_UNG'
    note=('observation-span schedule with tau=50 tail guard' if r.guard=='tau50' else 'observation-span schedule, unguarded')
    rows.append(dict(subset=r.subset,variant=f'{tag}_{r.arm}',folds=5,note=note,
        rmse_variant=round(r.rmse,4),rmse_M0=round(r.ref_rmse,4),delta=round(r.delta,4),
        ci_lo=round(r.ci_lo,4),ci_hi=round(r.ci_hi,4),engines=int(r.engines)))
rows.append(dict(subset='FD002',variant='CAPACITY_719k',folds=1,
    note='d128x2 + multiscale conv + extra layer + attention pooling, fold 2, M0, w50',
    rmse_variant=16.62,rmse_M0=13.732,delta=round(16.62-13.732,4),ci_lo='',ci_hi='',engines=52))
rows.append(dict(subset='FD002',variant='CAPACITY_258k',folds=1,
    note='d96x1 + modules + dropout0.3/wd1e-3, fold 2, M0, w50',
    rmse_variant=16.605,rmse_M0=13.732,delta=round(16.605-13.732,4),ci_lo='',ci_hi='',engines=52))
out=pd.DataFrame(rows)
out.to_csv(H/'ABLATIONS.csv',mode='a',header=False,index=False)
print(out.to_string(index=False))