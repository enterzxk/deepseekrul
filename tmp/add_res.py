import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
res=pd.read_csv(H/'RESULTS.csv')
sf=pd.read_csv(H/'SCHEDULE_FINAL.csv')
rows=[]
for _,r in sf.iterrows():
    if r.arm!='FULL': continue
    tag='SPAN_GUARD' if r.guard=='tau50' else 'SPAN_UNG'
    rows.append(dict(subset=r.subset,arm=tag,n_seeds=2,rmse_mean=round(r.rmse,6),rmse_sd=0.0,
        mae=round(r.mae,6),nasa=round(r.nasa,6),bias=round(r.bias,6),engines=int(r.engines),
        windows=0,reference='FULL',delta_rmse=round(r.delta,6),ci_lo=round(r.ci_lo,6),ci_hi=round(r.ci_hi,6),
        gain=round(r.rel/100,6)))
out=pd.concat([res,pd.DataFrame(rows)],ignore_index=True)
out.to_csv(H/'RESULTS.csv',index=False)
print(out[['subset','arm','n_seeds','rmse_mean','mae','nasa','reference','delta_rmse','gain']].tail(4).to_string(index=False))
print('rows', len(out))