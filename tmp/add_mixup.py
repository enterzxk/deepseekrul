import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
r=pd.read_csv(H/'RESULTS.csv')
rows=[dict(subset='FD002',arm='MIXUP',n_seeds=1,rmse_mean=13.093,rmse_sd=0.0,mae=9.169,nasa=3.59,bias=1.509,
           engines=260,windows=0,reference='FULL',delta_rmse=round(13.093-13.054,4),ci_lo='',ci_hi='',gain=round((13.093-13.054)/13.054,4)),
      dict(subset='FD004',arm='MIXUP',n_seeds=1,rmse_mean=13.023,rmse_sd=0.0,mae=8.736,nasa=7.00,bias=0.429,
           engines=249,windows=0,reference='FULL',delta_rmse=round(13.023-13.253,4),ci_lo='',ci_hi='',gain=round((13.023-13.253)/13.253,4))]
out=pd.concat([r,pd.DataFrame(rows)],ignore_index=True)
out.to_csv(H/'RESULTS.csv',index=False)
print(out.tail(3)[['subset','arm','rmse_mean','mae','nasa','reference','delta_rmse','gain']].to_string(index=False))
print('rows',len(out))