import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
r=pd.read_csv(H/'RESULTS.csv')
before=len(r)
r=r.drop_duplicates(subset=['subset','arm'],keep='last')
r.to_csv(H/'RESULTS.csv',index=False)
print('before',before,'after',len(r))
print(r[['subset','arm','rmse_mean','reference','gain']].to_string(index=False))