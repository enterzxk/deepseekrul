import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
for seed in [101,202]:
    a=pd.read_csv(HERE/'runs'/'FD004'/'0'/f'FULL_{seed}.csv')
    b=pd.read_csv(HERE/'runs3_w100_w100'/'FD004'/'0'/f'FULL_{seed}.csv')
    for u in [185,71,179]:
        ra=a[a.unit==u]; rb=b[b.unit==u]
        if len(ra)==0: continue
        print('seed %d unit %d: w50 bias %+.1f maxerr %+.1f | w100 bias %+.1f maxerr %+.1f'%(
            seed,u,(ra.y_pred-ra.y_true).mean(),(ra.y_pred-ra.y_true).abs().max(),
            (rb.y_pred-rb.y_true).mean(),(rb.y_pred-rb.y_true).abs().max()))