import json
import numpy as np, pandas as pd
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105')
res=json.loads((HERE/'official_fullbudget_results.json').read_text())
res=res.get('subsets',res)
ok=True
for subset in ['FD001','FD002','FD003','FD004']:
    y=np.loadtxt(f'/home/user/rulb/data/CMAPSS/RUL_{subset}.txt').reshape(-1).clip(0,125)
    for arm in ['M0','OPS','FULL','CET']:
        preds=[]
        for seed in [101,202,303]:
            d=pd.read_csv(HERE/'official_fullbudget'/subset/f'{arm}_{seed}.csv')
            preds.append(d.y_pred.to_numpy())
        ens=np.mean(preds,axis=0)
        rmse=float(np.sqrt(((ens-y)**2).mean()))
        stored=res[subset]['ensemble'][arm]['rmse']
        same=abs(rmse-stored)<1e-6
        ok&=same
        print(f'{subset} {arm}: recomputed {rmse:.4f} stored {stored:.4f} {"OK" if same else "MISMATCH"}')
print('ALL MATCH' if ok else 'MISMATCH FOUND')
