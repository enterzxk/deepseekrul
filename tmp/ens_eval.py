import numpy as np, pandas as pd, glob
for m in ['0','0.2']:
    fs=sorted(glob.glob('/tmp/pred_m%s_f*.csv'%m))
    if not fs: print('m'+m,'none'); continue
    dfs=[pd.read_csv(f).sort_values('unit').reset_index(drop=True) for f in fs]
    y=dfs[0].y_true.to_numpy()
    P=np.stack([d.y_pred.to_numpy() for d in dfs],axis=0)
    ens=P.mean(axis=0)
    rm=lambda p: float(np.sqrt(np.mean((p-y)**2)))
    print('mixup=%-4s models=%d  per-model mean RMSE %.3f   ENSEMBLE RMSE %.3f'%(m,len(fs),np.mean([rm(p) for p in P]),rm(ens)))