import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
import run_fold as base, run_ida as ida
subset=sys.argv[1]; fold=int(sys.argv[2])
folds,df=ida.outer_folds(subset)
df=df.sort_values(["unit","cycle"]).reset_index(drop=True)
train,valid=folds[fold]
tables=base.prior_tables(df,subset,train,valid,fold)
lives=df.groupby("unit").cycle.max()
rows=[]
for (u,c),v in tables["FULL"].items():
    if u in valid and c in (50,100,150,200):
        rows.append((u,c,v,float(lives[u])))
d=pd.DataFrame(rows,columns=["unit","cutoff","mu","life"])
for c,g in d.groupby("cutoff"):
    if len(g)<3:
        print("cutoff",c,"n",len(g)); continue
    err=g["mu"].to_numpy()-g["life"].to_numpy()
    print("%s f%d cutoff %3d n=%3d MAE %.2f RMSE %.2f corr %.3f sd_mu %.1f sd_life %.1f"%(
        subset,fold,c,len(g),np.abs(err).mean(),np.sqrt((err**2).mean()),
        np.corrcoef(g["mu"],g["life"])[0,1],g["mu"].std(),g["life"].std()))
