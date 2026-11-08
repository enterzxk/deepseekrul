"""Window-level potential of the lifetime-scale estimate (diagnostic, training side)."""
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb"]
import run_fold as base, run_ida as ida

subset=sys.argv[1]; fold=int(sys.argv[2])
pred=pd.read_csv(f"/home/user/rulb/tim_20261105/full_20261105/runs/{subset}/{fold}/M0_101.csv")
folds,df=ida.outer_folds(subset)
df=df.sort_values(["unit","cycle"]).reset_index(drop=True)
train,valid=folds[fold]
tables=base.prior_tables(df,subset,train,valid,fold)
tbl=tables["FULL"]
pred["mu"]=[tbl[(u,min(int(t)//10*10,200))] for u,t in zip(pred.unit,pred.end_cycle)]
pred["prior_rul"]=pred["mu"]-pred.end_cycle
pred["pe"]=pred.y_pred
y=pred.y_true.to_numpy()

def pooled(err):
    g=pd.DataFrame({"u":pred.unit,"e":err**2}).groupby("u").mean()
    return float(np.sqrt(g.e.mean()))

print("==",subset,"fold",fold,"n",len(pred),"engines",pred.unit.nunique())
print("  PE %.3f  prior-RUL %.3f"%(pooled(pred.pe-y),pooled(pred.prior_rul-y)))
best=pd.concat([pred.pe,pred.prior_rul],axis=1).sub(pred.y_true,axis=0)
print("  window oracle %.3f   engine oracle %.3f"%(
    pooled(best.min(1).to_numpy()),
    float(np.sqrt(pd.DataFrame({"u":pred.unit,"a":(pred.pe-y)**2,"b":(pred.prior_rul-y)**2}).groupby("u").mean().min(1).mean()))))
for w in [0.1,0.2,0.3,0.5]:
    print("  fixed blend w=%.1f -> %.3f"%(w,pooled(w*pred.prior_rul+(1-w)*pred.pe-y)))
for lo,hi in [(0,100),(100,150),(150,400)]:
    m=(pred.end_cycle>=lo)&(pred.end_cycle<hi)
    if m.sum()==0: continue
    print("  age [%d,%d): n=%5d  PE %.3f  prior %.3f"%(lo,hi,m.sum(),pooled((pred.pe-y)[m]),pooled((pred.prior_rul-y)[m])))
for lo,hi in [(0,25),(25,50),(50,75),(75,100),(100,126)]:
    m=(pred.y_true>=lo)&(pred.y_true<hi)
    if m.sum()==0: continue
    print("  RUL [%d,%d): n=%5d  PE %.3f  prior %.3f"%(lo,hi,m.sum(),pooled((pred.pe-y)[m]),pooled((pred.prior_rul-y)[m])))
