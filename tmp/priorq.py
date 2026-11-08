import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
for subset in ["FD002","FD004"]:
    rows=[]
    for k in range(5):
        p=Path(f"/home/user/rulb/experiments/ida/gate0/{subset}/M1p_f{k}_s101/val_predictions.csv")
        if p.is_file(): rows.append(pd.read_csv(p))
    d=pd.concat(rows,ignore_index=True)
    lives=pd.read_csv(f"/home/user/rulb/data/CMAPSS/train_{subset}.txt",sep=r"\s+",header=None,usecols=[0,1],names=["unit","cycle"]).groupby("unit").cycle.max()
    d["life"]=d.unit.map(lives)
    mu=d.mu_T.to_numpy(float); T=d["life"].to_numpy(float); c=d.end_cycle.to_numpy(float); y=d.y_true.to_numpy(float)
    eng=d.groupby("unit").agg(mu=("mu_T","first"),life=("life","first"))
    m=eng["mu"].to_numpy(float); t=eng["life"].to_numpy(float)
    b=np.cov(m,t)[0,1]/np.var(m); a=t.mean()-b*m.mean()
    print("==",subset,"engines",len(eng))
    print("  prior sd %.1f  true sd %.1f  corr %.3f  spearman %.3f"%(m.std(),t.std(),np.corrcoef(m,t)[0,1],spearmanr(m,t).statistic))
    print("  RMSE(mu-T) %.2f  best linear b=%.3f a=%.1f -> RMSE %.2f"%(np.sqrt(((m-t)**2).mean()),b,a,np.sqrt((((a+b*m)-t)**2).mean())))
    for k in [0.8,1.0,1.2,1.35,1.5]:
        mk=m.mean()+k*(m-m.mean())
        print("    expand k=%.2f -> RMSE %.2f"%(k,np.sqrt(((mk-t)**2).mean())))
    prior_rul=mu-c
    g=pd.DataFrame({"u":d.unit,"e_prior":(prior_rul-y)**2,"e_pe":(d.y_pred-y)**2}).groupby("u").mean()
    print("  engine-pooled RMSE prior-RUL %.3f  PE %.3f"%(np.sqrt(g.e_prior.mean()),np.sqrt(g.e_pe.mean())))
    for w in [0.0,0.1,0.2,0.3,0.5]:
        blend=w*prior_rul+(1-w)*d.y_pred
        gg=pd.DataFrame({"u":d.unit,"e":(blend-y)**2}).groupby("u").mean()
        print("    blend w=%.1f -> %.3f"%(w,np.sqrt(gg.e.mean())))
