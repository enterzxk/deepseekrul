"""Stratified error analysis on saved per-window predictions (read-only)."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

def load(base, subset, arm, seed, pattern):
    blocks=[]
    for k in range(5):
        p=Path(base)/subset/str(k)/pattern.format(arm=arm,seed=seed)
        if not p.is_file(): return None
        d=pd.read_csv(p); d["fold"]=k; blocks.append(d)
    return pd.concat(blocks,ignore_index=True)

def strat_rmse(d, mask):
    e=d.loc[mask,"y_pred"]-d.loc[mask,"y_true"]
    g=pd.DataFrame({"u":d.loc[mask,"unit"],"e2":e**2}).groupby("u").e2.mean()
    return float(np.sqrt(g.mean())), int(g.size), int(mask.sum())

def main():
    base, pattern, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
    for subset in ["FD002","FD004"]:
        print("==",subset)
        for arm in ["M0","OPS","FULL","NORM"]:
            rows=[]
            for seed in seeds:
                d=load(base,subset,arm,seed,pattern)
                if d is None: continue
                allr,_,_=strat_rmse(d, np.ones(len(d),bool))
                late,nu,nw=strat_rmse(d, d.y_true<125)
                rows.append((seed,allr,late,nu,nw))
            if not rows: continue
            a=np.array([[r[1],r[2]] for r in rows])
            print("  %-6s all-window mean %.3f | late(y<125) mean %.3f  (n_units=%d, n_win=%d)  per-seed late: %s"%(
                arm, a[:,0].mean(), a[:,1].mean(), rows[0][3], rows[0][4],
                ", ".join("%.2f"%r[2] for r in rows)))
if __name__=="__main__": main()
