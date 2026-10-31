import glob, numpy as np, pandas as pd
R="/home/yndx/zxk/bearing-rul-pi/experiments"
ARMS={'A':('e10',True,[2027,2028,2029]),'BE':('e10b',False,[2027,2028,2029]),
      'cnn1d':('e11',False,[2027,2028,2029]),'bilstm':('e11',False,[2027,2028,2029]),
      'gru':('e11',False,[2027,2028,2029]),'tcn':('e11',False,[2027]),'tf':('e11',False,[2027])}
def load(arm,s,f):
    exp,f0,_=ARMS[arm]
    pat=(f"{R}/e3/{arm}_s{s}/FD002/{arm}_s{s}/matched_cutoff_predictions.csv" if f==0 and f0
         else f"{R}/{exp}/{arm}_s{s}_f{f}/FD002/{arm}_s{s}/matched_cutoff_predictions.csv")
    g=glob.glob(pat); return pd.read_csv(g[0]).sort_values('unit_id') if g else None
tab={}
for arm in ARMS:
    row=[]
    for f in range(5):
        vals=[]
        for s in ARMS[arm][2]:
            d=load(arm,s,f)
            if d is None: break
            e=d.signed_error.to_numpy(float); vals.append(np.sqrt((e**2).mean()))
        row.append(np.mean(vals) if vals else np.nan)
    tab[arm]=row
print("per-fold RMSE")
print("%-8s %7s %7s %7s %7s %7s" % ("arm","f0","f1","f2","f3","f4"))
for a,v in tab.items(): print("%-8s"%" ".join("%7.2f"%x for x in v) if False else "%-8s %7.2f %7.2f %7.2f %7.2f %7.2f"%(a,*v))
print("\nrank per fold (1 = best)")
winners={}
for f in range(5):
    order=sorted(tab, key=lambda a: tab[a][f])
    winners[f]=order[0]
    print("fold%d: %s" % (f, " > ".join(order)))
print("\nfold-0 winner:", winners[0], "| winners elsewhere:", sorted(set(winners[f] for f in range(1,5))))
print("prediction-3 (fold0 leader not leading on >=2 other folds):",
      sum(1 for f in range(1,5) if winners[f]!=winners[0]), "folds disagree ->",
      "FAILED" if sum(1 for f in range(1,5) if winners[f]!=winners[0])<2 else "held")
