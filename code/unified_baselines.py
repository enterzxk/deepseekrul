import pandas as pd,numpy as np,random
from scipy.optimize import curve_fit
D=pd.read_csv('ba_clean.csv'); S=pd.read_csv('ba_cell_summary.csv').set_index('cell')
D=D.sort_values(['cell','cycle']); D['soh_in']=D.groupby('cell').soh.transform(lambda x:x.rolling(5,min_periods=1).median())
CELLS={c:g.sort_values('cycle').reset_index(drop=True) for c,g in D.groupby('cell')}
cells=sorted(CELLS)
rng=random.Random(0); by={}
for c in cells:
 key=(S.loc[c,'source'],S.loc[c,'chem']); by.setdefault(key,[]).append(c)
fold={}
for key,cs in by.items():
 rng.shuffle(cs)
 for i,c in enumerate(cs): fold[c]=i%5
def fit_lin(t,y,W=100):
 m=t>=t.max()-W+1; b,a=np.polyfit(t[m],y[m],1); return lambda x:a+b*x
def fit_exp(t,y,W=100):
 m=t>=t.max()-W+1; yy=np.log(np.clip(y[m],1e-3,None)); b,a=np.polyfit(t[m],yy,1); return lambda x:np.exp(a+b*x)
def fit_pow(t,y):
 def f(x,a,p): return 1-a*np.clip(x,0,None)**p
 try:
  popt,_=curve_fit(f,t,y,p0=[0.01,0.5],bounds=([1e-5,0.05],[1,2.0]),maxfev=20000); return lambda x:f(x,*popt)
 except Exception: return lambda x:np.nan
rows=[]
for f in range(5):
 te=[c for c in cells if fold[c]==f]
 for c in te:
  g=CELLS[c]; kmax=int(g.cycle.max())
  for k0 in range(50,kmax-300+1,10):
   h=g[g.cycle<=k0]
   if len(h)<10: continue
   t=h.cycle.values.astype(float); y=h.soh_in.values
   fits={'persist':lambda x:y[-1],'linear100':fit_lin(t,y,100),'exp100':fit_exp(t,y,100),'power':fit_pow(t,y)}
   for hh in [50,100,200,300]:
    k=k0+hh
    if k>kmax: continue
    true=float(np.interp(k,g.cycle,g.soh_s))
    for name,fn in fits.items():
     try: pred=float(fn(k))
     except Exception: pred=np.nan
     rows.append(dict(cell=c,source=S.loc[c,'source'],k0=k0,h=hh,model=name,true=true,pred=pred,err=pred-true))
R=pd.DataFrame(rows); R.to_csv('unified_percell.csv',index=False)
for hh in [50,100,200,300]:
 print('==== h',hh)
 for m in ['persist','linear100','exp100','power']:
  x=R[(R.h==hh)&(R.model==m)].dropna(subset=['err']); print(f'{m:10s} n={len(x):5d} MAE={x.err.abs().mean():.4f} bias={x.err.mean():+.4f}')
 print('by source h=300')
 x=R[(R.h==300)]; print(x.groupby(['source','model']).err.apply(lambda e:np.mean(np.abs(e))).unstack().round(4).to_string())
