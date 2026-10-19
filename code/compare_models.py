import pandas as pd,numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
D=pd.read_csv('ba_clean.csv'); S=pd.read_csv('ba_cell_summary.csv').set_index('cell')
CELLS={c:g.sort_values('cycle') for c,g in D.groupby('cell')}
# per-cell curve fits
def fit_linear(t,y,W=100):
 m=t>=t.max()-W+1; b,a=np.polyfit(t[m],y[m],1); return lambda x:a+b*x
def fit_exp(t,y,W=100):
 m=t>=t.max()-W+1; yy=np.log(np.clip(y[m],1e-3,None)); b,a=np.polyfit(t[m],yy,1); return lambda x:np.exp(a+b*x)
def fit_power(t,y):
 def f(x,a,p): return 1-a*np.clip(x,0,None)**p
 try:
  popt,_=curve_fit(f,t,y,p0=[0.01,0.5],bounds=([1e-5,0.05],[1,2.0]),maxfev=20000)
  return lambda x:f(x,*popt)
 except Exception: return lambda x:np.nan
# shape
cells_obs=S[S.eol80.notna()].index.tolist()
def build_shape(train, ugrid=np.linspace(0,1.2,121)):
 vals=[]
 for c in train:
  g=CELLS[c]; e=S.loc[c,'eol80']
  if not np.isfinite(e): continue
  u=g.cycle.values/e; m=(u>=0)&(u<=1.2)&np.isfinite(g.soh_s.values)
  if m.sum()<10: continue
  vals.append(np.interp(ugrid,u[m],g.soh_s.values[m]))
 V=np.vstack(vals); h=np.nanmedian(V,axis=0); h=np.minimum.accumulate(h)
 return interp1d(ugrid,h,bounds_error=False,fill_value=(h[0],h[-1]))
# EOL features/pred from RF (reuse eol_features and LOCO)
X=pd.read_csv('eol_features.csv'); feat=['k0','soh','slope20','slope50','slope100','curv','init','drop','temp','crate','nom','ch_LFP','ch_NCA','ch_NMC','ch_LCO']
eolpred={}
obs=X[X.cens]
for c in sorted(X.cell.unique()):
 tr=obs[obs.cell!=c]
 m=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,random_state=0).fit(tr[feat],np.log(tr.eol))
 for k0 in [50,100,200,300,400]:
  row=X[(X.cell==c)&(X.k0==k0)]
  if len(row): eolpred[(c,k0)]=float(np.exp(m.predict(row[feat])[0]))
rows=[]
for c,g in CELLS.items():
 s=S.loc[c]
 for k0 in [50,100,200,300,400]:
  h=g[g.cycle<=k0]
  if len(h)<15: continue
  t=h.cycle.values.astype(float); y=h.soh_s.values
  fits={'persist':lambda x:y[-1], 'linear100':fit_linear(t,y,100), 'exp100':fit_exp(t,y,100), 'power':fit_power(t,y)}
  # shape using predicted eol and oracle eol
  ep=eolpred.get((c,k0))
  if ep is not None and np.isfinite(ep):
   tr=[x for x in cells_obs if x!=c]
   hs=build_shape(tr); fits['scale_shape']=lambda x,hs=hs,ep=ep: float(hs(x/ep))
  if np.isfinite(s.eol80):
   tr=[x for x in cells_obs if x!=c]
   ho=build_shape(tr); fits['oracle_shape']=lambda x,ho=ho,e=s.eol80: float(ho(x/e))
  for hh in [50,100,200,300]:
   k=k0+hh
   if k>g.cycle.max(): continue
   true=np.interp(k,g.cycle,g.soh_s)
   for name,f in fits.items():
    try: pred=float(f(k))
    except Exception: pred=np.nan
    rows.append(dict(cell=c,source=s.source,chem=s.chem,k0=k0,h=hh,model=name,true=true,pred=pred,err=pred-true))
R=pd.DataFrame(rows); R.to_csv('compare_pred.csv',index=False)
print('rows',len(R),'models',R.model.unique())
for hh in [50,100,200,300]:
 print('==== h',hh)
 for m in ['persist','linear100','exp100','power','scale_shape','oracle_shape']:
  x=R[(R.h==hh)&(R.model==m)]
  print(f'{m:13s} n={len(x):4d} MAE={x.err.abs().mean():.4f} bias={x.err.mean():+.4f} over5%={((x.err>0.05).mean()*100):5.1f}%')
