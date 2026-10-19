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
def group(c):
 s=S.loc[c]; return (s.chem,float(s.temp),float(s.crate_d))
def build_mean(train,grid):
 groups={}
 for c in train:
  gg=CELLS[c]; y=np.interp(grid,gg.cycle,gg.soh_in); groups.setdefault(group(c),[]).append(y)
 V=np.vstack([v for vv in groups.values() for v in vv]); allm=np.nanmedian(V,axis=0)
 out={g:np.nanmedian(np.vstack(v),axis=0) for g,v in groups.items()}; out[('all',)]=allm; return out
def mu_at(m,c,grid):
 for g in [group(c),('all',)]:
  if g in m: return m[g]
 return np.full(len(grid),np.nan)
def cross(fn,k0,maxk=10000):
 ks=np.arange(k0+1,maxk+1); yy=np.array([fn(x) for x in ks]); idx=np.where(yy<=0.8)[0]
 return int(ks[idx[0]]) if len(idx) else None
MAX=10000; grid=np.arange(1,MAX+1)
rows=[]
for f in range(5):
 te=[c for c in cells if fold[c]==f]; tr=[c for c in cells if fold[c]!=f]; mu=build_mean(tr,grid)
 for c in te:
  if not np.isfinite(S.loc[c,'eol80']): continue
  g=CELLS[c]; mv=mu_at(mu,c,grid); e=S.loc[c,'eol80']
  for k0 in [100,200,300,400]:
   if k0>=e*0.9 or k0>g.cycle.max(): continue
   h=g[g.cycle<=k0]; t=h.cycle.values.astype(float); y=h.soh_in.values
   off=np.interp(k0,g.cycle,g.soh_in)-np.interp(k0,grid,mv)
   preds={}
   preds['hof']=cross(lambda x:np.interp(x,grid,mv)+off,k0)
   preds['pmean']=cross(lambda x:np.interp(x,grid,mv),k0)
   # per-cell fits
   try:
    b,a=np.polyfit(t,y,1); preds['linear']=cross(lambda x:a+b*x,k0)
   except: preds['linear']=None
   try:
    yy=np.log(np.clip(y,1e-3,None)); b,a=np.polyfit(t,yy,1); preds['exp']=cross(lambda x:np.exp(a+b*x),k0)
   except: preds['exp']=None
   def fpow(x,a,p): return 1-a*np.clip(x,0,None)**p
   try:
    popt,_=curve_fit(fpow,t,y,p0=[0.01,0.5],bounds=([1e-5,0.05],[1,2.0]),maxfev=20000); preds['power']=cross(lambda x:fpow(x,*popt),k0)
   except: preds['power']=None
   for name,pr in preds.items():
    rows.append(dict(cell=c,source=S.loc[c,'source'],k0=k0,model=name,eol=e,pred=pr,err=(pr-e) if pr else np.nan))
R=pd.DataFrame(rows); R.to_csv('eol_compare_clean.csv',index=False)
for m in ['hof','pmean','linear','exp','power']:
 x=R[R.model==m]
 print('====',m,'n',x.err.notna().sum(),'cens',x.pred.isna().sum(),'MAE',round(x.err.abs().mean(),1),'median',round(x.err.abs().median(),1),'within20',round(np.mean(x.err.abs()/x.eol<0.2),3),'bias',round(x.err.mean(),1))
 print('  by source',x.dropna(subset=['err']).groupby('source').err.apply(lambda e:round(np.mean(np.abs(e)),1)).to_dict())
