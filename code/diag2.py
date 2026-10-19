import pandas as pd, numpy as np, glob
from scipy.optimize import curve_fit
CELLS={}
for p in sorted(glob.glob('parsed2/*.csv')):
    d=pd.read_csv(p).sort_values('cycle_global'); cell=d.cell.iloc[0]; nom=1.1 if cell.startswith('CS2') else 1.35
    d=d[(d.vmin<=2.75)&(d.vmax>=4.15)&(d.cap_dis>0.15*nom)&np.isfinite(d.cap_dis)].copy()
    d['soh']=d.cap_dis/nom; d['k']=d.cycle_global.astype(int)
    d['soh_s']=d.soh.rolling(9,center=True,min_periods=3).median()
    CELLS[cell]=d
def interp(d,col,k):
    x=d.k.values; y=d[col].values
    return float(np.interp(k,x,y)) if x.min()<=k<=x.max() else np.nan
def fit_linear(t,y,W):
    m=t>=t.max()-W+1; b,a=np.polyfit(t[m],y[m],1); return lambda x:a+b*x
def fit_exp(t,y,W):
    m=t>=t.max()-W+1; yy=np.log(np.clip(y[m],1e-3,None)); b,a=np.polyfit(t[m],yy,1); return lambda x:np.exp(a+b*x)
def fit_power(t,y,W=None):
    m=np.ones(len(t),bool) if W is None else (t>=t.max()-W+1); tt=t[m]; yy=y[m]; x0=tt.min()
    def f(x,a,p): return 1-a*np.clip(x-x0,0,None)**p
    try: popt,_=curve_fit(f,tt,yy,p0=[0.01,0.5],bounds=([1e-5,0.05],[1,2.0]),maxfev=20000)
    except Exception: return lambda x:np.nan
    return lambda x:f(x,*popt)
models={'persist':lambda t,y:(lambda x:y[-1]),'linear100':lambda t,y:fit_linear(t,y,100),'linear200':lambda t,y:fit_linear(t,y,200),'exp100':lambda t,y:fit_exp(t,y,100),'exp200':lambda t,y:fit_exp(t,y,200),'power_all':lambda t,y:fit_power(t,y,None),'power300':lambda t,y:fit_power(t,y,300)}
def crossing(f,k0,th,maxk=2500):
    ks=np.arange(k0+1,maxk+1)
    yy=np.array([f(x) for x in ks])
    idx=np.where(yy<=th)[0]
    return int(ks[idx[0]]) if len(idx) else None
rows=[]
for cell,d in CELLS.items():
    for th in [0.8,0.7]:
        obs=crossing(lambda x: interp(d,'soh_s',x), 0, th) # first crossing from 0
        for k0 in [100,200,300,400]:
            if k0>= (obs or 99999): continue
            h=d[d.k<=k0]
            t=h.k.values.astype(float); y=h.soh.values
            for name,fit in models.items():
                f=fit(t,y); pr=crossing(f,k0,th)
                rows.append(dict(cell=cell,th=th,k0=k0,model=name,obs=obs,pred=pr,err=(pr-obs) if pr is not None else np.nan,censored=pr is None))
res=pd.DataFrame(rows); res.to_csv('diag_eol.csv',index=False)
for th in [0.8,0.7]:
 print('==== threshold',th)
 for k0 in [100,200,300,400]:
  print('-- k0',k0)
  for name in models:
   x=res[(res.th==th)&(res.k0==k0)&(res.model==name)]
   print(f'{name:11s} n={len(x):2d} cens={x.censored.sum():2d} MAE={x.err.abs().mean():7.1f} bias={x.err.mean():+7.1f} maxabs={x.err.abs().max():7.1f} sign+={((x.err>0).mean()*100):5.1f}%')
