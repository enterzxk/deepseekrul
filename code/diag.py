import pandas as pd, numpy as np, glob, os
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

CELLS={}
for p in sorted(glob.glob('parsed2/*.csv')):
    d=pd.read_csv(p).sort_values('cycle_global')
    cell=d.cell.iloc[0]; nom=1.1 if cell.startswith('CS2') else 1.35
    d=d[(d.vmin<=2.75)&(d.vmax>=4.15)&(d.cap_dis>0.15*nom)&np.isfinite(d.cap_dis)].copy()
    d['soh']=d.cap_dis/nom
    d['k']=d.cycle_global.astype(int)
    # target: centered median 9 on valid sequence, but only for evaluation
    d['soh_s']=d.soh.rolling(9,center=True,min_periods=3).median()
    CELLS[cell]=d
    print(cell,len(d),d.k.min(),d.k.max())

def interp_at(d,col,k):
    x=d.k.values; y=d[col].values
    if k<x.min() or k>x.max(): return np.nan
    return float(np.interp(k,x,y))

def fit_linear(t,y,W):
    m=t>=t.max()-W+1
    if m.sum()<5: return lambda x: np.nan
    b,a=np.polyfit(t[m],y[m],1)
    return lambda x: a+b*x

def fit_quad(t,y,W):
    m=t>=t.max()-W+1
    if m.sum()<8: return lambda x: np.nan
    c=np.polyfit(t[m],y[m],2)
    return lambda x: np.polyval(c,x)

def fit_exp(t,y,W):
    m=t>=t.max()-W+1
    if m.sum()<5: return lambda x: np.nan
    # log(y) linear
    yy=np.log(np.clip(y[m],1e-3,None))
    b,a=np.polyfit(t[m],yy,1)
    return lambda x: np.exp(a+b*x)

def fit_power(t,y,W=None):
    # soh = 1 - a*(k-k0)^p, fit on all or window; bounds
    m=np.ones(len(t),bool) if W is None else (t>=t.max()-W+1)
    tt=t[m]; yy=y[m]
    x0=tt.min()
    def f(x,a,p): return 1-a*np.clip(x-x0,0,None)**p
    try:
        popt,_=curve_fit(f,tt,yy,p0=[0.01,0.5],bounds=([1e-5,0.05],[1,2.0]),maxfev=20000)
    except Exception:
        return lambda x: np.nan
    return lambda x: f(x,*popt)

models={
 'persist': lambda t,y: (lambda x: y[-1]),
 'linear100': lambda t,y: fit_linear(t,y,100),
 'linear200': lambda t,y: fit_linear(t,y,200),
 'exp100': lambda t,y: fit_exp(t,y,100),
 'exp200': lambda t,y: fit_exp(t,y,200),
 'quad200': lambda t,y: fit_quad(t,y,200),
 'power_all': lambda t,y: fit_power(t,y,None),
 'power300': lambda t,y: fit_power(t,y,300),
}

origins=[100,200,300,400]
horizons=[50,100,200,300]
rows=[]
for cell,d in CELLS.items():
    for k0 in origins:
        h=d[d.k<=k0]
        if len(h)<20: continue
        t=h.k.values.astype(float); y=h.soh.values
        # use a causal-ish smoothed input? raw
        for name,fit in models.items():
            f=fit(t,y)
            for hh in horizons:
                yt=interp_at(d,'soh_s',k0+hh)
                yp=f(k0+hh)
                rows.append(dict(cell=cell,k0=k0,h=hh,model=name,y_true=yt,y_pred=yp))
res=pd.DataFrame(rows)
res['err']=res.y_pred-res.y_true
res.to_csv('diag_predictions.csv',index=False)
print('rows',len(res),'nan',res.err.isna().sum())
# horizon summary
for hh in horizons:
    s=res[res.h==hh]
    print('==== h',hh)
    for name in models:
        x=s[s.model==name]
        print(f'{name:12s} n={x.err.notna().sum():3d} MAE={x.err.abs().mean():.4f} bias={x.err.mean():+.4f} over={((x.err>0.05).mean()*100):5.1f}% under={((x.err<-0.05).mean()*100):5.1f}%')
