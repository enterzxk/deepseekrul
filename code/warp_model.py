import pandas as pd,numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
D=pd.read_csv('ba_clean.csv'); S=pd.read_csv('ba_cell_summary.csv').set_index('cell')
cells=S[S.eol80.notna()].index.tolist()
def build_shape(train_cells, ugrid=np.linspace(0,1.2,121)):
    vals=[]
    for c in train_cells:
        g=D[D.cell==c].sort_values('cycle'); e=S.loc[c,'eol80']
        if not np.isfinite(e): continue
        u=g.cycle.values/e; m=(u>=0)&(u<=1.2)&np.isfinite(g.soh_s.values)
        if m.sum()<5: continue
        vals.append(np.interp(ugrid,u[m],g.soh_s.values[m]))
    V=np.vstack(vals); h=np.nanmedian(V,axis=0); h=np.minimum.accumulate(h)
    return interp1d(ugrid,h,bounds_error=False,fill_value=(h[0],h[-1]))
def fit_warp(h,g,k0,init=(0,1.0)):
    t=g.cycle.values.astype(float); y=g.soh_s.values
    m=(t<=k0)&np.isfinite(y)
    t=t[m]; y=y[m]
    if len(t)<10: return None
    def resid(p):
        a,b=p
        return h((t-a)/b)-y
    try:
        r=least_squares(resid,init,bounds=([-2000,0.05],[2000,20000]),loss='soft_l1',f_scale=0.01,max_nfev=200)
        return r.x
    except Exception: return None
def eol_from_warp(ab): return ab[0]+ab[1]
rows=[]
for test in cells:
    src=S.loc[test,'source']
    for mode in ['global','source']:
        tr=[c for c in cells if c!=test and (mode=='global' or S.loc[c,'source']==src)]
        if len(tr)<3: continue
        h=build_shape(tr); g=D[D.cell==test].sort_values('cycle'); e=S.loc[test,'eol80']
        for k0 in [50,100,200,300,400]:
            if k0>=e*0.9: continue
            ab=fit_warp(h,g,k0)
            if ab is None: continue
            ep=eol_from_warp(ab)
            rows.append(dict(cell=test,source=src,mode=mode,k0=k0,a=ab[0],b=ab[1],eol=e,epred=ep,err=ep-e,pred_ok=ep>k0))
R=pd.DataFrame(rows); R.to_csv('warp_eol.csv',index=False)
for mode in ['global','source']:
 print('====',mode)
 for k0 in [50,100,200,300,400]:
  x=R[(R['mode']==mode)&(R.k0==k0)]
  print(f'k0={k0} n={len(x)} MAE={x.err.abs().mean():7.1f} bias={x.err.mean():+7.1f} medAE={x.err.abs().median():7.1f} within20%={((x.err.abs()/x.eol)<0.2).mean()*100:5.1f}%')
 print(R[R['mode']==mode].groupby('source').err.agg(['count','mean',lambda x: np.mean(np.abs(x))]).round(1).to_string())
