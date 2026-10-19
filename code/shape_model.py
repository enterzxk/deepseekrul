import pandas as pd,numpy as np
from scipy.interpolate import interp1d
D=pd.read_csv('ba_clean.csv'); S=pd.read_csv('ba_cell_summary.csv')
S=S.set_index('cell')
# cells with observed eol80
cells=S[S.eol80.notna()].index.tolist()
# shape builder from training cells
def build_shape(train_cells, ugrid=np.linspace(0,1.0,101)):
    vals=[]
    for c in train_cells:
        g=D[D.cell==c].sort_values('cycle'); e=S.loc[c,'eol80']
        if not np.isfinite(e): continue
        u=g.cycle.values/e
        # only u<=1.0 and valid
        m=(u>=0)&(u<=1.0)&np.isfinite(g.soh_s.values)
        if m.sum()<5: continue
        vals.append(np.interp(ugrid,u[m],g.soh_s.values[m]))
    if not vals: return None
    V=np.vstack(vals)
    h=np.nanmedian(V,axis=0)
    # enforce monotone non-increasing via isotonic-like cummin
    h=np.minimum.accumulate(h)
    return interp1d(ugrid,h,bounds_error=False,fill_value=(h[0],h[-1]))
def invert(h,s0):
    # find u where h(u)=s0; h decreasing
    us=np.linspace(0,1.0,1001); hs=h(us)
    idx=np.where(hs<=s0)[0]
    if len(idx)==0: return 1.0
    i=idx[0]
    if i==0: return 0.0
    # linear interp between i-1 and i
    y0,y1=hs[i-1],hs[i]
    return us[i-1]+(s0-y0)/(y1-y0)*(us[i]-us[i-1])
rows=[]
for test in cells:
    src=S.loc[test,'source']
    for mode in ['global','source']:
        tr=[c for c in cells if c!=test and (mode=='global' or S.loc[c,'source']==src)]
        if len(tr)<3: continue
        h=build_shape(tr)
        g=D[D.cell==test].sort_values('cycle'); e=S.loc[test,'eol80']
        for k0 in [100,200,300,400]:
            if k0>=e*0.9: continue
            s0=float(np.interp(k0,g.cycle,g.soh_s))
            u0=invert(h,s0)
            epred=k0/u0 if u0>1e-3 else np.nan
            rows.append(dict(cell=test,source=src,mode=mode,k0=k0,s0=s0,u0=u0,eol=e,epred=epred,err=epred-e))
R=pd.DataFrame(rows); R.to_csv('shape_eol.csv',index=False)
for mode in ['global','source']:
 print('====',mode)
 for k0 in [100,200,300,400]:
  x=R[(R['mode']==mode)&(R.k0==k0)]
  print(f'k0={k0} n={len(x)} MAE={x.err.abs().mean():7.1f} bias={x.err.mean():+7.1f} medAE={x.err.abs().median():7.1f} within20%={((x.err.abs()/x.eol)<0.2).mean()*100:5.1f}%')
 print('by source')
 print(R[R['mode']==mode].groupby('source').err.agg(['count','mean',lambda x: np.mean(np.abs(x))]).round(1).to_string())
