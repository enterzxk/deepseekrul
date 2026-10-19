import pandas as pd,numpy as np
D=pd.read_csv('ba_clean.csv'); S=pd.read_csv('ba_cell_summary.csv').set_index('cell')
D=D.sort_values(['cell','cycle']); D['soh_in']=D.groupby('cell').soh.transform(lambda x:x.rolling(5,min_periods=1).median())
CELLS={c:g.sort_values('cycle').reset_index(drop=True) for c,g in D.groupby('cell')}
cells=sorted(CELLS)
def group(c):
 s=S.loc[c]; return (s.chem,float(s.temp),float(s.crate_d))
def build_mean(train):
 groups={}
 for c in train:
  g=group(c); gg=CELLS[c]
  for k,v in zip(gg.cycle.values.astype(int),gg.soh_in.values):
   if np.isfinite(v): groups.setdefault(g,{}).setdefault(k,[]).append(v)
 out={}
 for g,dd in groups.items():
  ks=np.array(sorted(dd)); vals=pd.Series([np.nanmedian(dd[k]) for k in ks]).interpolate(limit_direction='both').values
  out[g]=(ks,vals)
 return out
def mvec(m,c,ks):
 g=group(c)
 if g in m: kk,vals=m[g]
 elif len(m)>0:
  allvals=np.concatenate([v for _,v in m.values()]); return np.full(len(ks),np.nanmedian(allvals))
 else: return np.full(len(ks),np.nan)
 return np.interp(ks,kk,vals)
rows=[]
for src in sorted(S.source.unique()):
 te=[c for c in cells if S.loc[c,'source']==src]; tr=[c for c in cells if S.loc[c,'source']!=src]
 mu=build_mean(tr)
 for c in te:
  g=CELLS[c]; grid=g.cycle.values.astype(float); y=np.interp(grid,g.cycle,g.soh_s); mv=mvec(mu,c,grid)
  if not np.all(np.isfinite(mv)): continue
  kmax=grid.max()
  for k0 in range(50,int(kmax)-300+1,10):
   hh=grid<=k0
   if hh.sum()<10: continue
   off=(np.interp(grid,g.cycle,g.soh_in)-mv)[hh][-1]
   for h in [50,100,200,300]:
    k=k0+h; idx=min(np.searchsorted(grid,k),len(y)-1); true=y[idx]; pred=mv[idx]+off
    rows.append(dict(cell=c,source=src,k0=k0,h=h,true=true,pred=pred,err=pred-true))
R=pd.DataFrame(rows); R.to_csv('losource_hof_pred.csv',index=False)
print('==== LO-source HOF')
for h in [50,100,200,300]:
 x=R[R.h==h]; print(f'h={h} n={len(x)} MAE={x.err.abs().mean():.4f} bias={x.err.mean():+.4f}')
print('by source h=300')
x=R[R.h==300]; print(x.groupby('source').err.apply(lambda e:np.mean(np.abs(e))).round(4).to_string())
# EOL from full predicted trajectory
rows=[]
for src in sorted(S.source.unique()):
 tr=[c for c in cells if S.loc[c,'source']!=src]; mu=build_mean(tr)
 for c in cells:
  if S.loc[c,'source']!=src or not np.isfinite(S.loc[c,'eol80']): continue
  g=CELLS[c]; grid=np.arange(1,int(max(g.cycle.max(),S.loc[c,'eol80']*1.5))+1); y=np.interp(grid,g.cycle,g.soh_s); mv=mvec(mu,c,grid)
  if not np.all(np.isfinite(mv)): continue
  for k0 in [100,200,300,400]:
   if k0>=S.loc[c,'eol80']*0.9 or k0>g.cycle.max(): continue
   off=np.interp(k0,g.cycle,g.soh_in)-np.interp(k0,grid,mv); pred=mv+off
   idx=np.where(pred<=0.8)[0]; ep=int(grid[idx[0]]) if len(idx) else None
   rows.append(dict(cell=c,source=src,k0=k0,eol=S.loc[c,'eol80'],pred=ep,err=(ep-S.loc[c,'eol80']) if ep else np.nan))
E=pd.DataFrame(rows); E.to_csv('losource_hof_eol.csv',index=False)
print('EOL by source')
print(E.dropna(subset=['err']).groupby('source').err.apply(lambda e:np.mean(np.abs(e))).round(1).to_string())
print('overall MAE',E.err.abs().mean().round(1),'n',E.err.notna().sum())
