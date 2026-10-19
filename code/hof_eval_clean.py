import pandas as pd,numpy as np,random
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
  gg=CELLS[c]; y=np.interp(grid,gg.cycle,gg.soh_in); g=group(c)
  groups.setdefault(g,[]).append(y)
 out={}
 for g,V in groups.items():
  V=np.vstack(V); out[g]=np.nanmedian(V,axis=0)
 allV=np.vstack([v for v in groups.values()]); out[('all',)]=np.nanmedian(allV,axis=0)
 return out
def mu_at(m,c,grid): 
 for g in [group(c),('all',)]:
  if g in m: return m[g]
 return np.full(len(grid),np.nan)
MAX=8000; grid=np.arange(1,MAX+1)
rows=[]; erows=[]
for f in range(5):
 te=[c for c in cells if fold[c]==f]; tr=[c for c in cells if fold[c]!=f]; mu=build_mean(tr,grid)
 for c in te:
  g=CELLS[c]; kmax=int(g.cycle.max()); mv=mu_at(mu,c,grid)
  if not np.all(np.isfinite(mv)): continue
  ytrue=np.interp(grid,g.cycle,g.soh_s)
  for k0 in range(50,kmax-300+1,10):
   off=np.interp(k0,g.cycle,g.soh_in)-np.interp(k0,grid,mv)
   for hh in [50,100,200,300]:
    k=k0+hh; pmean=np.interp(k,grid,mv); pred=pmean+off; true=np.interp(k,g.cycle,g.soh_s)
    rows.append(dict(cell=c,source=S.loc[c,'source'],k0=k0,h=hh,true=true,pmean=pmean,pred=pred,err=pred-true,fold=f))
  if np.isfinite(S.loc[c,'eol80']):
   for k0 in [100,200,300,400]:
    if k0>=S.loc[c,'eol80']*0.9: continue
    off=np.interp(k0,g.cycle,g.soh_in)-np.interp(k0,grid,mv)
    pred=mv+off; idx=np.where(pred<=0.8)[0]; ep=int(grid[idx[0]]) if len(idx) else None
    erows.append(dict(cell=c,source=S.loc[c,'source'],k0=k0,eol=S.loc[c,'eol80'],pred=ep,err=(ep-S.loc[c,'eol80']) if ep else np.nan,fold=f))
R=pd.DataFrame(rows); R.to_csv('hof_eval_clean.csv',index=False)
E=pd.DataFrame(erows); E.to_csv('hof_eol_clean.csv',index=False)
print('SOH rows',len(R),'cells',R.cell.nunique())
for h in [50,100,200,300]:
 x=R[R.h==h]; print(f'h={h} n={len(x)} MAE_cell={x.groupby("cell").err.apply(lambda e:np.mean(np.abs(e))).mean():.4f} MAE_win={x.err.abs().mean():.4f} bias={x.err.mean():+.4f}')
print('EOL n',int(E.err.notna().sum()),'MAE',round(float(E.err.abs().mean()),1),'median',round(float(E.err.abs().median()),1),'within20',round(float(np.mean(E.err.abs()/E.eol<0.2)),3),'cens',int(E.pred.isna().sum()))
print(E.dropna(subset=['err']).groupby('source').err.apply(lambda e:np.mean(np.abs(e))).round(1).to_string())
