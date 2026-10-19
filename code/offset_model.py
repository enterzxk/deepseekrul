import pandas as pd,numpy as np, random
D=pd.read_csv('ba_clean.csv'); S=pd.read_csv('ba_cell_summary.csv').set_index('cell')
CELLS={c:g.sort_values('cycle').reset_index(drop=True) for c,g in D.groupby('cell')}
cells=sorted(CELLS)
rng=random.Random(0); by={}
for c in cells:
 key=(S.loc[c,'source'],S.loc[c,'chem']); by.setdefault(key,[]).append(c)
fold={}
for key,cs in by.items():
 rng.shuffle(cs)
 for i,c in enumerate(cs): fold[c]=i%5
def group(c,level):
 s=S.loc[c]
 if level=='protocol': return (s.chem,float(s.temp),float(s.crate_d))
 if level=='chem': return (s.chem,)
 return ('all',)
def build_mean(train,level):
 groups={}
 for c in train:
  g=group(c,level); gg=CELLS[c]
  for k,v in zip(gg.cycle.values.astype(int),gg.soh_s.values):
   if np.isfinite(v): groups.setdefault(g,{}).setdefault(k,[]).append(v)
 out={}
 for g,dd in groups.items():
  ks=np.array(sorted(dd)); vals=pd.Series([np.nanmedian(dd[k]) for k in ks]).interpolate(limit_direction='both').values
  out[g]=(ks,vals)
 return out
def mean_vec(m,c,ks,level):
 g=group(c,level)
 if g in m: kk,vals=m[g]
 elif ('all',) in m: kk,vals=m[('all',)]
 else: return np.full(len(ks),np.nan)
 return np.interp(ks,kk,vals)
rows=[]
for f in range(5):
 te=[c for c in cells if fold[c]==f]; tr=[c for c in cells if fold[c]!=f]
 for level in ['protocol','chem','global']:
  mu=build_mean(tr,level)
  for c in te:
   g=CELLS[c]; grid=g.cycle.values.astype(float); y=np.interp(grid,g.cycle,g.soh_s); mv=mean_vec(mu,c,grid,level)
   if not np.all(np.isfinite(mv)): continue
   kmax=grid.max()
   for k0 in range(50,int(kmax)-300+1,10):
    h=grid<=k0
    if h.sum()<10: continue
    rr=(y-mv)[h]
    offs={'last':rr[-1],'med20':np.nanmedian(rr[-20:]),'mean50':np.nanmean(rr[-50:]),'zero':0.0}
    for hh in [50,100,200,300]:
     k=k0+hh; idx=min(np.searchsorted(grid,k),len(y)-1); true=y[idx]
     for name,off in offs.items():
      pred=mv[idx]+off
      rows.append(dict(cell=c,source=S.loc[c,'source'],k0=k0,h=hh,level=level,mode=name,true=true,pred=pred,err=pred-true))
R=pd.DataFrame(rows); R.to_csv('offset_pred.csv',index=False)
for level in ['protocol','chem','global']:
 print('====',level)
 for mode in ['zero','last','med20','mean50']:
  x=R[(R.level==level)&(R['mode']==mode)]
  print(f'{mode:7s} ' + ' '.join([f'h{h}={x[x.h==h].err.abs().mean():.4f}' for h in [50,100,200,300]]))
 print('by source h=300, med20')
 x=R[(R.level==level)&(R.h==300)&(R['mode']=='med20')]; print(x.groupby('source').err.apply(lambda e:np.mean(np.abs(e))).round(4).to_string())
