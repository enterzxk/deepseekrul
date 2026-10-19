import pandas as pd,numpy as np, torch, torch.nn as nn, random
from torch.utils.data import Dataset,DataLoader
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
chems=['LCO','LFP','NCA','NMC','NMC-LCO']
def group(c):
 s=S.loc[c]; return (s.chem,float(s.temp),float(s.crate_d))
def build_mean(train):
 groups={}
 for c in train:
  g=group(c); gg=CELLS[c]
  for k,v in zip(gg.cycle.values.astype(int),gg.soh_s.values):
   if np.isfinite(v): groups.setdefault(g,{}).setdefault(k,[]).append(v)
 out={}
 for g,dd in groups.items():
  ks=np.array(sorted(dd)); vals=pd.Series([np.nanmedian(dd[k]) for k in ks]).interpolate(limit_direction='both').values
  out[g]=(ks,vals)
 return out
def mvec(m,c,ks):
 g=group(c)
 if g in m: kk,vals=m[g]
 else:
  vals=[np.nanmedian(v) for kk2,v in m.values()]; return np.full(len(ks),np.nanmedian(vals))
 return np.interp(ks,kk,vals)
def cov(c,k0):
 s=S.loc[c]; v=[(s.temp-25)/10,np.log(s.crate_d),np.log(s.nom),k0/1000]
 v += [1.0 if s.chem==ch else 0.0 for ch in chems]
 return np.array(v,dtype=np.float32)
class DS(Dataset):
 def __init__(self,X,Y,M): self.X=torch.tensor(X); self.Y=torch.tensor(Y); self.M=torch.tensor(M)
 def __len__(self): return len(self.X)
 def __getitem__(self,i): return self.X[i],self.Y[i],self.M[i]
class Net(nn.Module):
 def __init__(self,sd):
  super().__init__(); self.gru=nn.GRU(2,32,batch_first=True); self.st=nn.Linear(sd,32); self.head=nn.Sequential(nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,4))
 def forward(self,x,m):
  o,_=self.gru(x); h=o[:,-1]; s=torch.relu(self.st(m)); return self.head(torch.cat([h,s],1))
L=100; HS=[50,100,200,300]
def samples(cs,mu):
 X=[];Y=[];M=[];meta=[]
 for c in cs:
  g=CELLS[c]; kmax=int(g.cycle.max()); grid=np.arange(1,kmax+1)
  soh=np.interp(grid,g.cycle,g.soh_in); mv=mvec(mu,c,grid.astype(float)); tgt=np.interp(grid,g.cycle,g.soh_s)
  if not np.all(np.isfinite(mv)): continue
  r=soh-mv
  for k0 in range(L,kmax-300+1,10):
   x=r[k0-L:k0].copy(); last=soh[k0-1]
   xx=np.stack([x,np.full(L,last)],1).astype(np.float32)
   y=np.array([tgt[k0+h-1]-mv[k0+h-1] for h in HS],dtype=np.float32)
   if not np.all(np.isfinite(y)): continue
   X.append(xx);Y.append(y);M.append(cov(c,k0));meta.append((c,k0))
 return np.array(X),np.array(Y),np.array(M),meta
rows=[]
for f in range(5):
 tr=[c for c in cells if fold[c] not in (f,(f+1)%5)]; va=[c for c in cells if fold[c]==(f+1)%5]; te=[c for c in cells if fold[c]==f]
 mu=build_mean(tr)
 Xtr,Ytr,Mtr,_=samples(tr,mu); Xva,Yva,Mva,_=samples(va,mu); Xte,Yte,Mte,met=samples(te,mu)
 net=Net(Mtr.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4); lf=nn.MSELoss()
 dl=DataLoader(DS(Xtr,Ytr,Mtr),batch_size=128,shuffle=True); best=1e9; bs=None
 for ep in range(40):
  net.train()
  for x,y,m in dl:
   opt.zero_grad(); p=net(x,m); loss=lf(p,y); loss.backward(); opt.step()
  net.eval()
  with torch.no_grad(): pv=net(torch.tensor(Xva),torch.tensor(Mva)); vl=((pv-torch.tensor(Yva)).abs().mean()).item()
  if vl<best: best=vl; bs={k:v.clone() for k,v in net.state_dict().items()}
 net.load_state_dict(bs); net.eval()
 with torch.no_grad(): pt=net(torch.tensor(Xte),torch.tensor(Mte)).numpy()
 for i,(c,k0) in enumerate(met):
  g=CELLS[c]; grid=np.arange(1,int(g.cycle.max())+1); mv=mvec(mu,c,grid.astype(float)); tgt=np.interp(grid,g.cycle,g.soh_s)
  for j,h in enumerate(HS):
   k=k0+h-1; pred=mv[k]+pt[i,j]; true=tgt[k]
   rows.append(dict(cell=c,source=S.loc[c,'source'],k0=k0,h=h,pred=float(pred),true=float(true),err=float(pred-true)))
 print('fold',f,'val',round(best,4),flush=True)
R=pd.DataFrame(rows); R.to_csv('resgru_pred.csv',index=False)
for h in HS:
 x=R[R.h==h]; print(f'h={h} n={len(x)} MAE={x.err.abs().mean():.4f} bias={x.err.mean():+.4f}')
