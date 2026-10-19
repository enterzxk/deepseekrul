import pandas as pd,numpy as np, torch, torch.nn as nn, random, os, json
from torch.utils.data import Dataset,DataLoader
D=pd.read_csv('ba_clean.csv'); S=pd.read_csv('ba_cell_summary.csv').set_index('cell')
# causal input
D=D.sort_values(['cell','cycle'])
D['soh_in']=D.groupby('cell').soh.transform(lambda x: x.rolling(5,min_periods=1).median())
CELLS={c:g.sort_values('cycle').reset_index(drop=True) for c,g in D.groupby('cell')}
chems=['LCO','LFP','NCA','NMC','NMC-LCO']
def cov(c):
 s=S.loc[c]; v=[1.0 if s.chem==ch else 0.0 for ch in chems]
 v += [(s.temp-25)/10, np.log(s.crate_d), np.log(s.nom)]
 return np.array(v,dtype=np.float32)
def make_samples(cells,L=100,HS=[50,100,200,300],stride=10):
 X=[];Y=[];M=[];meta=[]
 for c in cells:
  g=CELLS[c]; kmax=int(g.cycle.max()); grid=np.arange(1,kmax+1)
  soh=np.interp(grid,g.cycle,g.soh_in)
  tgt=np.interp(grid,g.cycle,g.soh_s)
  for k0 in range(L,kmax-max(HS)+1,stride):
   cv=np.concatenate([cov(c), np.array([k0/1000.0],dtype=np.float32)])
   x=soh[k0-L:k0].copy()
   # normalize by last value and include absolute last as extra channel
   last=x[-1]
   x=(x-last)  # relative history
   xx=np.stack([x, np.full(L,last)],axis=1).astype(np.float32) # [L,2]
   y=np.array([tgt[k0+h-1] for h in HS],dtype=np.float32)
   X.append(xx);Y.append(y);M.append(cv);meta.append((c,k0))
 return np.array(X),np.array(Y),np.array(M),meta
# folds stratified by source/chem
cells=sorted(CELLS)
rng=random.Random(0); by={}
for c in cells:
 key=(S.loc[c,'source'],S.loc[c,'chem']); by.setdefault(key,[]).append(c)
fold={}
for key,cs in by.items():
 rng.shuffle(cs)
 for i,c in enumerate(cs): fold[c]=i%5
for f in range(5):
 print('fold',f,'cells',sum(1 for c in cells if fold[c]==f))
class DS(Dataset):
 def __init__(self,X,Y,M): self.X=torch.tensor(X); self.Y=torch.tensor(Y); self.M=torch.tensor(M)
 def __len__(self): return len(self.X)
 def __getitem__(self,i): return self.X[i],self.Y[i],self.M[i]
class Net(nn.Module):
 def __init__(self,static_dim):
  super().__init__(); self.gru=nn.GRU(2,32,batch_first=True); self.static=nn.Linear(static_dim,32); self.head=nn.Sequential(nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,4))
 def forward(self,x,m):
  o,_=self.gru(x); h=o[:,-1]; s=torch.relu(self.static(m)); return self.head(torch.cat([h,s],dim=1))
def train_fold(f):
 tr=[c for c in cells if fold[c] not in (f,(f+1)%5)]; va=[c for c in cells if fold[c]==(f+1)%5]; te=[c for c in cells if fold[c]==f]
 Xtr,Ytr,Mtr,_=make_samples(tr); Xva,Yva,Mva,metva=make_samples(va); Xte,Yte,Mte,mette=make_samples(te)
 ds=DS(Xtr,Ytr,Mtr); dl=DataLoader(ds,batch_size=128,shuffle=True)
 net=Net(Mtr.shape[1]); opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.MSELoss()
 best=1e9; beststate=None
 for ep in range(40):
  net.train()
  for x,y,m in dl:
   opt.zero_grad(); p=net(x,m); loss=lossf(p,y); loss.backward(); opt.step()
  net.eval()
  with torch.no_grad():
   pv=net(torch.tensor(Xva),torch.tensor(Mva)); vloss=((pv-torch.tensor(Yva)).abs().mean()).item()
  if vloss<best: best=vloss; beststate={k:v.clone() for k,v in net.state_dict().items()}
 net.load_state_dict(beststate); net.eval()
 with torch.no_grad(): pt=net(torch.tensor(Xte),torch.tensor(Mte)).numpy()
 rows=[]
 for i,(c,k0) in enumerate(mette):
  for j,h in enumerate([50,100,200,300]): rows.append(dict(cell=c,k0=k0,h=h,pred=float(pt[i,j]),true=float(Yte[i,j])))
 return pd.DataFrame(rows),best
allr=[]
for f in range(5):
 r,b=train_fold(f); allr.append(r); print('fold',f,'bestval',round(b,4),'n',len(r),flush=True)
R=pd.concat(allr); R.to_csv('gru_pred.csv',index=False)
print('==== GRU overall')
for h in [50,100,200,300]:
 x=R[R.h==h]; print(f'h={h} n={len(x)} MAE={(x.pred-x.true).abs().mean():.4f} bias={(x.pred-x.true).mean():+.4f}')
print('by source h=300')
x=R[R.h==300]; x=x.assign(source=x.cell.map(S.source)); print(x.groupby('source').apply(lambda d:(d.pred-d.true).abs().mean()).round(4).to_string())
