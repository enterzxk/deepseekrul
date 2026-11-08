"""Input-sensitivity (GUM-style sensitivity coefficient) as an observability measure."""
import sys
import numpy as np, pandas as pd, torch
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb","/home/user/rulb/tim_20261105/full_20261105"]
import run_fold as base, run_ida as ida
from scipy.stats import spearmanr
from run_full_fold3 import build_net

subset=sys.argv[1]; fold=int(sys.argv[2]); rundir=sys.argv[3] if len(sys.argv)>3 else 'runs3_w50_base'
folds,df=ida.outer_folds(subset); df=df.sort_values(['unit','cycle']).reset_index(drop=True)
train,valid=folds[fold]
tables=base.prior_tables(df,subset,train,valid,fold)
tr,va=ida.prep_fold(df,subset,fold,train,valid)
xva,yva,eva,mva=va[:4]
net=build_net('M0',50)
net.load_state_dict(torch.load(f'{rundir}/{subset}/{fold}/M0_101.pt',map_location='cuda'))
net=net.cuda().eval()
vx=[base.device_array(a) for a in (xva,yva,eva,base.prior_values(mva,tables['FULL']))]
def predict(x):
    with torch.no_grad():
        return torch.cat([net(x[i],None,vx[2][i],vx[3][i])["prediction"].reshape(-1)
                          for i in torch.arange(len(x),device="cuda").split(256)]).cpu().numpy()
base_pred=predict(vx[0])
rng=np.random.default_rng(7)
draws=[]
for s in [0.05,0.1,0.2]:
    reps=[]
    for _ in range(5):
        xn=vx[0]+torch.randn_like(vx[0])*s
        reps.append(predict(xn))
    draws.append(np.stack(reps).std(0))
sens=np.stack(draws)   # (3, n)
m=pd.read_csv(f'runs/{subset}/{fold}/M0_101.csv'); o=pd.read_csv(f'runs/{subset}/{fold}/OPS_101.csv')
y=m.y_true.to_numpy(); e0=m.y_pred.to_numpy()-y; e1=o.y_pred.to_numpy()-y
print('==',subset,'fold',fold,'n',len(y),'sens mean',np.round(sens.mean(1),3))
for i,s in enumerate([0.05,0.1,0.2]):
    print('  sigma %.2f: corr(sens,|e_M0|) %.3f  spearman %.3f  corr(sens,|e_M0|-|e_OPS|) %.3f'%(
        s,np.corrcoef(sens[i],np.abs(e0))[0,1],spearmanr(sens[i],np.abs(e0)).statistic,
        np.corrcoef(sens[i],np.abs(e0)-np.abs(e1))[0,1]))
def pooled(err):
    g=pd.DataFrame({'u':m.unit,'e2':err**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
print('  M0 %.3f OPS %.3f blend %.3f'%(pooled(e0),pooled(e1),pooled(0.5*e0+0.5*e1)))
s=sens[1]
for k in [1,2,4]:
    z=(s-s.mean())/(s.std()+1e-9)
    w=1/(1+np.exp(-k*z))
    print('  sens gate k=%d -> %.3f'%(k,pooled(w*e1+(1-w)*e0)))
