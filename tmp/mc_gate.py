"""Does MC-dropout disagreement identify unreliable individual readings?"""
import sys
import numpy as np, pandas as pd, torch
from pathlib import Path
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb","/home/user/rulb/tim_20261105/full_20261105"]
import run_fold as base, run_ida as ida
from scipy.stats import spearmanr
from run_full_fold3 import build_net

subset=sys.argv[1]; fold=int(sys.argv[2])
folds,df=ida.outer_folds(subset)
df=df.sort_values(["unit","cycle"]).reset_index(drop=True)
train,valid=folds[fold]
tables=base.prior_tables(df,subset,train,valid,fold)
tr,va=ida.prep_fold(df,subset,fold,train,valid)
xva,yva,eva,mva=va[:4]
net=build_net('M0',50)
net.load_state_dict(torch.load(f'runs3_w50/{subset}/{fold}/M0_101.pt',map_location='cuda'))
net=net.cuda()
vx=[base.device_array(a) for a in (xva,yva,eva,base.prior_values(mva,tables["FULL"]))]
net.train()
preds=[]
with torch.no_grad():
    for _ in range(20):
        preds.append(torch.cat([net(vx[0][i],None,vx[2][i],vx[3][i])["prediction"].reshape(-1)
                                for i in torch.arange(len(vx[0]),device="cuda").split(256)]).cpu().numpy())
P=np.stack(preds); mean=P.mean(0); sd=P.std(0)
m=pd.read_csv(f'runs/{subset}/{fold}/M0_101.csv'); o=pd.read_csv(f'runs/{subset}/{fold}/OPS_101.csv')
y=m.y_true.to_numpy(); e0=m.y_pred.to_numpy()-y; e1=o.y_pred.to_numpy()-y
print('==',subset,'fold',fold,'n',len(y))
print('  sd(MC) mean %.2f  corr(sd,|e_M0|) %.3f  spearman %.3f'%(sd.mean(),np.corrcoef(sd,np.abs(e0))[0,1],spearmanr(sd,np.abs(e0)).statistic))
print('  corr(sd, |e_M0|-|e_OPS|) %.3f'%np.corrcoef(sd,np.abs(e0)-np.abs(e1))[0,1])
def pooled(err):
    g=pd.DataFrame({'u':m.unit,'e2':err**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
print('  M0 %.3f OPS %.3f blend0.5 %.3f'%(pooled(e0),pooled(e1),pooled(0.5*e0+0.5*e1)))
for k in [0.5,1.0,2.0]:
    w=1/(1+np.exp(-(sd-sd.mean())/ (sd.std()+1e-9) *k))
    print('  sd-gated (k=%.1f) -> %.3f'%(k,pooled(w*e1+(1-w)*e0)))
