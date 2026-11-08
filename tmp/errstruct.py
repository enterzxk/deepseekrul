import pandas as pd, numpy as np
from pathlib import Path
for subset in ['FD002','FD004']:
    b=[]
    for k in range(5):
        p=Path(f'runs/{subset}/{k}/M0_101.csv')
        if p.is_file(): b.append(pd.read_csv(p).assign(fold=k))
    if not b: continue
    d=pd.concat(b,ignore_index=True); d['err']=d.y_pred-d.y_true
    g=d.groupby('unit').agg(mse=('err',lambda x:(x**2).mean()),bias=('err','mean'),
                            life=('y_true',lambda x: 0))
    lives=d.groupby('unit').apply(lambda x: x.y_true.max()+x.end_cycle.max(), include_groups=False)
    g['life']=lives
    g['rmse']=np.sqrt(g.mse)
    g=g.sort_values('mse',ascending=False)
    tot=g.mse.mean()
    print('==',subset,'engines',len(g),'pooled RMSE %.3f'%np.sqrt(tot))
    for frac in [0.05,0.1,0.2,0.3]:
        n=int(len(g)*frac)
        rest=g.iloc[n:]
        print('  drop worst %2d%% (%2d eng): RMSE if perfect %.3f | remaining RMSE %.3f'%(
            frac*100,n,np.sqrt(rest.mse.mean()*len(rest)/len(g)),np.sqrt(rest.mse.mean())))
    print('  top5 rmse:',np.round(g.rmse.head(5).values,1),' their life:',g.life.head(5).values)
    q=pd.qcut(g.life,4,labels=['Q1 short','Q2','Q3','Q4 long'])
    print(g.groupby(q,observed=True).agg(n=('rmse','size'),rmse=('rmse','mean'),bias=('bias','mean')).round(2).to_string())
