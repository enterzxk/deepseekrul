import json
from pathlib import Path
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
rows=[]
for seed in [101,202,303]:
    d=json.loads((HERE/f'light_seed{seed}.json').read_text())
    for subset,entry in d['subsets'].items():
        for arm,m in entry['metrics'].items():
            rows.append((seed,subset,arm,m['rmse']))
import pandas as pd
df=pd.DataFrame(rows,columns=['seed','subset','arm','rmse'])
piv=df.pivot_table(index=['subset','arm'],columns='seed',values='rmse')
piv['mean']=piv.mean(axis=1); piv['sd']=piv[[101,202,303]].std(axis=1)
print('light budget (24 epochs, stride 5) 3-seed summary')
print(piv.round(3).to_string())
for subset in ['FD002','FD004']:
    a=piv.loc[(subset,'M0'),'mean']; b=piv.loc[(subset,'FULL'),'mean']; c=piv.loc[(subset,'NORM'),'mean']
    print('%s: M0 %.3f FULL %.3f (%.2f%%) NORM %.3f (%.2f%%)'%(subset,a,b,100*(b/a-1),c,100*(c/a-1)))
