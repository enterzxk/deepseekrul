import pandas as pd
from pathlib import Path
H=Path('/home/user/rulb/tim_20261105/full_20261105')
a=pd.read_csv(H/'ABLATIONS.csv'); v=pd.read_csv(H/'ABLATIONS_V3.csv')
v['note']=v['note'].str.replace(' E (fold 2)',' cross-sensor graph attention (~200 params) (fold 2)',regex=False)
out=pd.concat([a,v],ignore_index=True)
out.to_csv(H/'ABLATIONS.csv',index=False)
print('rows', len(out)); print(out.tail(3)[['subset','variant','delta','note']].to_string(index=False))