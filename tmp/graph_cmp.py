from pathlib import Path
import numpy as np, pandas as pd
HERE=Path('/home/user/rulb/tim_20261105/full_20261105')
def rmse(p):
    d=pd.read_csv(p); e=d.y_pred-d.y_true
    g=pd.DataFrame({'u':d.unit,'e2':e**2}).groupby('u').mean(); return float(np.sqrt(g.e2.mean()))
for fold in [2,3]:
    m=HERE/'runs/FD002'/str(fold)/'M0_101.csv'
    g=HERE/'runs3_w50_graph/FD002'/str(fold)/'GRAPH_101.csv'
    if not g.is_file(): print('fold',fold,'graph pending'); continue
    print('fold %d  M0 %.3f  GRAPH %.3f  delta %+.3f (%+.2f%%)'%(fold,rmse(m),rmse(g),rmse(g)-rmse(m),100*(rmse(g)/rmse(m)-1)))
