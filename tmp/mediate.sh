#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys, torch, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,'.'); sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import run_punet_stage as S
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('subset | rho(sigma,err)  rho(T,sigma)  rho(T,err)  rho(T,err/T)  rho(sigma,err/T)')
for sub in ['FD001','FD002','FD003','FD004']:
    data = S.build(sub, 0, 42, 50, 'experiments/ida/prior_boot')
    model = S.make_model(device)
    blob = torch.load(f'experiments/punet_5fold/{sub}/f0_s42/backbone.pt', map_location=device, weights_only=False)
    model.load_state_dict(blob['state_dict'])
    h, yb, y = S.extract(model, data['val_loader'], device)
    sg = np.asarray(data['sg_va'], float); units = np.asarray(data['meta_va'])[:,0]
    cut = np.asarray(data['meta_va'])[:,1].astype(float)
    err = np.abs(yb.numpy()-y.numpy())
    df = pd.DataFrame({'u':units,'s':sg,'e':err,'c':cut})
    per = df.groupby('u').agg(s=('s','mean'), e=('e','mean'), life=('c','max'))
    r = lambda a,b: round(float(spearmanr(a,b).statistic),3)
    life = per["life"]
    rel = per["e"] / life
    a1 = r(per.s, per.e); a2 = r(life, per.s); a3 = r(life, per.e)
    a4 = r(life, rel); a5 = r(per.s, rel)
    print(sub, "| sigma-err", a1, "| life-sigma", a2, "| life-err", a3, "| life-relerr", a4, "| sigma-relerr", a5)
PYEOF
