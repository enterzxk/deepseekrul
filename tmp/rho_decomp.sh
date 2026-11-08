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
print('subset fold | overall  within(mean of per-engine)  between(engine means)')
for sub in ['FD001','FD002','FD003','FD004']:
    for fold in range(2):
        data = S.build(sub, fold, 42, 50, 'experiments/ida/prior_boot')
        model = S.make_model(device)
        blob = torch.load(f'experiments/punet_5fold/{sub}/f{fold}_s42/backbone.pt', map_location=device, weights_only=False)
        model.load_state_dict(blob['state_dict'])
        h, yb, y = S.extract(model, data['val_loader'], device)
        sg = np.asarray(data['sg_va'], float); units = np.asarray(data['meta_va'])[:,0]
        err = np.abs(yb.numpy()-y.numpy())
        overall = float(spearmanr(sg, err).statistic)
        df = pd.DataFrame({'u':units,'s':sg,'e':err})
        per = df.groupby('u').apply(lambda g: spearmanr(g.s, g.e).statistic if len(g)>3 else np.nan)
        between = float(spearmanr(df.groupby('u').s.mean(), df.groupby('u').e.mean()).statistic)
        print(f'{sub} f{fold} | {overall:+.3f}  {float(np.nanmean(per)):+.3f}  {between:+.3f}')
PYEOF
