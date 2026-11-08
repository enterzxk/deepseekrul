#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys, torch, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,'.'); sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import run_punet_stage as S
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for sub in ['FD001','FD002','FD003','FD004']:
    rs=[]
    for fold in range(5):
        try:
            data = S.build(sub, fold, 42, 50, 'experiments/ida/prior_boot')
            model = S.make_model(device)
            blob = torch.load(f'experiments/punet_5fold/{sub}/f{fold}_s42/backbone.pt', map_location=device, weights_only=False)
            model.load_state_dict(blob['state_dict'])
            h, yb, y = S.extract(model, data['val_loader'], device)
            sg = np.asarray(data['sg_va'], float)
            err = np.abs(yb.numpy()-y.numpy())
            rs.append(float(spearmanr(sg, err).statistic))
        except Exception as e:
            rs.append(float('nan'))
    print(sub, 'per-fold spearman', [round(r,3) for r in rs], 'mean', round(float(np.nanmean(rs)),3))
PYEOF
