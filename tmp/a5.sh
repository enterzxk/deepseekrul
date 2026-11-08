#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys, torch, numpy as np
from scipy.stats import spearmanr, pearsonr
sys.path.insert(0,'.'); sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import run_punet_stage as S
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = S.build('FD001', 0, 42, 50, 'experiments/ida/prior_boot')
model = S.make_model(device)
blob = torch.load('experiments/punet_boot/FD001/f0_s42/backbone.pt', map_location=device, weights_only=False)
model.load_state_dict(blob['state_dict'])
h_va, yb_va, y_va = S.extract(model, data['val_loader'], device)
sg = np.asarray(data['sg_va'], float)
err = np.abs(yb_va.numpy() - y_va.numpy())
print('n', len(sg), 'sigma min/med/max', round(sg.min(),2), round(np.median(sg),2), round(sg.max(),2))
print('spearman(sigma, |err|) =', round(float(spearmanr(sg, err).statistic), 4))
print('pearson (sigma, |err|) =', round(float(pearsonr(sg, err)[0]), 4))
# also correlation with squared error and with normalized error
print('spearman(sigma, err^2) =', round(float(spearmanr(sg, err**2).statistic), 4))
PYEOF
