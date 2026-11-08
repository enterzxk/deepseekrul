#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys, json, torch, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import run_punet_stage as S
args = type('A', (), dict(subset='FD001', fold=0, seed=42, window=50, prior_dir='experiments/ida/prior'))()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = S.build('FD001', 0, 42, 50, 'experiments/ida/prior')
model = S.make_model(device)
blob = torch.load('experiments/punet/FD001/f0_s42/backbone.pt', map_location=device)
model.load_state_dict(blob['state_dict'])
print('ckpt stats:', blob['stats'])
# direct prediction RMSE
model.eval()
errs=[]
with torch.no_grad():
    for x,y,cond,extras,prior in data['val_loader']:
        p = model(x.to(device), cond.to(device), extras.to(device), prior.to(device))['prediction'].reshape(-1).cpu()
        errs.append(((p - y.reshape(-1))**2).numpy())
rmse_direct = float(np.sqrt(np.concatenate(errs).mean()))
h, yb, yt = S.extract(model, data['val_loader'], device)
rmse_extract = float(torch.sqrt(((yb - yt)**2).mean()))
print('rmse_direct', round(rmse_direct,3), 'rmse_extract', round(rmse_extract,3))
print('h shape', tuple(h.shape), 'yb shape', tuple(yb.shape))
PYEOF
