#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys, torch, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import run_punet_stage as S
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = S.build('FD001', 0, 42, 50, 'experiments/ida/prior')
model = S.make_model(device)
blob = torch.load('experiments/punet/FD001/f0_s42/backbone.pt', map_location=device, weights_only=False)
model.load_state_dict(blob['state_dict'])
h_tr, yb_tr, y_tr = S.extract(model, data['train_loader'], device)
h_va, yb_va, y_va = S.extract(model, data['val_loader'], device)
sg_tr = torch.tensor(np.asarray(data['mu_tr'])*0 + np.asarray(data['sg_tr']), dtype=torch.float32)
cut_tr = np.asarray(data['meta_tr'])[:,1].astype(float)
tau_tr = torch.tensor(cut_tr/np.maximum(np.asarray(data['mu_tr']),1e-6), dtype=torch.float32)
from src.punet.train import fit_heads
out = fit_heads(h_tr, sg_tr, tau_tr, y_tr, yb_tr, d_model=h_tr.shape[1], epochs=100, lr=1e-3, device=device, seed=42)
a = float(out['scale_head'].exponent)
w = out['w']
print('learned a =', round(a,5))
print('w stats min/med/max', round(float(w.min()),4), round(float(w.median()),4), round(float(w.max()),4))
print('sigma stats min/med/max', round(float(sg_tr.min()),3), round(float(sg_tr.median()),3), round(float(sg_tr.max()),3))
PYEOF
