#!/bin/bash
cd ~/deepseek-rul
export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:${LD_LIBRARY_PATH}
PY=/home/yndx/miniconda3/envs/a40_env/bin/python
$PY - <<'PYEOF'
import sys, torch, numpy as np, pandas as pd
sys.path.insert(0,'.'); sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import run_punet_stage as S
from src.punet.train import fit_heads
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = S.build('FD003', 0, 42, 50, 'experiments/ida/prior_boot')
model = S.make_model(device)
blob = torch.load('experiments/punet_5fold/FD003/f0_s42/backbone.pt', map_location=device, weights_only=False)
model.load_state_dict(blob['state_dict'])
h_tr, yb_tr, y_tr = S.extract(model, data['train_loader'], device)
h_va, yb_va, y_va = S.extract(model, data['val_loader'], device)
units = np.asarray(data['meta_va'])[:,0]
sg_tr = torch.tensor(np.asarray(data['sg_tr'], dtype=float), dtype=torch.float32)
cut_tr = np.asarray(data['meta_tr'])[:,1].astype(float)
tau_tr = torch.tensor(cut_tr/np.maximum(np.asarray(data['mu_tr']),1e-6), dtype=torch.float32)
out = fit_heads(h_tr, sg_tr, tau_tr, y_tr, yb_tr, d_model=h_tr.shape[1], epochs=100, lr=1e-3, device=device, seed=42)
print('learned a =', round(float(out['scale_head'].exponent), 4))
print('train w min/med/max', [round(float(x),4) for x in (out['w'].min(), out['w'].median(), out['w'].max())])
sg_va = torch.tensor(np.asarray(data['sg_va'], dtype=float), dtype=torch.float32)
eng = pd.Series(np.asarray(data['sg_va'], dtype=float)).groupby(units).mean()
s_e = torch.tensor([float(eng[int(u)]) for u in units], dtype=torch.float32)
tau_va = torch.tensor(np.asarray(data['meta_va'])[:,1].astype(float)/np.maximum(np.asarray(data['mu_va']),1e-6), dtype=torch.float32)
with torch.no_grad():
    w0 = out['scale_head'](h_va, (0.0*s_e + 1.0*s_e), tau_va)
    w5 = out['scale_head'](h_va, 0.5*s_e + 0.5*sg_va, tau_va)
    w1 = out['scale_head'](h_va, sg_va, tau_va)
for name, w in [('lam0', w0), ('lam.5', w5), ('lam1', w1)]:
    print(name, 'median w', round(float(w.median()),4), 'iqr', round(float(w.quantile(0.75)-w.quantile(0.25)),4))
PYEOF
