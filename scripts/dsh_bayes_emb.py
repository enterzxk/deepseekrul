"""Bayes-error estimate in the model's own embedding space (inference only)."""
import sys, glob, json, warnings
import numpy as np, torch
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
from candidates.model import CandidateModel

ck = sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/best_model.pt'))
mf = sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/metrics.json'))
if not ck: print('no checkpoint'); raise SystemExit
cfg = json.load(open(mf[0]))['config']
print('config keys:', sorted(cfg)[:8])
prep = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='global')
train_set, val_set = prep[0], prep[1]
m = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1,
                  dropout=cfg.get('dropout', 0.15), use_baseline=True,
                  use_ema_features=True, use_cycle=True, use_graph=False,
                  use_dual_head=False, use_reliability=bool(cfg.get('use_reliability', False)))
sd = torch.load(ck[0], map_location='cpu')
m.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False)
m.eval()
x = torch.as_tensor(val_set.x[:512], dtype=torch.float32)
ex = torch.as_tensor(val_set.extras[:512], dtype=torch.float32)
y = np.asarray(val_set.y[:512], dtype=float)
with torch.no_grad():
    out = m(x, extras=ex)
emb = out['embedding'].detach().numpy()
pred = out['prediction'].detach().numpy().reshape(-1)
print(f'windows={len(y)} embedding_dim={emb.shape[1]}')
print(f'val RMSE on this subset: {np.sqrt(((pred-y)**2).mean()):.2f}')
Z = (emb - emb.mean(0)) / (emb.std(0) + 1e-9)
n = len(y)
d2 = ((Z[:,None,:]-Z[None,:,:])**2).sum(-1)
np.fill_diagonal(d2, np.inf)
for k in (5, 10, 20):
    nn = np.argsort(d2, axis=1)[:, :k]
    var = y[nn].var(axis=1)
    print(f'k={k:2d}: mean cond var={var.mean():8.1f} -> implied floor RMSE={np.sqrt(var.mean()):6.2f}')
