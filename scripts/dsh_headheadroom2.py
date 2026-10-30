"""Engine-grouped CV: does kNN on embeddings still beat the head without adjacent-window leakage?"""
import sys, glob, json, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
from candidates.model import CandidateModel
ck = sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/best_model.pt'))
cfg = json.load(open(sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/metrics.json'))[0]))['config']
prep = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='global')
val_set = prep[1]
m = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1,
                  dropout=cfg.get('dropout',0.15), use_baseline=True, use_ema_features=True,
                  use_cycle=True, use_graph=False, use_dual_head=False,
                  use_reliability=bool(cfg.get('use_reliability',False)))
sd = torch.load(ck[0], map_location='cpu')
m.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False); m.eval()
x = torch.as_tensor(val_set.x, dtype=torch.float32); ex = torch.as_tensor(val_set.extras, dtype=torch.float32)
y = np.asarray(val_set.y, dtype=float)
with torch.no_grad():
    out = m(x, extras=ex)
emb = out['embedding'].numpy(); head = out['prediction'].numpy().reshape(-1)
units = np.asarray(val_set.units) if hasattr(val_set, 'units') else None
if units is None:
    print('no unit ids on window set'); raise SystemExit
print(f'windows={len(y)} engines={len(np.unique(units))} head RMSE={np.sqrt(((head-y)**2).mean()):.2f}')
Z = (emb - emb.mean(0)) / (emb.std(0)+1e-9)
rng = np.random.default_rng(7)
u = np.unique(units); rng.shuffle(u)
folds = np.array_split(u, 5)
for k in (5, 10, 25):
    errs = []
    for f in folds:
        te = np.isin(units, f); tr = ~te
        if te.sum() == 0 or tr.sum() == 0: continue
        d2 = ((Z[te][:,None,:]-Z[tr][None,:,:])**2).sum(-1)
        nn = np.argsort(d2, axis=1)[:, :k]
        errs.append(y[tr][nn].mean(1) - y[te])
    e = np.concatenate(errs)
    print(f'  engine-grouped kNN k={k:2d}: RMSE={np.sqrt((e**2).mean()):6.2f}')
print(f'  (head on the same windows: {np.sqrt(((head-y)**2).mean()):.2f})')
