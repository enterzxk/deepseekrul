"""Engine-grouped kNN with reconstructed unit ids (validated by total window count)."""
import sys, glob, json, warnings
import numpy as np, torch
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
from candidates.model import CandidateModel
from dataset import load_cmapss_train
prep = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='global')
val_set, val_units = prep[1], list(prep[3])
df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
life = df.groupby('unit')['cycle'].max()
counts = [int(life[u]) - 49 for u in val_units]
print('sum counts =', sum(counts), '| windows =', len(val_set))
if sum(counts) != len(val_set):
    print('count mismatch - aborting'); raise SystemExit
units = np.concatenate([[u]*c for u, c in zip(val_units, counts)])
cfg = json.load(open(sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/metrics.json'))[0]))['config']
m = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1,
                  dropout=cfg.get('dropout',0.15), use_baseline=True, use_ema_features=True,
                  use_cycle=True, use_graph=False, use_dual_head=False,
                  use_reliability=bool(cfg.get('use_reliability',False)))
sd = torch.load(sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/best_model.pt'))[0], map_location='cpu')
m.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False); m.eval()
x = torch.as_tensor(val_set.x, dtype=torch.float32); ex = torch.as_tensor(val_set.extras, dtype=torch.float32)
y = np.asarray(val_set.y, dtype=float)
with torch.no_grad(): out = m(x, extras=ex)
emb = out['embedding'].numpy(); head = out['prediction'].numpy().reshape(-1)
Z = (emb - emb.mean(0)) / (emb.std(0)+1e-9)
print(f'head RMSE={np.sqrt(((head-y)**2).mean()):.2f}')
rng = np.random.default_rng(7); u = np.unique(units); rng.shuffle(u); folds = np.array_split(u, 5)
for k in (5, 10, 25):
    errs = []
    for f in folds:
        te = np.isin(units, f); tr = ~te
        d2 = ((Z[te][:,None,:]-Z[tr][None,:,:])**2).sum(-1)
        nn = np.argsort(d2, axis=1)[:, :k]
        errs.append(y[tr][nn].mean(1) - y[te])
    e = np.concatenate(errs)
    print(f'  engine-grouped kNN k={k:2d}: RMSE={np.sqrt((e**2).mean()):6.2f}')
