"""Does the saved weight reproduce the archived predictions? (inference only)"""
import sys, glob, json, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
from candidates.model import CandidateModel
ck = sorted(glob.glob('experiments/order03/M1_global/FD002/A_s2027/best_model.pt'))
mf = sorted(glob.glob('experiments/order03/M1_global/FD002/A_s2027/metrics.json'))
if not ck: print('no ckpt'); raise SystemExit
cfg = json.load(open(mf[0]))['config']
prep = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='global')
val = prep[1]; val_units = list(prep[3])
m = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1,
                  dropout=cfg.get('dropout',0.15), use_baseline=cfg.get('use_baseline',True),
                  use_ema_features=cfg.get('use_ema_features',True), use_cycle=cfg.get('use_cycle',True),
                  use_graph=cfg.get('use_graph',False), use_dual_head=cfg.get('use_dual_head',False),
                  use_reliability=bool(cfg.get('use_reliability',False)))
sd = torch.load(ck[0], map_location='cpu')
m.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False); m.eval()
x = torch.as_tensor(val.x, dtype=torch.float32); ex = torch.as_tensor(val.extras, dtype=torch.float32)
with torch.no_grad(): p = m(x, extras=ex)['prediction'].numpy().reshape(-1)
y = np.asarray(val.y, dtype=float)
arc = pd.read_csv('results/dsh/review_r1/rul_archive/predictions/order03__M1_global__A_s2027__matched.csv')
print(f'windows={len(p)} val_units={len(val_units)} archived_rows={len(arc)}')
print(f'fresh predictions: RMSE(clip)={np.sqrt((np.clip(p,0,125)-y)**2).mean():.3f}')
ea = arc.y_pred.to_numpy(float); ya = arc.y_true.to_numpy(float)
print(f'archived          : RMSE(clip)={np.sqrt((np.clip(ea,0,125)-ya)**2).mean():.3f}')
if len(p) == len(arc):
    print(f'max |fresh - archived| = {np.abs(p - ea).max():.6f}')
