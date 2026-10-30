import sys, json, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
from candidates.model import CandidateModel
run = 'experiments/split8/M1_global/FD002/A_s2027'
cfg = json.load(open(run + '/metrics.json'))['config']
prep = rc.prepare('FD002', 8, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='global')
matched = prep[7]
print('matched windows:', len(matched), '| archived rows:', len(pd.read_csv(run + '/matched_cutoff_predictions.csv')))
m = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1,
                  dropout=cfg.get('dropout',0.15), use_baseline=cfg.get('use_baseline',True),
                  use_ema_features=cfg.get('use_ema_features',True), use_cycle=cfg.get('use_cycle',True),
                  use_graph=cfg.get('use_graph',False), use_dual_head=cfg.get('use_dual_head',False),
                  use_reliability=bool(cfg.get('use_reliability',False)))
sd = torch.load(run + '/best_model.pt', map_location='cpu')
m.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False); m.eval()
x = torch.as_tensor(matched.x, dtype=torch.float32); ex = torch.as_tensor(matched.extras, dtype=torch.float32)
with torch.no_grad(): p = m(x, extras=ex)['prediction'].numpy().reshape(-1)
y = np.asarray(matched.y, dtype=float)
arc = pd.read_csv(run + '/matched_cutoff_predictions.csv')
print(f'fresh RMSE={np.sqrt(((np.clip(p,0,125)-y)**2).mean()):.4f}  archived RMSE={np.sqrt(((np.clip(arc.y_pred.to_numpy(float),0,125)-arc.y_true.to_numpy(float))**2).mean()):.4f}')
if len(p)==len(arc):
    print(f'corr={np.corrcoef(p, arc.y_pred.to_numpy(float))[0,1]:.6f}  max|diff|={np.abs(p-arc.y_pred.to_numpy(float)).max():.6f}')
