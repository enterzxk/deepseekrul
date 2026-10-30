import sys, glob, json, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import run_candidate as rc
import candidates.model as M
ck = sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/best_model.pt'))[0]
cfg = json.load(open(sorted(glob.glob('experiments/order03/M1_global/FD002/A_s*/metrics.json'))[0]))['config']
val = rc.prepare('FD002', 7, 50, 125.0, condition_aware=True, n_cond=6, ref_mode='global')[1]
m = M.CandidateModel(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1,
                    dropout=cfg.get('dropout',0.15), use_baseline=True, use_ema_features=True,
                    use_cycle=True, use_graph=False, use_dual_head=False,
                    use_reliability=bool(cfg.get('use_reliability',False)))
sd = torch.load(ck, map_location='cpu')
m.load_state_dict(sd if isinstance(sd, dict) else sd.state_dict(), strict=False); m.eval()
x = torch.as_tensor(val.x, dtype=torch.float32); ex = torch.as_tensor(val.extras, dtype=torch.float32)
y = np.asarray(val.y, dtype=float)
CLS = M.SensorGraphAttention
orig = CLS.forward
def make(scale):
    def fwd(self, h, cond_t):
        update, _ = self.attn(h, h, h, need_weights=False)
        a = 0.5*(torch.sigmoid(self.gate(cond_t))-0.5) if scale is None else torch.full((h.shape[0],1,1), scale)
        return h + a*self.norm(update)
    return fwd
for label, scale in (('learned evidence', None), ('alpha=0.00', 0.0), ('alpha=0.10', 0.1), ('alpha=0.25', 0.25), ('alpha=0.50', 0.5)):
    CLS.forward = make(scale)
    with torch.no_grad():
        p = m(x, extras=ex)['prediction'].numpy().reshape(-1)
    print(f'{label:18s} RMSE={np.sqrt(((p-y)**2).mean()):7.3f}')
CLS.forward = orig
