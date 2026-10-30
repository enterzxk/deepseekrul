import sys, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from candidates.model import CandidateModel
torch.manual_seed(0)
base = dict(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1, dropout=0.0,
            use_baseline=True, use_ema_features=True, use_cycle=True, use_dual_head=False,
            use_graph=True, use_reliability=True)
outs = {}
for mode in ('evidence','ones','half','const'):
    torch.manual_seed(0)
    m = CandidateModel(**base, gate_mode=mode); m.eval()
    x = torch.randn(8, 50, 14); ex = torch.randn(8, 21)
    with torch.no_grad():
        outs[mode] = m(x, extras=ex)['prediction'].numpy().reshape(-1)
print('max |pred_i - pred_j| at init (same seed):')
for a in ('evidence','ones','half','const'):
    for b in ('evidence','ones','half','const'):
        if a < b:
            print(f'  {a:9s} vs {b:9s}: {np.abs(outs[a]-outs[b]).max():.6f}')
