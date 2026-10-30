import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from candidates.model import CandidateModel
base = dict(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1, dropout=0.15,
            use_baseline=True, use_ema_features=True, use_cycle=True, use_dual_head=False)
for name, kw in (('A (no graph)', dict(use_graph=False)),
                 ('AG_ones', dict(use_graph=True, gate_mode='ones')),
                 ('AG_const', dict(use_graph=True, gate_mode='const')),
                 ('AG_evidence', dict(use_graph=True, gate_mode='evidence'))):
    m = CandidateModel(**base, **kw)
    n = sum(p.numel() for p in m.parameters())
    print(f'{name:14s} params={n:,}')
