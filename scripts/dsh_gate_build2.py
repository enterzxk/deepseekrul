import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from candidates.model import CandidateModel
base = dict(num_sensors=14, window_size=50, d_model=64, num_heads=4, num_layers=1, dropout=0.15,
            use_baseline=True, use_ema_features=True, use_cycle=True, use_dual_head=False)
for name, kw in (('A (no graph, no rel)', dict(use_graph=False)),
                 ('graph only', dict(use_graph=True, use_reliability=False)),
                 ('graph+rel evidence', dict(use_graph=True, use_reliability=True, gate_mode='evidence')),
                 ('graph+rel ones', dict(use_graph=True, use_reliability=True, gate_mode='ones')),
                 ('graph+rel const', dict(use_graph=True, use_reliability=True, gate_mode='const'))):
    m = CandidateModel(**base, **kw)
    print(f'{name:22s} params={sum(p.numel() for p in m.parameters()):,}')
