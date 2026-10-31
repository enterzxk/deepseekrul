import sys, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from baselines import build_baseline
from candidates.model import CandidateModel
reg = {"cnn1d":"cnn1d","bilstm":"bilstm","gru":"gru_attention","tcn":"tcn","tf":"transformer"}
for k, v in reg.items():
    for dm, nl in ((64,1),(128,2)):
        m = build_baseline(v, num_sensors=14, d_model=dm, dropout=0.15)
        print(f"{k:8s} d_model={dm:3d} layers={nl}  params={sum(p.numel() for p in m.parameters()):,}")
a = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1, dropout=0.15,
                   use_baseline=True, use_ema_features=True, use_cycle=True,
                   use_graph=False, use_dual_head=False)
print(f"{'A':8s} d_model= 64 layers=1  params={sum(p.numel() for p in a.parameters()):,}")
