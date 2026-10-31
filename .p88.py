import sys
sys.path.insert(0, "src")
from baselines import build_baseline
for k, v in {"cnn1d":"cnn1d","bilstm":"bilstm","gru":"gru_attention","tcn":"tcn","tf":"transformer"}.items():
    m = build_baseline(v, num_sensors=14, d_model=88, dropout=0.15)
    print(f"{k:8s} {sum(p.numel() for p in m.parameters()):,}")
