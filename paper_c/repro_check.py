"""A0.2 reproducibility: same fold/seed, 5 epochs, deterministic, twice."""
import json
from pathlib import Path

import torch

from paper_c.train import run_fold

r1 = run_fold('FD001', 'gru', 50, 0, epochs=5, ckpt_dir='paper_c/repro1',
              device='cuda:1', deterministic=True)
r2 = run_fold('FD001', 'gru', 50, 0, epochs=5, ckpt_dir='paper_c/repro2',
              device='cuda:1', deterministic=True)
s1 = torch.load('paper_c/repro1/FD001_gru_L50_f0.pt', map_location='cpu')['state_dict']
s2 = torch.load('paper_c/repro2/FD001_gru_L50_f0.pt', map_location='cpu')['state_dict']
maxd = max(float((s1[k] - s2[k]).abs().max()) for k in s1)
out = dict(run1=r1, run2=r2, max_state_diff=maxd, A0_2_pass=maxd < 1e-5)
Path('results/paper_c').mkdir(parents=True, exist_ok=True)
Path('results/paper_c/p0_repro.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
