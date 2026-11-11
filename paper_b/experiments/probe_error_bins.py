"""Per-RUL-bin error profile of archived base/FULL checkpoints on FD001 test."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.models import build_dspt
from paper_b.training.data import prepare_test
from paper_b.training.train import predict

ROOT = Path('/home/user/rulb')
CASES = {'base': dict(ms_mode='off', sa_mode='off', hi_source='none'),
         'FULL': dict(ms_mode='full', sa_mode='basic', hi_source='hf')}
BINS = [(0, 50), (50, 80), (80, 100), (100, 125), (125, 200)]


def remap(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith('multi_conv.'):
            idx = int(k.split('.')[1]); rest = '.'.join(k.split('.')[2:])
            out[{0: 'conv3', 1: 'conv5', 2: 'conv7'}[idx] + '.' + rest] = v
        elif k.startswith('mscale_proj.'):
            out['ms_proj.' + '.'.join(k.split('.')[1:])] = v
        elif k.startswith('slot_score.'):
            continue
        else:
            out[k] = v
    return out


te = prepare_test('FD001', ROOT / 'data' / 'CMAPSS', span=70, split_seed=7, fold=0,
                  rul_cap=125, n_cond=6)['test']
rows = []
for cfg, kw in CASES.items():
    sd = remap(torch.load(ROOT / 'experiments' / f'ABL_FD001_{cfg}' / 'FD001' / 'dspt_s101.pt',
                          map_location='cpu', weights_only=True))
    net = build_dspt(max_span=70, **kw); net.load_state_dict(sd, strict=False)
    p = predict(net, te, torch.device('cpu'), 'dual')
    for i in range(len(p)):
        rows.append(dict(cfg=cfg, unit=int(te['unit'][i]), y_true=float(te['y'][i]), y_pred=float(p[i])))
df = pd.DataFrame(rows)
df.to_csv('paper_b/results/legacy_preds_bins.csv', index=False)
for cfg, g in df.groupby('cfg'):
    print('==', cfg, 'overall RMSE', round(float(np.sqrt(np.mean((g.y_pred-g.y_true)**2))), 3))
    for lo, hi in BINS:
        s = g[(g.y_true >= lo) & (g.y_true < hi)]
        if len(s) == 0: continue
        e = s.y_pred - s.y_true
        print(f'  [{lo:3d},{hi:3d}) n={len(s):3d} rmse={np.sqrt(np.mean(e**2)):6.2f} bias={e.mean():7.2f}')
