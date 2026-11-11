"""Recompute archived base/FULL test RMSE on all four subsets with the frozen evaluator."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.models import build_dspt
from paper_b.training.data import prepare_test
from paper_b.training.train import predict
from paper_b.training.metrics import summarize

ROOT = Path('/home/user/rulb')
CASES = {'base': dict(ms_mode='off', sa_mode='off', hi_source='none'),
         'FULL': dict(ms_mode='full', sa_mode='basic', hi_source='hf')}


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


rows = []
for subset in ('FD001', 'FD002', 'FD003', 'FD004'):
    te = prepare_test(subset, ROOT / 'data' / 'CMAPSS', span=70, split_seed=7, fold=0,
                      rul_cap=125, n_cond=6)['test']
    for cfg, kw in CASES.items():
        sd = remap(torch.load(ROOT / 'experiments' / f'ABL_{subset}_{cfg}' / subset / 'dspt_s101.pt',
                              map_location='cpu', weights_only=True))
        net = build_dspt(max_span=70, **kw); net.load_state_dict(sd, strict=False)
        p = predict(net, te, torch.device('cpu'), 'dual')
        m = summarize(p, te['y'])
        rows.append(dict(subset=subset, cfg=cfg, rmse=m['rmse'], nasa=m['nasa_sum'], bias=m['bias']))
        print(rows[-1])
pd.DataFrame(rows).to_csv('paper_b/results/legacy_all_repro.csv', index=False)
