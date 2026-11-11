"""Can the archived checkpoints reproduce the archived test predictions?

Legacy checkpoints store the three convolutions as multi_conv.{0,1,2}; the
current model names them conv3/conv5/conv7. Remap before loading.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.models import build_dspt
from paper_b.training.data import prepare_test
from paper_b.training.train import predict
from paper_b.training.metrics import rmse

ROOT = Path('/home/user/rulb')
CASES = {
    'base': dict(ms_mode='off', sa_mode='off', hi_source='none'),
    'FULL': dict(ms_mode='full', sa_mode='basic', hi_source='hf'),
    'SA': dict(ms_mode='off', sa_mode='basic', hi_source='none'),
    'HI': dict(ms_mode='off', sa_mode='off', hi_source='hf'),
}


def remap(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith('multi_conv.'):
            idx = int(k.split('.')[1])
            rest = '.'.join(k.split('.')[2:])
            out[{0: 'conv3', 1: 'conv5', 2: 'conv7'}[idx] + '.' + rest] = v
        elif k.startswith('mscale_proj.'):
            out['ms_proj.' + '.'.join(k.split('.')[1:])] = v
        elif k.startswith('slot_score.'):
            continue
        else:
            out[k] = v
    return out


for cfg, kw in CASES.items():
    ck = ROOT / 'experiments' / f'ABL_FD001_{cfg}' / 'FD001' / 'dspt_s101.pt'
    sd = remap(torch.load(ck, map_location='cpu', weights_only=True))
    net = build_dspt(max_span=70, **kw)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    ds = prepare_test('FD001', ROOT / 'data' / 'CMAPSS', span=70, split_seed=7, fold=0,
                      rul_cap=125, n_cond=6)
    te = ds['test']
    p = predict(net, te, torch.device('cpu'), 'dual')
    ref = pd.read_csv(ROOT / 'experiments' / 'ABLTEST' / f'FD001_{cfg}.csv')
    d = p - ref.y_pred.values
    print(f'== {cfg}: our_rmse_vs_truth={rmse(p, te["y"]):.3f} '
          f'archived_rmse={rmse(ref.y_pred.values, te["y"]):.3f} '
          f'corr={np.corrcoef(p, ref.y_pred.values)[0,1]:.4f} maxdiff={np.abs(d).max():.2f} '
          f'missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}')
