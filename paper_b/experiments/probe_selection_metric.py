"""Compare window-level vs engine-level validation against test for archived checkpoints."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_b.models import build_dspt
from paper_b.training.data import prepare_train, prepare_test, make_train_windows
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
for cfg, kw in CASES.items():
    sd = remap(torch.load(ROOT / 'experiments' / f'ABL_FD001_{cfg}' / 'FD001' / 'dspt_s101.pt',
                          map_location='cpu', weights_only=True))
    net = build_dspt(max_span=70, **kw)
    net.load_state_dict(sd, strict=False)
    ds = prepare_train('FD001', ROOT / 'data' / 'CMAPSS', span=70, split_seed=7, fold=0,
                       rul_cap=125, n_cond=6)
    allw = make_train_windows(ds['va'], ds['va_s'], ds['lv'], ds['pop'], span=70, hi_map=None)
    p_win = predict(net, allw, torch.device('cpu'), 'dual')
    win = rmse(p_win, allw['y'])
    p_cut = predict(net, ds['val'], torch.device('cpu'), 'dual')
    eng = rmse(p_cut, ds['val']['y'])
    te = prepare_test('FD001', ROOT / 'data' / 'CMAPSS', span=70, split_seed=7, fold=0,
                      rul_cap=125, n_cond=6)['test']
    p_te = predict(net, te, torch.device('cpu'), 'dual')
    test = rmse(p_te, te['y'])
    rows.append(dict(cfg=cfg, window_val=win, engine_val=eng, test=test))
    print(rows[-1])
pd.DataFrame(rows).to_csv('paper_b/results/selection_metric_probe.csv', index=False)
