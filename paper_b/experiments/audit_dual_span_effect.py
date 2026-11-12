"""Inference-time ablation of two DSPT-Base functions on real validation data.

Uses a saved A8-base checkpoint (no retraining) and the frozen validation split:
  * dual-span vs long-span vs short-span predictions;
  * cycle encoding present vs zeroed.
Reports engine-level validation RMSE for each variant.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..training.data import prepare_train
from ..training.metrics import per_engine_rmse
from ..training.train import _build, predict


def engine_rmse(pred, y, unit):
    return float(per_engine_rmse(unit, y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', default='paper_b/results/raw/A8_base_L70/FD001/s101')
    ap.add_argument('--data-dir', default='data/CMAPSS')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='paper_b/results/dual_span_effect.json')
    a = ap.parse_args()
    rd = Path(a.run_dir)
    cfg = json.loads((rd / 'config.json').read_text())
    ds = prepare_train(cfg['subset'], Path(a.data_dir), span=cfg['span'],
                       split_seed=cfg['split_seed'], fold=cfg['fold'],
                       val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                       legacy_rul=cfg.get('legacy_rul', False),
                       global_hi=cfg.get('global_hi', False))
    va = ds['val']
    dev = torch.device(a.device)
    net = _build(cfg).to(dev)
    net.load_state_dict(torch.load(rd / 'model.pt', map_location='cpu', weights_only=True))
    net.eval()

    out = {'run_dir': str(rd), 'subset': cfg['subset'], 'seed': cfg['seed']}
    preds = {}
    for mode in ('dual', 'long', 'short'):
        p = predict(net, va, dev, mode)
        preds[mode] = p
        out[f'val_eng_{mode}'] = engine_rmse(p, va['y'], va['unit'])
    out['val_eng_dual_minus_long'] = out['val_eng_dual'] - out['val_eng_long']
    out['val_eng_dual_minus_short'] = out['val_eng_dual'] - out['val_eng_short']
    out['pred_dual_vs_long_max_abs'] = float(np.abs(preds['dual'] - preds['long']).max())
    out['pred_dual_vs_short_max_abs'] = float(np.abs(preds['dual'] - preds['short']).max())

    # cycle ablation: same weights, cycle input zeroed and shuffled
    va0 = dict(va)
    va0['cycle'] = np.zeros_like(va['cycle'])
    p0 = predict(net, va0, dev, 'dual')
    out['val_eng_cycle_zeroed'] = engine_rmse(p0, va['y'], va['unit'])
    out['val_eng_cycle_delta'] = out['val_eng_cycle_zeroed'] - out['val_eng_dual']
    out['pred_cycle_max_abs_shift'] = float(np.abs(p0 - preds['dual']).max())
    out['pred_cycle_mean_abs_shift'] = float(np.abs(p0 - preds['dual']).mean())

    rng = np.random.default_rng(0)
    va1 = dict(va)
    va1['cycle'] = va['cycle'][rng.permutation(len(va['cycle']))]
    p1 = predict(net, va1, dev, 'dual')
    out['val_eng_cycle_shuffled'] = engine_rmse(p1, va['y'], va['unit'])
    out['val_eng_cycle_shuffle_delta'] = out['val_eng_cycle_shuffled'] - out['val_eng_dual']

    txt = json.dumps(out, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


if __name__ == '__main__':
    main()
