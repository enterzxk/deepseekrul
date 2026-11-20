"""Phase 2 fault robustness: compare M0 / M0Aug / M1 / M2 on the same faulted
validation windows. Recovery uses M0's clean/faulted RMSE as the denominator:
    recovery_A = (R1_M0 - R_A_faulted) / (R1_M0 - R0_M0)
Test sealed.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from paper_b.models import build_dspt
from paper_b.modules.model_rhse_dspt import RHSE_DSPT
from paper_b.modules.stress_suite import inject
from paper_b.training.data import prepare_train


def load_model(arm, subset, seed, root, dev):
    if arm in ('M0', 'M0Aug'):
        net = build_dspt(num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
                         max_span=70, ms_mode='full', sa_mode='basic',
                         hi_source='hf', hi_mono='none', use_pad_mask=True,
                         use_cycle=True, n_cond=6)
    else:
        net = RHSE_DSPT(np.zeros((14, 3), np.float32))
    ckpt = Path(root) / f'phase2_{arm}_{subset}_s{seed}' / 'ckpt' / 'best.pt'
    net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    return net.to(dev).eval()


def predict(net, row, x_win, dev):
    x = torch.tensor(x_win[None], dtype=torch.float32, device=dev)
    m = torch.tensor(row['mask'][None], dtype=torch.float32, device=dev)
    b = torch.tensor(row['baseline'][None], dtype=torch.float32, device=dev)
    cy = torch.tensor([row['cycle']], dtype=torch.float32, device=dev)
    cd = torch.tensor([row['cond']], dtype=torch.long, device=dev)
    with torch.no_grad():
        out = net(x, mask=m, baseline=b, cycle=cy, cond=cd, mode='dual')
    return float(out['prediction'].reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--fault', default='bias', choices=['bias', 'drift', 'noise', 'stuck'])
    ap.add_argument('--mag', type=float, default=3.0)
    ap.add_argument('--sensors', default='0,5,10')
    ap.add_argument('--onset', type=float, default=0.5)
    ap.add_argument('--root', default='/home/user/sprint24/experiments/phase2')
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    ds = prepare_train(a.subset, a.data_dir, span=70, split_seed=7, fold=0,
                       val_frac=0.2, rul_cap=125.0, n_cond=6, cuts=(0.4, 0.55, 0.7, 0.85, 0.95))
    va = ds['val']
    late = []
    for i in range(0, len(ds['train']['x']), max(1, len(ds['train']['x']) // 4000)):
        mask = ds['train']['mask'][i].astype(bool)
        if mask.sum() >= 10:
            late.append(ds['train']['x'][i][mask] - ds['train']['baseline'][i])
    late = np.concatenate(late, 0)
    sd = late.std(0); sd[sd == 0] = 1.0
    nets = {arm: load_model(arm, a.subset, a.seed, a.root, dev)
            for arm in ('M0', 'M0Aug', 'M1', 'M2')}
    sensors = [int(s) for s in a.sensors.split(',')]
    rows = []
    for i in range(len(va['x'])):
        row = {k: va[k][i] for k in ('x', 'mask', 'baseline', 'cycle', 'cond', 'y')}
        mask = row['mask'].astype(bool)
        if mask.sum() < 20:
            continue
        base = row['baseline']
        dy = (row['x'] - base) / sd
        y = float(row['y'])
        p0 = predict(nets['M0'], row, row['x'], dev)
        for j in sensors:
            rng = np.random.default_rng(1000 + i * 100 + j)
            dy_f, _ = inject(dy, mask, j, a.fault, a.mag, rng, onset=a.onset)
            x_f = base + dy_f * sd
            p1 = {arm: predict(nets[arm], row, x_f, dev) for arm in nets}
            rows.append((y, p0, p1['M0'], p1['M0Aug'], p1['M1'], p1['M2']))
    r = np.asarray(rows, float)
    r0 = float(np.sqrt(np.mean((r[:, 1] - r[:, 0]) ** 2)))
    r1 = float(np.sqrt(np.mean((r[:, 2] - r[:, 0]) ** 2)))
    den = r1 - r0
    out = [f'{a.subset} s{a.seed} {a.fault} mag{a.mag}: R0 {r0:.4f} R1 {r1:.4f} den {den:+.4f}']
    for k, arm in [(3, 'M0Aug'), (4, 'M1'), (5, 'M2')]:
        rmse = float(np.sqrt(np.mean((r[:, k] - r[:, 0]) ** 2)))
        rec = (r1 - rmse) / den if den > 0.05 else float('nan')
        out.append(f'{arm} {rmse:.4f} rec {rec:.3f}' if den > 0.05 else f'{arm} {rmse:.4f} rec n/a')
    print(' | '.join(out))


if __name__ == '__main__':
    main()
