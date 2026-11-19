"""Phase 2 M_front: frozen DSPT-V0 + static RHSE-TSH front-end (no retraining).

Uses the frozen ``prepare_train`` validation windows (V0-compatible), not the
Stage-D ``engine_batch``. Validation only; test sealed.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.models import build_dspt
from paper_b.modules.rhse import RHSE, pca_H
from paper_b.modules.rhse_rul_eval import theta_solve
from paper_b.modules.rhse_tsh import stats
from paper_b.modules.stress_suite import inject, valid_span
from paper_b.training.data import prepare_train


def load_v0(subset, seed, root, dev):
    net = build_dspt(num_sensors=14, d_model=64, num_layers=2, dropout=0.2,
                     max_span=70, ms_mode='full', sa_mode='basic',
                     hi_source='hf', hi_mono='none', use_pad_mask=True,
                     use_cycle=True, n_cond=6)
    ckpt = Path(root) / f'phase2_M0_{subset}_s{seed}' / 'ckpt' / 'best.pt'
    net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    return net.to(dev).eval()


def calibrate_windows(tr, sd, rhse, n=6000):
    vals = [[], [], [], []]
    idx = np.random.default_rng(0).choice(len(tr['x']), min(n, len(tr['x'])), replace=False)
    for i in idx:
        mask = tr['mask'][i].astype(bool)
        if mask.sum() < 15:
            continue
        dy = (tr['x'][i] - tr['baseline'][i]) / sd
        with torch.no_grad():
            _, _, r = rhse(torch.tensor(dy[None], dtype=torch.float32),
                           torch.tensor(mask[None], dtype=torch.float32))
        S1, S2, S3h, S3l = stats(r[0].numpy(), dy, mask)
        vals[0].append(S1[mask]); vals[1].append(S2[mask])
        vals[2].append(S3h[mask])
        good = mask & ~np.isnan(S3l).any(1)
        if good.any():
            vals[3].append(S3l[good])
    q1 = np.maximum(np.percentile(np.concatenate(vals[0]), 99.75, axis=0), 1e-6)
    q2 = np.maximum(np.percentile(np.concatenate(vals[1]), 99.75, axis=0), 1e-6)
    q3h = np.maximum(np.percentile(np.concatenate(vals[2]), 99.75, axis=0), 1e-6)
    q3l_raw = np.percentile(np.concatenate(vals[3]), 0.25, axis=0)
    return dict(q1=q1, q2=q2, q3h=q3h, q3l=np.maximum(q3l_raw, 1e-7),
                quantized=q3l_raw <= 1e-6)


def detect(r, dy, mask, q, gamma):
    S1, S2, S3h, S3l = stats(r, dy, mask)
    stuck = np.zeros_like(S3l, dtype=bool)
    run = np.zeros(S3l.shape[1], np.int32)
    for t in range(len(mask)):
        if not mask[t]:
            continue
        run = np.where(S3l[t] <= q['q3l'][None, :], run + 1, 0)
        stuck[t] = run >= 15
    u = np.nanmax(np.stack([S1 / q['q1'], S2 / q['q2'], S3h / q['q3h'],
                            np.where(stuck, 2.0, 0.0)], 0), axis=0)
    raw = (u > gamma) & mask[:, None]
    flag = np.zeros_like(raw)
    last = []
    for t in range(len(mask)):
        if not mask[t]:
            continue
        last.append(raw[t]); last = last[-3:]
        confirm = len(last) >= 2 and np.sum(np.stack(last, 0), axis=0) >= 2
        prev = flag[t - 1] if t > 0 else np.zeros(raw.shape[1], bool)
        flag[t] = (confirm | prev) & mask[t]
    return flag


def prep(subset, seed, root, data_dir):
    ds = prepare_train(subset, data_dir, span=C.SPAN, split_seed=7, fold=0,
                       val_frac=0.2, rul_cap=125.0, n_cond=6,
                       cuts=tuple(C.VAL_CUTS))
    tr, va = ds['train'], ds['val']
    late = []
    for i in range(0, len(tr['x']), max(1, len(tr['x']) // 4000)):
        mask = tr['mask'][i].astype(bool)
        if mask.sum() >= 10:
            late.append(tr['x'][i][mask] - tr['baseline'][i])
    late = np.concatenate(late, 0)
    sd = late.std(0); sd[sd == 0] = 1.0
    H = pca_H(late / sd, 3)
    rhse = RHSE(14, 3, H_init=torch.tensor(H))
    q = calibrate_windows(tr, sd, rhse)
    dev = torch.device('cuda:0')
    net = load_v0(subset, seed, root, dev)
    return tr, va, sd, rhse, q, net, dev


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
    ap.add_argument('--root', default='/home/user/sprint24/experiments/phase2')
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--fault', default='bias', choices=['bias', 'drift', 'noise', 'stuck'])
    ap.add_argument('--mag', type=float, default=3.0)
    ap.add_argument('--sensors', default='0,5,10')
    ap.add_argument('--gamma', type=float, default=1.30)
    ap.add_argument('--onset', type=float, default=0.5)
    a = ap.parse_args()
    tr, va, sd, rhse, q, net, dev = prep(a.subset, a.seed, a.root, a.data_dir)
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
        p0 = predict(net, row, row['x'], dev)
        for j in sensors:
            rng = np.random.default_rng(1000 + i * 100 + j)
            dy_f, _ = inject(dy, mask, j, a.fault, a.mag, rng, onset=a.onset)
            x_f = base + dy_f * sd
            p1 = predict(net, row, x_f, dev)
            with torch.no_grad():
                _, w, r = rhse(torch.tensor(dy_f[None], dtype=torch.float32),
                               torch.tensor(mask[None], dtype=torch.float32))
            flag = detect(r[0].numpy(), dy_f, mask, q, a.gamma)
            x_tsh = x_f.copy()
            if flag.any():
                w2 = w[0].numpy().copy(); w2[flag] = 0.0
                th, Hs = theta_solve(rhse, dy_f, mask, w2)
                dh = th @ Hs.T
                xr = base + dh * sd
                x_tsh[flag] = xr[flag]
            p3 = predict(net, row, x_tsh, dev)
            first, last = valid_span(mask)
            x_lv = x_f.copy()
            for jj in np.where(flag.any(0))[0]:
                fb = int(np.argmax(flag[:, jj]))
                x_lv[fb:last, jj] = x_f[max(0, fb - 1), jj]
            p5 = predict(net, row, x_lv, dev)
            rows.append((y, p0, p1, p3, p5))
    r = np.asarray(rows, float)
    r0 = float(np.sqrt(np.mean((r[:, 1] - r[:, 0]) ** 2)))
    r1 = float(np.sqrt(np.mean((r[:, 2] - r[:, 0]) ** 2)))
    den = r1 - r0
    print(f'R0 {r0:.4f} R1 {r1:.4f} den {den:+.4f}')
    for k, name in [(3, 'R3'), (4, 'R5')]:
        rmse = float(np.sqrt(np.mean((r[:, k] - r[:, 0]) ** 2)))
        if den <= 0.05:
            print(f'{name} {rmse:.4f} recovery n/a')
        else:
            print(f'{name} {rmse:.4f} recovery {(r1 - rmse) / den:.3f}')


if __name__ == '__main__':
    main()
