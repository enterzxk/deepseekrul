"""Training-free RUL robustness pilot: can RHSE reconstruction recover B0?

R0 clean, R1 faulted, R2 oracle reconstruction (known faulty sensor), R3 RHSE
reconstruction (low-weight sensors), R4 last-healthy-value interpolation.
One cut per engine (default 0.95); fault in one sensor; no backbone training.
"""
import argparse

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.modules.rhse import RHSE, pca_H
from paper_b.modules.stress_suite import inject, valid_span
from paper_b.modules.temporal_agg import build_model, predict_windows


def prep(subset, data_dir, seed, root, m=3, iters=3):
    tr = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='train')
    va = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='val')
    N = tr['x'].shape[-1]
    late = []
    for i in range(len(tr['lengths'])):
        n = int(tr['lengths'][i])
        for f in (0.7, 0.85, 0.95):
            t = int(round(f * n)) - 1
            mask = tr['mask'][i, t].astype(bool)
            if mask.sum() < 10:
                continue
            late.append(tr['x'][i, t][mask] - tr['baseline'][i, t])
    late = np.concatenate(late, 0)
    sd = late.std(0)
    sd[sd == 0] = 1.0
    H = pca_H(late / sd, m)
    rhse = RHSE(N, m, iters=iters, H_init=torch.tensor(H))
    rhse.eval()
    run = 'HC-D-B0-%s-s%d-D1' % (subset, seed)
    ckpt = root / run / 'ckpt' / 'best.pt'
    net = build_model(subset, seed, ckpt, torch.device('cuda:0'))
    return tr, va, sd, rhse, net


def theta_solve(rhse, dy, mask, weights=None):
    with torch.no_grad():
        s = rhse.log_s.exp()
        Hn = rhse.H / rhse.H.norm(dim=0, keepdim=True).clamp_min(1e-6)
        Hs = Hn / s[:, None]
        z = torch.tensor(dy, dtype=torch.float32) / s
        if weights is None:
            w = np.ones_like(dy)
        else:
            w = weights
        wt = torch.tensor(w[None, :, :], dtype=torch.float32)
        HtW = Hs.t()[None, None] * wt.unsqueeze(-2)
        A = HtW @ Hs + rhse.lam * torch.eye(Hs.size(1))
        b = HtW @ z[None].unsqueeze(-1)
        th = torch.linalg.solve(A, b).squeeze(-1)[0]
        return th.numpy(), Hs.numpy()


def predict_at(net, eb, i, t, x_win):
    old = eb['x'][i, t].copy()
    eb['x'][i, t] = x_win.astype(eb['x'].dtype)
    try:
        return float(predict_windows(net, eb, i, [t], torch.device('cuda:0'))[0])
    finally:
        eb['x'][i, t] = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/_gate')
    ap.add_argument('--fault', default='bias', choices=['bias', 'drift', 'noise', 'stuck'])
    ap.add_argument('--mag', type=float, default=3.0)
    ap.add_argument('--sensors', default='0,5,10')
    ap.add_argument('--cut-frac', type=float, default=0.95)
    a = ap.parse_args()

    tr, va, sd, rhse, net = prep(a.subset, a.data_dir, a.seed, __import__('pathlib').Path(a.root))
    sensors = [int(s) for s in a.sensors.split(',')]
    rows = []
    for i in range(len(va['lengths'])):
        for cut in va['cut_t'][i]:
            if cut <= 0:
                continue
            t = int(cut) - 1
            mask = va['mask'][i, t].astype(bool)
            if mask.sum() < 20:
                continue
            base = va['baseline'][i, t]
            x = va['x'][i, t]
            dy = (x - base) / sd
            y = float(va['y'][i, t])
            p0 = predict_at(net, va, i, t, x)
            for j in sensors:
                rng = np.random.default_rng(1000 + i * 100 + j)
                dy_f, pos = inject(dy, mask, j, a.fault, a.mag, rng)
                x_f = base + dy_f * sd
                p1 = predict_at(net, va, i, t, x_f)
                # oracle: exclude known faulty sensor
                w_or = np.ones_like(dy_f)
                w_or[:, j] = 0.0
                th_or, Hs = theta_solve(rhse, dy_f, mask, w_or)
                dy_hat = th_or @ Hs.T
                x_or = x_f.copy()
                x_or[:, j] = base[j] + dy_hat[:, j] * sd[j]
                p2 = predict_at(net, va, i, t, x_or)
                # RHSE: reconstruct sensors with low final weight
                with torch.no_grad():
                    _, w, _ = rhse(torch.tensor(dy_f[None], dtype=torch.float32),
                                   torch.tensor(mask[None], dtype=torch.float32))
                w = w[0].numpy()
                bad = w[-1] < 0.5
                x_rh = x_f.copy()
                if bad.any():
                    th_rh, Hs = theta_solve(rhse, dy_f, mask, w)
                    dy_hat_rh = th_rh @ Hs.T
                    x_rh[:, bad] = base[bad] + dy_hat_rh[:, bad] * sd[bad]
                p3 = predict_at(net, va, i, t, x_rh)
                # last healthy value interpolation
                first, last = valid_span(mask)
                t0 = first + max(1, (last - first) // 2)
                x_lv = x_f.copy()
                x_lv[t0:last, j] = x[max(0, t0 - 1), j]
                p4 = predict_at(net, va, i, t, x_lv)
                rows.append((i, j, y, p0, p1, p2, p3, p4))
    r = np.asarray(rows, dtype=float)
    for k, name in [(3, 'R0'), (4, 'R1'), (5, 'R2'), (6, 'R3'), (7, 'R4')]:
        rmse = float(np.sqrt(np.mean((r[:, k] - r[:, 2]) ** 2)))
        print(f'{name} RMSE {rmse:.4f}')
    base = float(np.sqrt(np.mean((r[:, 4] - r[:, 2]) ** 2)))
    clean = float(np.sqrt(np.mean((r[:, 3] - r[:, 2]) ** 2)))
    for k, name in [(5, 'R2'), (6, 'R3'), (7, 'R4')]:
        rmse = float(np.sqrt(np.mean((r[:, k] - r[:, 2]) ** 2)))
        rec = (base - rmse) / max(base - clean, 1e-9)
        print(f'{name} recovery {rec:.3f}')


if __name__ == '__main__':
    main()
