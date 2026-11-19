"""Training-free RUL pilot with RHSE-TSH detection + reconstruction.

R0 clean, R1 faulted, R2 oracle H-theta, R3 RHSE-TSH H-theta, R4 oracle
last-value, R5 TSH-detected last-value.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from paper_b.modules.rhse_rul_eval import predict_at, prep, theta_solve
from paper_b.modules.rhse_tsh import calibrate, detect
from paper_b.modules.stress_suite import inject, valid_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/_gate')
    ap.add_argument('--fault', default='drift', choices=['bias', 'drift', 'noise', 'stuck'])
    ap.add_argument('--mag', type=float, default=3.0)
    ap.add_argument('--sensors', default='0,5,10')
    ap.add_argument('--gamma', type=float, default=1.0)
    a = ap.parse_args()

    tr, va, sd, rhse, net = prep(a.subset, a.data_dir, a.seed, Path(a.root))
    q = calibrate(tr, sd, rhse)
    sensors = [int(s) for s in a.sensors.split(',')]
    rows = []
    clean_rows = []
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
            # clean TSH cost
            with torch.no_grad():
                _, wc, rc = rhse(torch.tensor(dy[None], dtype=torch.float32),
                                 torch.tensor(mask[None], dtype=torch.float32))
            flagc, _ = detect(rc[0].numpy(), dy, mask, q, gamma=a.gamma)
            xc = x.copy()
            if flagc.any():
                w2 = wc[0].numpy().copy()
                w2[flagc] = 0.0
                th, Hs = theta_solve(rhse, dy, mask, w2)
                dh = th @ Hs.T
                xr = base + dh * sd
                xc[flagc] = xr[flagc]
            p3c = predict_at(net, va, i, t, xc)
            clean_rows.append((y, p0, p3c))
            for j in sensors:
                rng = np.random.default_rng(1000 + i * 100 + j)
                dy_f, _ = inject(dy, mask, j, a.fault, a.mag, rng)
                x_f = base + dy_f * sd
                p1 = predict_at(net, va, i, t, x_f)
                w_or = np.ones_like(dy_f)
                w_or[:, j] = 0.0
                th_or, Hs = theta_solve(rhse, dy_f, mask, w_or)
                dh_or = th_or @ Hs.T
                x_or = x_f.copy()
                x_or[:, j] = base[j] + dh_or[:, j] * sd[j]
                p2 = predict_at(net, va, i, t, x_or)
                with torch.no_grad():
                    _, w, r = rhse(torch.tensor(dy_f[None], dtype=torch.float32),
                                   torch.tensor(mask[None], dtype=torch.float32))
                w = w[0].numpy()
                flag, _ = detect(r[0].numpy(), dy_f, mask, q, gamma=a.gamma)
                x_tsh = x_f.copy()
                if flag.any():
                    w2 = w.copy()
                    w2[flag] = 0.0
                    th, Hs = theta_solve(rhse, dy_f, mask, w2)
                    dh = th @ Hs.T
                    xr = base + dh * sd
                    x_tsh[flag] = xr[flag]
                p3 = predict_at(net, va, i, t, x_tsh)
                first, last = valid_span(mask)
                t0 = first + max(1, (last - first) // 2)
                x_lv = x_f.copy()
                x_lv[t0:last, j] = x[max(0, t0 - 1), j]
                p4 = predict_at(net, va, i, t, x_lv)
                x_lvd = x_f.copy()
                for jj in np.where(flag.any(0))[0]:
                    fb = int(np.argmax(flag[:, jj]))
                    x_lvd[fb:last, jj] = x[max(0, fb - 1), jj]
                p5 = predict_at(net, va, i, t, x_lvd)
                rows.append((i, j, y, p0, p1, p2, p3, p4, p5))

    c = np.asarray(clean_rows, float)
    r0c = float(np.sqrt(np.mean((c[:, 1] - c[:, 0]) ** 2)))
    r3c = float(np.sqrt(np.mean((c[:, 2] - c[:, 0]) ** 2)))
    print(f'clean R0 {r0c:.4f} R3c {r3c:.4f} cost {r3c - r0c:+.4f}')
    r = np.asarray(rows, float)
    for k, name in [(3, 'R0'), (4, 'R1'), (5, 'R2'), (6, 'R3'),
                    (7, 'R4'), (8, 'R5')]:
        print(f'{name} RMSE {float(np.sqrt(np.mean((r[:, k] - r[:, 2]) ** 2))):.4f}')
    base = float(np.sqrt(np.mean((r[:, 4] - r[:, 2]) ** 2)))
    clean = float(np.sqrt(np.mean((r[:, 3] - r[:, 2]) ** 2)))
    for k, name in [(5, 'R2'), (6, 'R3'), (7, 'R4'), (8, 'R5')]:
        rmse = float(np.sqrt(np.mean((r[:, k] - r[:, 2]) ** 2)))
        print(f'{name} recovery {(base - rmse) / max(base - clean, 1e-9):.3f}')


if __name__ == '__main__':
    main()
