"""Phase 4: fair naive baselines R5b (backtracked hold) and R5m (backtracked median).

All use observed x_f only; no clean counterfactual. R3 is TSH+H-theta.
FD001/FD003 x {bias,drift,stuck} x mag3 x onset {0.2,0.5} x seeds {101,202,303}.
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
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/_gate')
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--fault', default='bias', choices=['bias', 'drift', 'stuck'])
    ap.add_argument('--mag', type=float, default=3.0)
    ap.add_argument('--onset', type=float, default=0.5)
    ap.add_argument('--sensors', default='0,5,10')
    ap.add_argument('--gamma', type=float, default=1.30)
    a = ap.parse_args()
    tr, va, sd, rhse, net = prep(a.subset, a.data_dir, a.seed, Path(a.root))
    q = calibrate(tr, sd, rhse)
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
                dy_f, _ = inject(dy, mask, j, a.fault, a.mag, rng, onset=a.onset)
                x_f = base + dy_f * sd
                p1 = predict_at(net, va, i, t, x_f)
                with torch.no_grad():
                    _, w, r = rhse(torch.tensor(dy_f[None], dtype=torch.float32),
                                   torch.tensor(mask[None], dtype=torch.float32))
                flag, _ = detect(r[0].numpy(), dy_f, mask, q, gamma=a.gamma)
                x3 = x_f.copy()
                if flag.any():
                    w2 = w[0].numpy().copy(); w2[flag] = 0.0
                    th, Hs = theta_solve(rhse, dy_f, mask, w2)
                    dh = th @ Hs.T
                    xr = base + dh * sd
                    x3[flag] = xr[flag]
                p3 = predict_at(net, va, i, t, x3)
                first, last = valid_span(mask)
                x5 = x_f.copy(); x5b = x_f.copy(); x5m = x_f.copy()
                for jj in np.where(flag.any(0))[0]:
                    fb = int(np.argmax(flag[:, jj]))
                    x5[fb:last, jj] = x_f[max(0, fb - 1), jj]
                    ib = max(0, fb - 15 - 1)
                    x5b[fb:last, jj] = x_f[ib, jj]
                    lo = max(0, fb - 15 - 10)
                    hi = max(1, fb - 15)
                    x5m[fb:last, jj] = np.median(x_f[lo:hi, jj])
                p5 = predict_at(net, va, i, t, x5)
                p5b = predict_at(net, va, i, t, x5b)
                p5m = predict_at(net, va, i, t, x5m)
                rows.append((y, p0, p1, p3, p5, p5b, p5m))
    r = np.asarray(rows, float)
    r0 = float(np.sqrt(np.mean((r[:, 1] - r[:, 0]) ** 2)))
    r1 = float(np.sqrt(np.mean((r[:, 2] - r[:, 0]) ** 2)))
    den = r1 - r0
    print(f'{a.subset} s{a.seed} {a.fault} onset{a.onset}: R0 {r0:.4f} R1 {r1:.4f} den {den:+.4f}')
    for k, name in [(3, 'R3'), (4, 'R5'), (5, 'R5b'), (6, 'R5m')]:
        rmse = float(np.sqrt(np.mean((r[:, k] - r[:, 0]) ** 2)))
        rec = (r1 - rmse) / den if den > 0.05 else float('nan')
        print(f'{name} {rmse:.4f} rec {rec:.3f}' if den > 0.05 else f'{name} {rmse:.4f} rec n/a')


if __name__ == '__main__':
    main()
