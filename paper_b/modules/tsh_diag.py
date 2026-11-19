"""Diagnose which TSH statistic drives clean EOL false alarms."""
import argparse
from pathlib import Path

import numpy as np
import torch

from paper_b.modules.rhse_rul_eval import prep
from paper_b.modules.rhse_tsh import calibrate, stats
from paper_b.modules.stress_suite import valid_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/_gate')
    ap.add_argument('--gamma', type=float, default=1.30)
    a = ap.parse_args()
    tr, va, sd, rhse, net = prep(a.subset, '/home/user/rulb/data/CMAPSS', 101, Path(a.root))
    q = calibrate(tr, sd, rhse)
    print('q1', np.round(q['q1'], 3))
    print('q2', np.round(q['q2'], 3))
    print('q3h', np.round(q['q3h'], 3))
    print('q3l', np.round(q['q3l'], 5))
    print('quantized', q.get('quantized'))
    counts = np.zeros(4)
    total = 0
    for i in range(len(va['lengths'])):
        for cut in va['cut_t'][i]:
            if cut <= 0:
                continue
            t = int(cut) - 1
            mask = va['mask'][i, t].astype(bool)
            if mask.sum() < 20:
                continue
            dy = (va['x'][i, t] - va['baseline'][i, t]) / sd
            with torch.no_grad():
                _, _, r = rhse(torch.tensor(dy[None], dtype=torch.float32),
                               torch.tensor(mask[None], dtype=torch.float32))
            S1, S2, S3h, S3l = stats(r[0].numpy(), dy, mask)
            first, last = valid_span(mask)
            e0 = first + int(0.8 * (last - first + 1))
            eol = mask & (np.arange(len(mask)) >= e0)
            u = np.stack([S1 / q['q1'], S2 / q['q2'],
                          S3h / q['q3h'],
                          np.where(S3l <= q['q3l'][None, :], 2.0, 0.0)], -1)
            e = u[eol]
            total += e.shape[0] * e.shape[1]
            for k in range(4):
                counts[k] += int((e[..., k] > a.gamma).sum())
    print('total sensor-cycles', total, 'per-stat flag rate', counts / total)


if __name__ == '__main__':
    main()
