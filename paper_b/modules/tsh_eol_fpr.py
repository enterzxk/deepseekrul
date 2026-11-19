"""Clean end-of-life false-alarm rate for RHSE-TSH.

The gate is: on clean validation engines, the last 20% of valid cycles should
not be flagged at more than 2% (per window, any sensor) at the chosen gamma.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from paper_b.modules.rhse_rul_eval import prep
from paper_b.modules.rhse_tsh import calibrate, detect
from paper_b.modules.stress_suite import valid_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/_gate')
    ap.add_argument('--gamma', type=float, default=1.30)
    a = ap.parse_args()
    tr, va, sd, rhse, net = prep(a.subset, a.data_dir, a.seed, Path(a.root))
    q = calibrate(tr, sd, rhse)
    fprs, rates = [], []
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
                _, w, r = rhse(torch.tensor(dy[None], dtype=torch.float32),
                               torch.tensor(mask[None], dtype=torch.float32))
            flag, _ = detect(r[0].numpy(), dy, mask, q, gamma=a.gamma)
            first, last = valid_span(mask)
            e0 = first + int(0.8 * (last - first + 1))
            eol = mask & (np.arange(len(mask)) >= e0)
            if eol.sum() == 0:
                continue
            fprs.append(float(flag[eol].any(1).mean()))
            rates.append(float(flag[eol].mean()))
    print(f'{a.subset} gamma={a.gamma} n={len(fprs)} '
          f'EOL_any_FPR={np.mean(fprs):.4f} (median {np.median(fprs):.4f}) '
          f'EOL_cycle_flag_rate={np.mean(rates):.4f}')


if __name__ == '__main__':
    main()
