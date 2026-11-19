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
    ap.add_argument('--cut', type=float, default=0.95)
    ap.add_argument('--sensors', default='0,5,10')
    a = ap.parse_args()
    tr, va, sd, rhse, net = prep(a.subset, a.data_dir, a.seed, Path(a.root))
    q = calibrate(tr, sd, rhse)
    sensors = [int(s) for s in a.sensors.split(',')]
    per_cycle, any_cycle, any_win = [], [], []
    for i in range(len(va['lengths'])):
        n = int(va['lengths'][i])
        t = int(round(a.cut * n)) - 1
        mask = va['mask'][i, t].astype(bool)
        if mask.sum() < 20:
            continue
        dy = (va['x'][i, t] - va['baseline'][i, t]) / sd
        with torch.no_grad():
            _, w, r = rhse(torch.tensor(dy[None], dtype=torch.float32),
                           torch.tensor(mask[None], dtype=torch.float32))
        flag, _ = detect(r[0].numpy(), dy, mask, q, gamma=a.gamma)
        flag = flag[:, sensors]
        first, last = valid_span(mask)
        e0 = first + int(0.8 * (last - first + 1))
        eol = mask & (np.arange(len(mask)) >= e0)
        if eol.sum() == 0:
            continue
        per_cycle.append(float(flag[eol].mean()))
        any_cycle.append(float(flag[eol].any(1).mean()))
        any_win.append(float(flag[eol].any()))
    print(f'{a.subset} gamma={a.gamma} cut={a.cut} n={len(per_cycle)} '
          f'EOL_cycle_flag_rate={np.mean(per_cycle):.4f} '
          f'EOL_any_sensor_cycle={np.mean(any_cycle):.4f} '
          f'EOL_any_window={np.mean(any_win):.4f}')


if __name__ == '__main__':
    main()
