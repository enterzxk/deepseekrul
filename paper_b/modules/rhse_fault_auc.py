"""Post-hoc diagnostic: does RHSE reliability detect injected sensor faults?

No training: H is PCA-initialized on training late-life dy; log_s=0 after
per-sensor standardization. For each validation window we inject a bias/drift/
stuck/noise fault into one sensor in the second half of the valid history and
measure the AUC of 1-w (faulted vs clean/pre-fault cycles).
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.modules.rhse import RHSE, pca_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--m', type=int, default=3)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--fault', default='bias', choices=['bias', 'drift', 'stuck', 'noise'])
    ap.add_argument('--mag', type=float, default=3.0)
    a = ap.parse_args()

    tr = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='train')
    va = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='val')
    N = tr['x'].shape[-1]

    # training dy at late cuts, standardized per sensor
    late = []
    for i in range(len(tr['lengths'])):
        n = int(tr['lengths'][i])
        for f in (0.7, 0.85, 0.95):
            c = int(round(f * n))
            t = c - 1
            m = tr['mask'][i, t].astype(bool)
            if m.sum() < 10:
                continue
            dy = tr['x'][i, t][m] - tr['baseline'][i, t]
            late.append(dy)
    late = np.concatenate(late, 0)
    sd = late.std(0)
    sd[sd == 0] = 1.0
    late_z = late / sd
    if len(late_z) > 200000:
        late_z = late_z[np.random.default_rng(0).choice(len(late_z), 200000, replace=False)]
    H = pca_H(late_z, a.m)
    rhse = RHSE(N, a.m, iters=a.iters, H_init=torch.tensor(H))
    rhse.eval()

    aucs = []
    for i in range(len(va['lengths'])):
        n = int(va['lengths'][i])
        c = int(round(0.95 * n))
        t = c - 1
        mask = va['mask'][i, t].astype(bool)
        if mask.sum() < 20:
            continue
        dy = (va['x'][i, t] - va['baseline'][i, t]) / sd
        first = int(np.argmax(mask))
        valid = int(mask.sum())
        t0 = first + valid // 2
        m_t = torch.tensor(mask[None, :], dtype=torch.float32)
        dy_t = torch.tensor(dy[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            _, w_clean, _ = rhse(dy_t, m_t)
            w_clean = w_clean[0].numpy()
            for j in range(N):
                dy_f = dy.copy()
                if a.fault == 'bias':
                    dy_f[t0:, j] += a.mag
                elif a.fault == 'drift':
                    ramp = np.linspace(0, a.mag, len(dy_f) - t0)
                    dy_f[t0:, j] += ramp
                elif a.fault == 'stuck':
                    dy_f[t0:, j] = dy_f[t0, j]
                else:
                    rng = np.random.default_rng(1000 + i * N + j)
                    dy_f[t0:, j] += rng.normal(0, a.mag, len(dy_f) - t0)
                _, w_f, _ = rhse(torch.tensor(dy_f[None, :, :], dtype=torch.float32), m_t)
                w_f = w_f[0].numpy()
                pos = 1.0 - w_f[t0:, j]
                neg = np.concatenate([1.0 - w_clean[:t0, j], 1.0 - w_clean[t0:, j]])
                y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
                s = np.concatenate([pos, neg])
                if len(np.unique(y)) < 2:
                    continue
                aucs.append(float(roc_auc_score(y, s)))
    aucs = np.asarray(aucs)
    print(f'{a.subset} fault={a.fault} mag={a.mag} n_auc={len(aucs)} '
          f'mean={aucs.mean():.3f} median={np.median(aucs):.3f} '
          f'frac>=0.85={(aucs >= 0.85).mean():.3f}')


if __name__ == '__main__':
    main()
