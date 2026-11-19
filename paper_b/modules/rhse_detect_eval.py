"""Corrected RHSE detection evaluation.

- amplitude-stratified AUC (drift only counts cycles with injected magnitude >= 1 sigma)
- clean end-of-life false-alarm rate at a 1% clean-training threshold
- per-sensor AUC, averaged over validation engines
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.modules.rhse import RHSE, pca_H
from paper_b.modules.stress_suite import inject


def prep(subset, data_dir, m=3, iters=3):
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
    late_z = late / sd
    if len(late_z) > 200000:
        late_z = late_z[np.random.default_rng(0).choice(len(late_z), 200000, replace=False)]
    H = pca_H(late_z, m)
    rhse = RHSE(N, m, iters=iters, H_init=torch.tensor(H))
    rhse.eval()
    return tr, va, sd, rhse


def score_window(rhse, dy, mask):
    m = torch.tensor(mask[None, :], dtype=torch.float32)
    d = torch.tensor(dy[None, :, :], dtype=torch.float32)
    with torch.no_grad():
        _, w, _ = rhse(d, m)
    return (1.0 - w[0].numpy()), w[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--m', type=int, default=3)
    ap.add_argument('--iters', type=int, default=3)
    a = ap.parse_args()

    tr, va, sd, rhse = prep(a.subset, a.data_dir, a.m, a.iters)
    N = len(sd)

    # clean-training q99 per sensor (includes late life)
    train_scores = []
    for i in range(len(tr['lengths'])):
        n = int(tr['lengths'][i])
        for f in (0.7, 0.85, 0.95):
            t = int(round(f * n)) - 1
            mask = tr['mask'][i, t].astype(bool)
            if mask.sum() < 10:
                continue
            dy = (tr['x'][i, t] - tr['baseline'][i, t]) / sd
            s, _ = score_window(rhse, dy, mask)
            train_scores.append(s[mask])
    train_scores = np.concatenate(train_scores, 0)
    q99 = np.percentile(train_scores, 99, axis=0)  # (N,)

    clean_scores = []
    eol_fpr = []
    for i in range(len(va['lengths'])):
        n = int(va['lengths'][i])
        t = int(round(0.95 * n)) - 1
        mask = va['mask'][i, t].astype(bool)
        if mask.sum() < 20:
            continue
        dy = (va['x'][i, t] - va['baseline'][i, t]) / sd
        s, _ = score_window(rhse, dy, mask)
        clean_scores.append((s, mask))
        first, last = np.where(mask)[0][[0, -1]]
        e0 = first + int(0.8 * (last - first + 1))
        eol = np.zeros(len(mask), dtype=bool)
        eol[e0:last + 1] = True
        eol_fpr.append(float((s[eol] > q99[None, :]).mean()))

    print(f'{a.subset} clean_EOL_FPR={np.mean(eol_fpr):.4f} '
          f'(median {np.median(eol_fpr):.4f}, n={len(eol_fpr)})')
    for fault in ['bias', 'noise', 'drift', 'stuck']:
        for mag in [1.0, 2.0, 3.0, 5.0]:
            aucs = []
            for i, (s_clean, mask) in enumerate(clean_scores):
                n = int(va['lengths'][i])
                t = int(round(0.95 * n)) - 1
                dy = (va['x'][i, t] - va['baseline'][i, t]) / sd
                for j in range(N):
                    rng = np.random.default_rng(1000 + i * N + j)
                    dy_f, pos = inject(dy, mask, j, fault, mag, rng)
                    s_f, _ = score_window(rhse, dy_f, mask)
                    if pos.sum() == 0:
                        continue
                    y = np.concatenate([np.ones(pos.sum()), np.zeros(mask.sum())])
                    sc = np.concatenate([s_f[pos, j], s_clean[mask, j]])
                    if len(np.unique(y)) < 2:
                        continue
                    aucs.append(float(roc_auc_score(y, sc)))
            aucs = np.asarray(aucs)
            print(f'  {fault:5s} mag={mag:.0f} n={len(aucs)} '
                  f'AUC={aucs.mean():.3f} frac>=0.90={(aucs >= 0.90).mean():.3f}')


if __name__ == '__main__':
    main()
