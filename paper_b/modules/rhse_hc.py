"""Multi-condition H_c diagnostic: does a per-condition measurement matrix
reconstruct the faulty sensor better than a single global H?
"""
import argparse

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.modules.rhse import RHSE, pca_H
from paper_b.modules.rhse_rul_eval import theta_solve


def build_hc(tr, sd, m=3):
    N = tr['x'].shape[-1]
    n_cond = int(tr['cond'].max()) + 1
    late = []
    for i in range(len(tr['lengths'])):
        n = int(tr['lengths'][i])
        for f in (0.7, 0.85, 0.95):
            t = int(round(f * n)) - 1
            mask = tr['mask'][i, t].astype(bool)
            if mask.sum() >= 10:
                late.append(tr['x'][i, t][mask] - tr['baseline'][i, t])
    late = np.concatenate(late, 0)
    H0 = pca_H(late / sd, m)
    out = {}
    for c in range(n_cond):
        vals = []
        for i in range(len(tr['lengths'])):
            n = int(tr['lengths'][i])
            for f in (0.7, 0.85, 0.95):
                t = int(round(f * n)) - 1
                mask = tr['mask'][i, t].astype(bool)
                if mask.sum() < 10 or int(tr['cond'][i, t]) != c:
                    continue
                vals.append(tr['x'][i, t][mask] - tr['baseline'][i, t])
        if vals:
            Hc = pca_H(np.concatenate(vals, 0) / sd, m)
        else:
            Hc = H0
        out[c] = RHSE(N, m, H_init=torch.tensor(Hc))
        print(f'cond {c}: n={sum(len(v) for v in vals)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD002')
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--sensors', default='0,5,10')
    ap.add_argument('--mag', type=float, default=3.0)
    a = ap.parse_args()
    tr = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='train')
    va = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='val')
    late = []
    for i in range(len(tr['lengths'])):
        n = int(tr['lengths'][i])
        for f in (0.7, 0.85, 0.95):
            t = int(round(f * n)) - 1
            mask = tr['mask'][i, t].astype(bool)
            if mask.sum() >= 10:
                late.append(tr['x'][i, t][mask] - tr['baseline'][i, t])
    late = np.concatenate(late, 0)
    sd = late.std(0)
    sd[sd == 0] = 1.0
    H0 = pca_H(late / sd, 3)
    rhse_g = RHSE(tr['x'].shape[-1], 3, H_init=torch.tensor(H0))
    hc = build_hc(tr, sd)
    sensors = [int(s) for s in a.sensors.split(',')]
    mg, mc = [], []
    for i in range(len(va['lengths'])):
        n = int(va['lengths'][i])
        t = int(round(0.95 * n)) - 1
        mask = va['mask'][i, t].astype(bool)
        if mask.sum() < 20:
            continue
        c = int(va['cond'][i, t])
        base = va['baseline'][i, t]
        dy = (va['x'][i, t] - base) / sd
        for j in sensors:
            dyf = dy.copy()
            dyf[mask, j] += a.mag
            w = np.ones_like(dyf)
            w[:, j] = 0.0
            for name, rhse, out in [('global', rhse_g, mg), ('hc', hc[c], mc)]:
                th, Hs = theta_solve(rhse, dyf, mask, w)
                dh = th @ Hs.T
                out.append(float(np.mean((dh[mask, j] - dy[mask, j]) ** 2)))
    print(f'{a.subset} global_mse={np.mean(mg):.4f} hc_mse={np.mean(mc):.4f} '
          f'ratio={np.mean(mc) / max(np.mean(mg), 1e-9):.3f}')


if __name__ == '__main__':
    main()
