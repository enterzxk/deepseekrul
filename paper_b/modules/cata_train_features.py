"""Per-cut features on TRAINING engines for the learned CATA gate.

The frozen B0 checkpoint is applied to the training engines at the same
life-fraction cuts used for validation. The predictions are in-sample with
respect to the backbone (the backbone saw these engines), so the gate is a
calibration on top of the model, not an independent predictor. Cross-fitting by
engine is used in ``learned_gate.py``; validation engines are never touched here.

Output columns: engine, cut, n, y_cap, y_raw, p_k1, p_med, p_mean, p_std,
stat, shift, regime_dist, cap, cycle_frac.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.modules.cap_align import align_cap_aware
from paper_b.modules.temporal_agg import build_model, predict_windows
from paper_b.training.data import _prepare_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--root', default=str(Path.home() / 'sprint24/experiments/hc/_gate'))
    ap.add_argument('--data-dir', default=str(Path.home() / 'rulb/data/CMAPSS'))
    ap.add_argument('--out-dir', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/train'))
    ap.add_argument('--delta', type=int, default=5)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--cap-align', action='store_true',
                    help='cap-aware age alignment (soft plateau gate)')
    ap.add_argument('--cap-tau', type=float, default=3.0)
    ap.add_argument('--cap-margin', type=float, default=3.0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    run = 'HC-D-B0-%s-s%d-D1' % (a.subset, a.seed)
    ckpt = Path(a.root) / run / 'ckpt' / 'best.pt'
    if not ckpt.exists():
        raise SystemExit('missing checkpoint %s' % ckpt)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    net = build_model(a.subset, a.seed, ckpt, dev)
    eb = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='train')

    s = _prepare_split(a.subset, a.data_dir, C.SPLIT_SEED, C.FOLD, C.VAL_FRAC, C.N_COND)
    op_cols = ['op_1', 'op_2', 'op_3']
    tr_op = s['tr'][op_cols].to_numpy(float)
    mu, sd = tr_op.mean(0), tr_op.std(0)
    sd[sd == 0] = 1.0
    tr_opz = (tr_op - mu) / sd
    op_map = {(int(u), int(c)): tr_opz[k]
              for k, (u, c) in enumerate(zip(s['tr'].unit, s['tr'].cycle))}

    rows = []
    for i in range(len(eb['lengths'])):
        n = int(eb['lengths'][i])
        uid = int(eb['engine_id'][i])
        seen = set()
        for f in C.VAL_CUTS:
            c = int(min(n, max(1, round(f * n))))
            if c in seen:
                continue
            seen.add(c)
            t = c - 1
            idxs = [t - j * a.delta for j in range(a.K) if t - j * a.delta >= 0]
            preds = predict_windows(net, eb, i, idxs, dev)
            sel_idx = np.asarray(idxs, np.int64)
            age = c - (sel_idx + 1)
            aligned = (align_cap_aware(preds, age, tau=a.cap_tau,
                                       margin=a.cap_margin)
                       if a.cap_align else preds - age)
            med = float(np.median(aligned))
            mad = float(np.median(np.abs(aligned - med)))
            ops = np.stack([op_map[(uid, int(x) + 1)] for x in sel_idx])
            dmat = np.linalg.norm(ops[:, None, :] - ops[None, :, :], axis=-1)
            rows.append(dict(
                engine=uid, cut=c, n=n,
                y_cap=float(min(n - c, 125)), y_raw=float(n - c),
                p_k1=float(preds[0]), p_med=med,
                p_mean=float(aligned.mean()), p_std=float(aligned.std(ddof=0)),
                stat=mad / max(abs(med), 1.0), shift=abs(med - float(preds[0])),
                regime_dist=float(dmat.max()),
                cap=float(float(preds[0]) >= 124.5 or med >= 124.5),
                cycle_frac=float(c / n)))
    out = Path(a.out_dir) / ('%s_s%d.csv' % (a.subset, a.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote %s rows=%d engines=%d' % (out, len(rows), len(eb['lengths'])))


if __name__ == '__main__':
    main()
