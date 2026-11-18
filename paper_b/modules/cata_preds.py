"""Per-cut predictions for the CATA primary config (K=3, delta=5, tau=0.05).

Writes one CSV per (subset, seed) with K=1, all-window median, and gated median.
The gated decision uses only the aligned predictions; labels are stored for
evaluation only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.modules.temporal_agg import build_model, predict_windows
from paper_b.training.data import _prepare_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--root', default=str(Path.home() / 'sprint24/experiments/hc/_gate'))
    ap.add_argument('--data-dir', default=str(Path.home() / 'rulb/data/CMAPSS'))
    ap.add_argument('--out-dir', default=str(Path.home() / 'sprint24/experiments/hc/temporal_agg/preds'))
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--delta', type=int, default=5)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    run = 'HC-D-B0-%s-s%d-D1' % (a.subset, a.seed)
    ckpt = Path(a.root) / run / 'ckpt' / 'best.pt'
    if not ckpt.exists():
        raise SystemExit('missing checkpoint %s' % ckpt)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    net = build_model(a.subset, a.seed, ckpt, dev)
    eb = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='val')
    # Raw operating-condition vectors for the regime-stability gate. These are
    # measured inputs, not labels; standardisation uses training engines only.
    s = _prepare_split(a.subset, a.data_dir, C.SPLIT_SEED, C.FOLD, C.VAL_FRAC, C.N_COND)
    op_cols = ['op_1', 'op_2', 'op_3']
    tr_op = s['tr'][op_cols].to_numpy(float)
    mu = tr_op.mean(0)
    sd = tr_op.std(0)
    sd[sd == 0] = 1.0
    va_op = s['va'][op_cols].to_numpy(float)
    opz = (va_op - mu) / sd
    op_map = {(int(u), int(c)): opz[k]
              for k, (u, c) in enumerate(zip(s['va'].unit, s['va'].cycle))}

    rows = []
    for i in range(len(eb['lengths'])):
        n = int(eb['lengths'][i])
        uid = int(eb['engine_id'][i])
        cuts = [int(c) for c in eb['cut_t'][i] if int(c) > 0]
        for c in cuts:
            idxs = [c - 1 - j * a.delta for j in range(a.K)]
            idxs = [t for t in idxs if t >= 0]
            preds = predict_windows(net, eb, i, idxs, dev)
            sel_idx = np.asarray(idxs, np.int64)
            aligned = preds - (c - (sel_idx + 1))
            med = float(np.median(aligned))
            mad = float(np.median(np.abs(aligned - med)))
            stat = mad / max(abs(med), 1.0)
            gated = int(stat <= a.tau)
            pred_gate = med if gated else float(preds[0])
            ops = np.stack([op_map[(uid, int(t) + 1)] for t in sel_idx])
            dmat = np.linalg.norm(ops[:, None, :] - ops[None, :, :], axis=-1)
            regime_dist = float(dmat.max())
            rows.append(dict(engine=uid, cut=c, y_cap=float(min(n - c, 125)),
                             y_raw=float(n - c), pred_k1=float(preds[0]),
                             pred_all=med, pred_gate=pred_gate,
                             p_mean=float(aligned.mean()),
                             p_std=float(aligned.std(ddof=0)),
                             stat=stat, gated=gated, K=len(idxs),
                             regime_dist=regime_dist,
                             shift=abs(med - float(preds[0])),
                             cap=float(float(preds[0]) >= 124.5 or med >= 124.5),
                             cycle_frac=float(c / n), n=n,
                             delta=a.delta, tau=a.tau))
    out = Path(a.out_dir) / ('%s_s%d.csv' % (a.subset, a.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote %s rows=%d' % (out, len(rows)))


if __name__ == '__main__':
    main()
