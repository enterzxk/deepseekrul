"""Test-time features for the frozen global learned gate.

Builds K causal windows ending at L, L-5, ... for each official test engine,
predicts RUL at each, age-aligns to L, and writes the same feature columns the
gate was trained on. Only observable quantities are used: the current observed
length L, the current cycle, the window content, and the operating variables.
The engine's failure time is never used.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_b.hc import config_hc as C
from paper_b.modules.temporal_agg import build_model, predict_windows
from paper_b.training.data import (OP_COLS, _baseline_ref, _prepare_split,
                                   _window, load_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--root', default=str(Path.home() / 'sprint24/experiments/hc/_gate'))
    ap.add_argument('--data-dir', default=str(Path.home() / 'rulb/data/CMAPSS'))
    ap.add_argument('--out-dir', default=str(Path.home() / 'sprint24/experiments/hc/cata_gate/test'))
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

    s = _prepare_split(a.subset, a.data_dir, C.SPLIT_SEED, C.FOLD, C.VAL_FRAC, C.N_COND)
    scaler, pop = s['scaler'], s['pop']
    test, rul_map = load_test(a.data_dir, a.subset)
    test = test.sort_values(['unit', 'cycle']).reset_index(drop=True)
    ts = scaler.transform(test)
    lt = scaler.labels(test)
    tr_op = s['tr'][OP_COLS].to_numpy(float)
    mu, sd = tr_op.mean(0), tr_op.std(0)
    sd[sd == 0] = 1.0
    test_opz = (test[OP_COLS].to_numpy(float) - mu) / sd
    op_map = {(int(u), int(c)): test_opz[k]
              for k, (u, c) in enumerate(zip(test['unit'], test['cycle']))}

    rows = []
    for uid, g in test.groupby('unit', sort=True):
        uid = int(uid)
        pos = g.index.to_numpy()
        seq = np.asarray(ts[pos], np.float32)
        cl = np.asarray(lt[pos], np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        L = len(seq)
        ends = [L - j * a.delta for j in range(a.K) if L - j * a.delta >= 1]
        xs, ms, bs, cds = [], [], [], []
        for e in ends:
            t = e - 1
            x, mask = _window(seq, e, C.SPAN)
            xs.append(x); ms.append(mask)
            bs.append(_baseline_ref(seq, cl, cyc, t, pop))
            cds.append(int(cl[t]))
        eb = dict(x=np.stack(xs)[None], mask=np.stack(ms)[None],
                  baseline=np.stack(bs)[None], cond=np.asarray(cds)[None])
        preds = predict_windows(net, eb, 0, np.arange(len(ends)), dev)
        sel_idx = np.asarray([e - 1 for e in ends], np.int64)
        aligned = preds - (L - np.asarray(ends, np.float64))
        med = float(np.median(aligned))
        mad = float(np.median(np.abs(aligned - med)))
        ops = np.stack([op_map[(uid, int(e))] for e in ends])
        dmat = np.linalg.norm(ops[:, None, :] - ops[None, :, :], axis=-1)
        y_raw = float(rul_map[uid])
        rows.append(dict(
            subset=a.subset, seed=a.seed, engine=uid, cut=L, n=L,
            y_cap=float(min(y_raw, 125.0)), y_raw=y_raw,
            p_k1=float(preds[0]), p_med=med,
            p_mean=float(aligned.mean()), p_std=float(aligned.std(ddof=0)),
            stat=mad / max(abs(med), 1.0), shift=abs(med - float(preds[0])),
            regime_dist=float(dmat.max()),
            cap=float(float(preds[0]) >= 124.5 or med >= 124.5),
            cycle_frac=1.0))
    out = Path(a.out_dir) / ('%s_s%d.csv' % (a.subset, a.seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote %s rows=%d' % (out, len(rows)))


if __name__ == '__main__':
    main()
