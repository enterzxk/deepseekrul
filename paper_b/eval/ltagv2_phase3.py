"""LTAG v3 Phase 3: mechanism/metrics for the post-calibration candidate (LS-ctx).

Questions: is the gain just a constant bias shift? Does it help the tail? Does it
trade RMSE for MAE/bias? Test sealed; validation only.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from paper_b.eval.metrics_v2 import (bias, engine_mean_rmse, mae, nasa_split,
                                     pooled_rmse)
from paper_b.modules.learned_gate import FEATS, canon

CTX = ['regime_dist', 'cap', 'cut_norm', 'hist_norm']
FEATS_OBS = [f for f in FEATS if f != 'cycle_frac'] + ['cut_norm', 'hist_norm']


def load(root, kind):
    frames = []
    for p in sorted(glob.glob(str(Path(root) / kind / '*.csv'))):
        name = Path(p).name
        subset = name.split('_')[0]
        seed = int(name.split('_s')[1].split('.')[0])
        if seed not in (101, 202, 303):
            continue
        df = canon(pd.read_csv(p))
        df['subset'] = subset
        df['seed'] = seed
        df['cut_norm'] = df['cut'] / 250.0
        df['hist_norm'] = np.minimum(df['cut'], 70.0) / 70.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def met(df, pred, name):
    y = df['y_cap'].to_numpy(float)
    ns = nasa_split(y, pred)
    return dict(arm=name, engine_mean=engine_mean_rmse(y, pred, df['engine']),
                pooled=pooled_rmse(y, pred), mae=mae(y, pred), bias=bias(y, pred),
                nasa_late=ns['late_mean'], nasa_early=ns['early_mean'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/cata_gate')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--out', default='/home/user/sprint24/results/ltagv2_p3.json')
    a = ap.parse_args()
    tr = load(a.root, 'train')
    va = load(a.root, 'preds')
    va = va[va.subset == a.subset].copy()
    y_tr = tr['y_cap'].to_numpy(float)
    p_k1_tr = tr['p_k1'].to_numpy(float)
    p_med_tr = tr['p_med'].to_numpy(float)
    p_k1_va = va['p_k1'].to_numpy(float)
    p_med_va = va['p_med'].to_numpy(float)

    # K1 + constant bias fitted on training engines
    best_b, best_r = 0.0, float('inf')
    for b in np.arange(-5, 5.01, 0.1):
        r = engine_mean_rmse(y_tr, p_k1_tr + b, tr['engine'])
        if r < best_r:
            best_b, best_r = float(b), r
    pred_b = p_k1_va + best_b

    # LS-ctx
    cols = ['p_k1', 'p_med'] + CTX
    m = Ridge(alpha=1.0).fit(tr[cols].to_numpy(float), y_tr)
    pred_ls = m.predict(va[cols].to_numpy(float))

    rows = [met(va, p_k1_va, 'K1'), met(va, pred_b, f'K1+b (b={best_b:+.1f})'),
            met(va, pred_ls, 'LS-ctx')]

    # Tail analysis: mean Δ|e| by K1 error quantile
    e_k1 = np.abs(p_k1_va - va['y_cap'].to_numpy(float))
    e_ls = np.abs(pred_ls - va['y_cap'].to_numpy(float))
    d = e_ls - e_k1
    qs = np.quantile(e_k1, [0.5, 0.9, 1.0])
    bins = [0, qs[0], qs[1], qs[2] + 1e-9]
    tail = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (e_k1 >= lo) & (e_k1 < hi)
        tail.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                         mean_delta=float(d[m].mean()) if m.any() else None))
    res = dict(subset=a.subset, best_b=best_b, table=rows, tail=tail)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
