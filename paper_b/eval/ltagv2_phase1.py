"""LTAG v3 Phase 1: strong baselines on the frozen cata_gate feature table.

Arms: K1, Median, LTAG, Global-g, Linear-Stack, K1-Calib, LTAG-noMW, LTAG-noCtx.
5-fold GroupKFold by (subset, engine); validation engines never used for fitting.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from paper_b.eval.metrics_v2 import (bias, engine_mean_rmse, mae, nasa_mean,
                                     paired_engine_bootstrap, pooled_rmse)
from paper_b.hc.train_d import rul_loss
from paper_b.modules.learned_gate import FEATS, canon, train_gate

MW = ['p_med', 'p_mean', 'p_std', 'stat', 'shift', 'p_med_minus_k1', 'p_mean_minus_k1']
CTX = ['regime_dist', 'cap', 'cycle_frac']
NO_MW = ['p_k1'] + CTX
NO_CTX = [f for f in FEATS if f not in CTX]


def load_train(root):
    frames = []
    for p in sorted(glob.glob(str(Path(root) / 'train' / '*.csv'))):
        name = Path(p).name
        subset = name.split('_')[0]
        seed = int(name.split('_s')[1].split('.')[0])
        df = pd.read_csv(p)
        df['subset'] = subset
        df['seed'] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_val(root, subset):
    frames = []
    for p in sorted(glob.glob(str(Path(root) / 'preds' / f'{subset}_s*.csv'))):
        df = pd.read_csv(p)
        df['subset'] = subset
        df['seed'] = int(Path(p).name.split('_s')[1].split('.')[0])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class Resid(nn.Module):
    def __init__(self, nf, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_resid(X, y, p_k1, epochs=300, lr=1e-3, wd=1e-3, seed=7):
    torch.manual_seed(seed)
    m = Resid(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X, dtype=torch.float32)
    kt = torch.tensor(p_k1, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        pred = kt + m(Xt)
        loss = rul_loss(pred, yt, 125.0).mean()
        loss.backward()
        opt.step()
    m.eval()
    return m


def fit_gate(X, p_med, p_k1, y, seed=7):
    return train_gate(X, p_med, p_k1, y, hidden=16, epochs=300, lr=1e-3, wd=1e-3, seed=seed)


def metrics(df, pred, name):
    y = df['y_cap'].to_numpy(float)
    return dict(arm=name,
                engine_mean_rmse=engine_mean_rmse(y, pred, df['engine']),
                pooled_rmse=pooled_rmse(y, pred), mae=mae(y, pred),
                bias=bias(y, pred), nasa_mean=nasa_mean(y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/user/sprint24/experiments/hc/cata_gate')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--out', default='/home/user/sprint24/results/ltagv2_p1.json')
    a = ap.parse_args()
    tr = canon(load_train(a.root))
    va = canon(load_val(a.root, a.subset))
    groups = (tr['subset'] + '_' + tr['engine'].astype(str)).to_numpy()
    gkf = GroupKFold(n_splits=5)

    def prep(cols):
        Xtr = tr[cols].to_numpy(float)
        Xva = va[cols].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        return (Xtr - mu) / sd, (Xva - mu) / sd

    p_k1_tr, p_med_tr = tr['p_k1'].to_numpy(float), tr['p_med'].to_numpy(float)
    p_k1_va, p_med_va = va['p_k1'].to_numpy(float), va['p_med'].to_numpy(float)
    y_tr = tr['y_cap'].to_numpy(float)

    # LTAG: 5-fold, average fold gates on val
    Xtr, Xva = prep(FEATS)
    gv = []
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, groups=groups)):
        m = fit_gate(Xtr[ti], p_med_tr[ti], p_k1_tr[ti], y_tr[ti], seed=7 + fold)
        gv.append(m(torch.tensor(Xva, dtype=torch.float32)).detach().numpy())
    g = np.mean(gv, axis=0)
    pred_ltag = g * p_med_va + (1 - g) * p_k1_va

    # K1-Calib: residual on p_k1 + context only
    Xtr_c, Xva_c = prep(NO_MW)
    preds_c = []
    for fold, (ti, vi) in enumerate(gkf.split(Xtr_c, groups=groups)):
        m = fit_resid(Xtr_c[ti], y_tr[ti], p_k1_tr[ti], seed=7 + fold)
        preds_c.append(p_k1_va + m(torch.tensor(Xva_c, dtype=torch.float32)).detach().numpy())
    pred_calib = np.mean(preds_c, axis=0)

    # Global-g
    gs = np.linspace(0, 1, 101)
    best = min(gs, key=lambda q: engine_mean_rmse(y_tr, q * p_med_tr + (1 - q) * p_k1_tr, tr['engine']))
    pred_gg = best * p_med_va + (1 - best) * p_k1_va

    # Linear stack
    Ztr = np.column_stack([p_k1_tr, p_med_tr, tr[FEATS].to_numpy(float)])
    Zva = np.column_stack([p_k1_va, p_med_va, va[FEATS].to_numpy(float)])
    ridge = Ridge(alpha=1.0).fit(Ztr, y_tr)
    pred_ls = ridge.predict(Zva)

    # LTAG-noMW / noCtx
    out = []
    for name, cols, pred in [
        ('K1', None, p_k1_va), ('Median', None, p_med_va),
        ('LTAG', FEATS, pred_ltag), ('Global-g', None, pred_gg),
        ('Linear-Stack', None, pred_ls), ('K1-Calib', NO_MW, pred_calib),
    ]:
        out.append(metrics(va, pred, name))

    # noMW/noCtx gates
    for name, cols in [('LTAG-noMW', NO_MW), ('LTAG-noCtx', NO_CTX)]:
        Xt, Xv = prep(cols)
        preds = []
        for fold, (ti, vi) in enumerate(gkf.split(Xt, groups=groups)):
            m = fit_gate(Xt[ti], p_med_tr[ti], p_k1_tr[ti], y_tr[ti], seed=7 + fold)
            gg = m(torch.tensor(Xv, dtype=torch.float32)).detach().numpy()
            preds.append(gg * p_med_va + (1 - gg) * p_k1_va)
        out.append(metrics(va, np.mean(preds, 0), name))

    boot = paired_engine_bootstrap(pred_ltag, pred_calib, va['y_cap'], va['engine'])
    res = dict(subset=a.subset, n_train=int(len(tr)), n_val=int(len(va)),
               best_g=float(best), ltag_vs_calib=boot, table=out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
