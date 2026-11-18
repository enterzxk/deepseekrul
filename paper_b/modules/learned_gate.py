"""Cross-fitted learned gate/aggregator for CATA.

The gate is a small MLP on per-cut features (the K=1 prediction, the aligned
median, their difference, dispersion, regime distance, cap flag, cycle
fraction). It outputs g in [0,1] and the prediction is

    pred = g * median(aligned) + (1-g) * newest

so it can learn to use the temporal aggregate where it helps and fall back to
the single window where it hurts. Training is on TRAINING engines only, with
GroupKFold by engine; validation engines are never used for fitting or for
choosing the number of epochs.

The gate is initialized at g=0.5 (zero last layer), not at 0 or 1, so the
comparison is not biased toward either endpoint.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from paper_b.hc.train_d import rul_loss

FEATS = ['p_k1', 'p_med', 'p_mean', 'p_std', 'stat', 'shift',
         'regime_dist', 'cap', 'cycle_frac',
         'p_med_minus_k1', 'p_mean_minus_k1']


def canon(df):
    df = df.copy()
    if 'pred_k1' in df.columns:
        df['p_k1'] = df['pred_k1']
    if 'pred_all' in df.columns:
        df['p_med'] = df['pred_all']
    df['p_med_minus_k1'] = df['p_med'] - df['p_k1']
    df['p_mean_minus_k1'] = df['p_mean'] - df['p_k1']
    return df


class Gate(nn.Module):
    def __init__(self, nf, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nf, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_gate(X, p_med, p_k1, y, hidden=16, epochs=300, lr=1e-3, wd=1e-3,
               seed=7):
    torch.manual_seed(seed)
    model = Gate(X.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X, dtype=torch.float32)
    mt = torch.tensor(p_med, dtype=torch.float32)
    kt = torch.tensor(p_k1, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        g = model(Xt)
        pred = g * mt + (1.0 - g) * kt
        loss = rul_loss(pred, yt, 125.0).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model


def engine_mean_rmse(df, pred):
    p = np.clip(np.asarray(pred, float), 0.0, 125.0)
    y = df['y_cap'].to_numpy(float)
    se = (p - y) ** 2
    tmp = pd.DataFrame(dict(engine=df['engine'].to_numpy(), se=se))
    return float(np.sqrt(tmp.groupby('engine')['se'].mean()).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()

    tr = canon(pd.read_csv(a.train_csv))
    va = canon(pd.read_csv(a.val_csv))
    Xtr = tr[FEATS].to_numpy(float)
    Xva = va[FEATS].to_numpy(float)
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    p_med_tr = tr['p_med'].to_numpy(float)
    p_k1_tr = tr['p_k1'].to_numpy(float)
    p_med_va = va['p_med'].to_numpy(float)
    p_k1_va = va['p_k1'].to_numpy(float)
    y_tr = tr['y_cap'].to_numpy(float)

    gkf = GroupKFold(n_splits=a.folds)
    oof_g = np.zeros(len(tr))
    models = []
    for fold, (ti, vi) in enumerate(gkf.split(Xtr, groups=tr['engine'].to_numpy())):
        m = train_gate(Xtr[ti], p_med_tr[ti], p_k1_tr[ti], y_tr[ti],
                       hidden=a.hidden, epochs=a.epochs, lr=a.lr, wd=a.wd,
                       seed=a.seed + fold)
        oof_g[vi] = m(torch.tensor(Xtr[vi], dtype=torch.float32)).detach().numpy()
        models.append(m)
    oof_pred = oof_g * p_med_tr + (1.0 - oof_g) * p_k1_tr

    gv = np.mean([m(torch.tensor(Xva, dtype=torch.float32)).detach().numpy()
                  for m in models], axis=0)
    pred_va = gv * p_med_va + (1.0 - gv) * p_k1_va

    rng = np.random.default_rng(12345)
    rgta = np.where(va['stat'].to_numpy(float) <= 0.05, p_med_va, p_k1_va)
    rate = float(np.mean(va['stat'].to_numpy(float) <= 0.05))
    rand_g = (rng.random(len(va)) <= rate).astype(float)
    rand_pred = rand_g * p_med_va + (1.0 - rand_g) * p_k1_va

    rows = []
    for name, pred in [('k1', p_k1_va), ('median', p_med_va), ('rgta', rgta),
                       ('random', rand_pred), ('learned', pred_va)]:
        rmse = engine_mean_rmse(va, pred)
        rows.append(dict(config=name, engine_mean_rmse=rmse,
                         delta_vs_k1=engine_mean_rmse(va, p_k1_va) - rmse,
                         gate_mean=float(np.mean(pred_va == p_med_va))
                         if name == 'learned' else None))
    out = dict(subset=a.subset, n_train=int(len(tr)), n_val=int(len(va)),
               oof_rmse=dict(k1=engine_mean_rmse(tr, p_k1_tr),
                             median=engine_mean_rmse(tr, p_med_tr),
                             learned=engine_mean_rmse(tr, oof_pred)),
               oof_gate_mean=float(oof_g.mean()),
               val_gate_mean=float(gv.mean()),
               val_gate_frac_gt_half=float((gv > 0.5).mean()),
               rgta_rate=rate, table=rows)
    print(json.dumps(out, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    pd.DataFrame(dict(engine=va['engine'], cut=va['cut'], y_cap=va['y_cap'],
                      p_k1=p_k1_va, p_med=p_med_va, g=gv,
                      pred_learned=pred_va, pred_rgta=rgta,
                      pred_random=rand_pred)).to_csv(
        str(Path(a.out).with_suffix('.preds.csv')), index=False)


if __name__ == '__main__':
    main()
