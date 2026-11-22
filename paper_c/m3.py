"""Paper C M3 CD-RCL: censored Beta-NLL head + 90% interval diagnostics."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from paper_c.data import (CUTS, Scaler, add_rul, load_train, make_train_windows,
                          make_val_windows, split_units)
from paper_c.metrics import engine_mean_rmse, mae, nasa_score, picp_mpiw, pooled_rmse
from paper_c.models import GRUModel, n_params
from paper_c.train import set_seed


class M3Model(nn.Module):
    def __init__(self, input_dim=14, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.back = GRUModel(input_dim, hidden, layers, dropout)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, mask):
        z = self.back.ln(self.back.features(x, mask))
        out = self.head(z)
        return out[:, 0], out[:, 1]


def beta_nll(mu, log_sigma, y, beta=0.5, censored=True):
    sigma2 = (torch.nn.functional.softplus(log_sigma) + 1e-3) ** 2
    w = sigma2.detach() ** beta
    unc = y < 125.0
    loss = torch.zeros_like(mu)
    if unc.any():
        loss[unc] = 0.5 * torch.log(sigma2[unc]) + (y[unc] - mu[unc]) ** 2 / (2 * sigma2[unc])
    if censored and (~unc).any():
        z = (125.0 - mu[~unc]) / torch.sqrt(sigma2[~unc])
        # P(Y > 125) = 1 - Phi(z)
        tail = 0.5 * (1.0 - torch.erf(z / np.sqrt(2.0)))
        loss[~unc] = -torch.log(tail.clamp(min=1e-6))
    return (w * loss).mean()


def vanilla_nll(mu, log_sigma, y):
    sigma2 = (torch.nn.functional.softplus(log_sigma) + 1e-3) ** 2
    return (0.5 * torch.log(sigma2) + (y - mu) ** 2 / (2 * sigma2)).mean()


def run_fold_m3(subset, fold, L=50, seed=0, epochs=40, patience=8, batch=256,
                lr=1e-3, wd=1e-4, device='cuda:0', ckpt_dir=None,
                loss_name='beta', data_dir='/home/user/rulb/data/CMAPSS'):
    set_seed(seed + fold)
    df = add_rul(load_train(subset, data_dir))
    tr_units, va_units = split_units(df.unit.unique(), fold)
    tr_df = df[df.unit.isin(tr_units)].copy()
    va_df = df[df.unit.isin(va_units)].copy()
    scaler = Scaler(subset).fit(tr_df)
    Xtr, Mtr, ytr = make_train_windows(tr_df, tr_units, scaler, L)
    Xva, Mva, yva, eva, cva = make_val_windows(va_df, va_units, scaler, L, CUTS)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr))
    g = torch.Generator().manual_seed(seed + fold)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    model = M3Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xva_t = torch.from_numpy(Xva).to(device)
    Mva_t = torch.from_numpy(Mva).to(device)
    best, best_ep, best_state = float('inf'), -1, None
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, mb, yb in dl:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            mu, ls = model(xb, mb)
            loss = beta_nll(mu, ls, yb) if loss_name == 'beta' else vanilla_nll(mu, ls, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        mus, sigs = [], []
        with torch.no_grad():
            for i in range(0, len(Xva_t), 512):
                mu, ls = model(Xva_t[i:i + 512], Mva_t[i:i + 512])
                mus.append(mu.cpu().numpy())
                sigs.append((torch.nn.functional.softplus(ls) + 1e-3).cpu().numpy())
        pv = np.clip(np.concatenate(mus), 0, 125)
        sv = np.concatenate(sigs)
        rmse = engine_mean_rmse(yva, pv, eva)
        if rmse < best - 1e-6:
            best, best_ep = rmse, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    mus, sigs = [], []
    with torch.no_grad():
        for i in range(0, len(Xva_t), 512):
            mu, ls = model(Xva_t[i:i + 512], Mva_t[i:i + 512])
            mus.append(mu.cpu().numpy())
            sigs.append((torch.nn.functional.softplus(ls) + 1e-3).cpu().numpy())
    pv = np.clip(np.concatenate(mus), 0, 125)
    sv = np.concatenate(sigs)
    lo = np.clip(pv - 1.645 * sv, 0, 125)
    hi = np.clip(pv + 1.645 * sv, 0, 125)
    res = dict(subset=subset, model=f'm3_{loss_name}', L=L, fold=fold,
               seed=seed + fold, n_params=n_params(model), best_epoch=best_ep,
               engine_mean_rmse=best, pooled_rmse=pooled_rmse(yva, pv),
               mae=mae(yva, pv), nasa=nasa_score(yva, pv),
               picp=picp_mpiw(yva, lo, hi)['picp'],
               mpiw=picp_mpiw(yva, lo, hi)['mpiw'])
    if ckpt_dir:
        p = Path(ckpt_dir); p.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=best_state, config=res), p / f'{subset}_m3_{loss_name}_f{fold}.pt')
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--loss', default='beta', choices=['beta', 'vanilla'])
    ap.add_argument('--epochs', type=int, default=20)
    a = ap.parse_args()
    t = time.time()
    print(json.dumps(run_fold_m3(a.subset, a.fold, epochs=a.epochs,
                                 loss_name=a.loss), indent=1))
    print('elapsed', round(time.time() - t, 1))
