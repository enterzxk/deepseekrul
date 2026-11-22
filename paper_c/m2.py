"""Paper C M2 CMST: condition-modulated sensor gate + dual-scale causal conv + attention pooling."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from paper_c.data import (CUTS, Scaler, add_rul, load_train, make_train_windows,
                          make_val_windows, split_units)
from paper_c.metrics import engine_mean_rmse, mae, nasa_score, pooled_rmse
from paper_c.models import GRUModel, n_params
from paper_c.train import set_seed


class M2Model(nn.Module):
    def __init__(self, gate_only=False, pool_only=False, hidden=64, dropout=0.2):
        super().__init__()
        self.gate_only = gate_only
        self.pool_only = pool_only
        self.back = GRUModel(hidden=hidden, dropout=dropout)
        self.cond = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 14))
        self.film = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 2 * hidden))
        self.dw3 = nn.Conv1d(hidden, hidden, 3, groups=hidden)
        self.pw3 = nn.Conv1d(hidden, hidden, 1)
        self.dw11 = nn.Conv1d(hidden, hidden, 11, groups=hidden)
        self.pw11 = nn.Conv1d(hidden, hidden, 1)
        self.att = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask, op):
        if not self.pool_only:
            g = 2.0 * torch.sigmoid(self.cond(op))
            x = x * g[:, None, :]
        h = self.back.seq_features(x, mask)
        if not self.pool_only:
            gamma, beta = self.film(op).chunk(2, dim=-1)
            h = (1.0 + gamma[:, None, :]) * h + beta[:, None, :]
        if self.gate_only:
            lengths = mask.sum(1).long().clamp(min=1)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1))
            z = h.gather(1, idx).squeeze(1)
        else:
            ht = h.transpose(1, 2)
            c3 = self.pw3(F.relu(self.dw3(F.pad(ht, (2, 0)))))
            c11 = self.pw11(F.relu(self.dw11(F.pad(ht, (10, 0)))))
            ht = ht + c3 + c11
            h = ht.transpose(1, 2)
            score = self.att(h).squeeze(-1) + (mask - 1.0) * 1e9
            a = torch.softmax(score, dim=1)
            z = (a.unsqueeze(-1) * h).sum(1)
        return self.head(self.back.ln(z)).squeeze(-1)


def run_fold_m2(subset, fold, L=50, seed=0, epochs=40, patience=8, batch=256,
                lr=1e-3, wd=1e-4, device='cuda:0', ckpt_dir=None,
                arm='m2', data_dir='/home/user/rulb/data/CMAPSS'):
    set_seed(seed + fold)
    df = add_rul(load_train(subset, data_dir))
    tr_units, va_units = split_units(df.unit.unique(), fold)
    tr_df = df[df.unit.isin(tr_units)].copy()
    va_df = df[df.unit.isin(va_units)].copy()
    scaler = Scaler(subset).fit(tr_df)
    Xtr, Mtr, ytr, Otr = make_train_windows(tr_df, tr_units, scaler, L, return_op=True)
    Xva, Mva, yva, eva, cva, Ova = make_val_windows(va_df, va_units, scaler, L, CUTS, return_op=True)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr),
                       torch.from_numpy(ytr), torch.from_numpy(Otr))
    g = torch.Generator().manual_seed(seed + fold)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    model = M2Model(gate_only=(arm == 'm2_gate_only'),
                    pool_only=(arm == 'm2_pool_only')).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xva_t = torch.from_numpy(Xva).to(device)
    Mva_t = torch.from_numpy(Mva).to(device)
    Ova_t = torch.from_numpy(Ova).to(device)
    best, best_ep, best_state = float('inf'), -1, None
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, mb, yb, ob in dl:
            xb, mb, yb, ob = xb.to(device), mb.to(device), yb.to(device), ob.to(device)
            opt.zero_grad()
            pred = model(xb, mb, ob)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xva_t), 512):
                preds.append(model(Xva_t[i:i + 512], Mva_t[i:i + 512], Ova_t[i:i + 512]).cpu().numpy())
        pv = np.concatenate(preds)
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
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xva_t), 512):
            preds.append(model(Xva_t[i:i + 512], Mva_t[i:i + 512], Ova_t[i:i + 512]).cpu().numpy())
    pv = np.concatenate(preds)
    res = dict(subset=subset, model=arm, L=L, fold=fold, seed=seed + fold,
               n_params=n_params(model), best_epoch=best_ep,
               engine_mean_rmse=best, pooled_rmse=pooled_rmse(yva, pv),
               mae=mae(yva, pv), nasa=nasa_score(yva, pv))
    if ckpt_dir:
        p = Path(ckpt_dir); p.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=best_state, config=res), p / f'{subset}_{arm}_f{fold}.pt')
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--arm', default='m2')
    ap.add_argument('--epochs', type=int, default=20)
    a = ap.parse_args()
    t = time.time()
    print(json.dumps(run_fold_m2(a.subset, a.fold, epochs=a.epochs, arm=a.arm), indent=1))
    print('elapsed', round(time.time() - t, 1))
