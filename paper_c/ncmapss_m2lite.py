"""N-CMAPSS M2-Lite: per-timestep W FiLM, last-step pooling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from paper_c.metrics import engine_mean_rmse, mae, nasa_score, pooled_rmse
from paper_c.models import GRUModel, n_params
from paper_c.ncmapss_data import (CUTS, FEATS, Scaler, load_subset,
                                  make_windows_meta, split_units)
from paper_c.train import set_seed


class M2Lite(nn.Module):
    def __init__(self, input_dim=18, hidden=64, dropout=0.2):
        super().__init__()
        self.back = GRUModel(input_dim=input_dim, hidden=hidden, dropout=dropout)
        self.film = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 2 * hidden))
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask, w):
        h = self.back.seq_features(x, mask)
        gamma, beta = self.film(w).chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        lengths = mask.sum(1).long().clamp(min=1)
        z = h[torch.arange(h.size(0), device=h.device), lengths - 1]
        return self.head(self.back.ln(z)).squeeze(-1)


def run_fold_nc_m2lite(subset, L=50, fold=0, seed=0, epochs=40, patience=8,
                       batch=64, lr=1e-3, wd=1e-4, device='cuda:0', ckpt_dir=None):
    set_seed(seed + fold)
    df = load_subset(subset)
    tr_units, va_units = split_units(df.unit.unique(), fold)
    tr_df = df[df.unit.isin(tr_units)].copy()
    va_df = df[df.unit.isin(va_units)].copy()
    scaler = Scaler().fit(tr_df)
    Xtr, Mtr, ytr, _, _, _, _, Wtr = make_windows_meta(tr_df, tr_units, scaler, L, cuts=None)
    Xva, Mva, yva, eva, _, _, _, Wva = make_windows_meta(va_df, va_units, scaler, L, cuts=CUTS)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr),
                       torch.from_numpy(ytr), torch.from_numpy(Wtr))
    g = torch.Generator().manual_seed(seed + fold)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    model = M2Lite(input_dim=len(FEATS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xva_t = torch.from_numpy(Xva).to(device)
    Mva_t = torch.from_numpy(Mva).to(device)
    Wva_t = torch.from_numpy(Wva).to(device)
    best, best_ep, best_state = float('inf'), -1, None
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, mb, yb, wb in dl:
            xb, mb, yb, wb = xb.to(device), mb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            pred = model(xb, mb, wb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xva_t), 256):
                preds.append(model(Xva_t[i:i + 256], Mva_t[i:i + 256],
                                   Wva_t[i:i + 256]).cpu().numpy())
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
        for i in range(0, len(Xva_t), 256):
            preds.append(model(Xva_t[i:i + 256], Mva_t[i:i + 256],
                               Wva_t[i:i + 256]).cpu().numpy())
    pv = np.concatenate(preds)
    res = dict(subset=subset, model='m2_lite', L=L, fold=fold, seed=seed + fold,
               n_params=n_params(model), best_epoch=best_ep,
               engine_mean_rmse=best, pooled_rmse=pooled_rmse(yva, pv),
               mae=mae(yva, pv), nasa=nasa_score(yva, pv))
    if ckpt_dir:
        p = Path(ckpt_dir); p.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=best_state, config=res), p / f'{subset}_m2lite_f{fold}.pt')
    return res
