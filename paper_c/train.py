"""Paper C training loop for Phase 0 backbone selection."""
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
from paper_c.metrics import engine_mean_rmse, mae, nasa_score, pooled_rmse
from paper_c.models import build, n_params


def set_seed(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def run_fold(subset, model_name, L, fold, seed=0, epochs=40, patience=8,
             batch=256, lr=1e-3, wd=1e-4, device='cuda:0', ckpt_dir=None,
             data_dir='/home/user/rulb/data/CMAPSS', deterministic=False,
             model_kwargs=None):
    set_seed(seed + fold, deterministic)
    df = add_rul(load_train(subset, data_dir))
    tr_units, va_units = split_units(df.unit.unique(), fold)
    tr_df = df[df.unit.isin(tr_units)].copy()
    va_df = df[df.unit.isin(va_units)].copy()
    scaler = Scaler(subset).fit(tr_df)
    Xtr, Mtr, ytr = make_train_windows(tr_df, tr_units, scaler, L)
    Xva, Mva, yva, eva, cva = make_val_windows(va_df, va_units, scaler, L, CUTS)

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr), torch.from_numpy(ytr))
    g = torch.Generator().manual_seed(seed + fold)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g, drop_last=False)

    model = build(model_name, **(model_kwargs or {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    Xva_t = torch.from_numpy(Xva).to(device)
    Mva_t = torch.from_numpy(Mva).to(device)
    best, best_ep, best_state = float('inf'), -1, None
    bad = 0
    for ep in range(epochs):
        model.train()
        for xb, mb, yb in dl:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, mb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xva_t), 512):
                preds.append(model(Xva_t[i:i + 512], Mva_t[i:i + 512]).cpu().numpy())
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
            preds.append(model(Xva_t[i:i + 512], Mva_t[i:i + 512]).cpu().numpy())
    pv = np.concatenate(preds)
    res = dict(subset=subset, model=model_name, L=L, fold=fold, seed=seed + fold,
               model_kwargs=model_kwargs or {},
               n_params=n_params(model), best_epoch=best_ep,
               engine_mean_rmse=best, pooled_rmse=pooled_rmse(yva, pv),
               mae=mae(yva, pv), nasa=nasa_score(yva, pv),
               n_train=int(len(ytr)), n_val=int(len(yva)))
    if ckpt_dir:
        p = Path(ckpt_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=best_state, config=res), p / f'{subset}_{model_name}_L{L}_f{fold}.pt')
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--model', default='gru')
    ap.add_argument('--L', type=int, default=50)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--ckpt-dir', default=None)
    a = ap.parse_args()
    t = time.time()
    print(json.dumps(run_fold(a.subset, a.model, a.L, a.fold, epochs=a.epochs,
                              ckpt_dir=a.ckpt_dir), indent=1))
    print('elapsed', round(time.time() - t, 1))
