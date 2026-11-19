"""Phase 2 M1/M2 trainer: RHSE-DSPT on frozen-protocol validation.

M1 = clean training; M2 = + static 30% training-window sensor augmentation.
Validation only; test sealed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from paper_b.modules.model_rhse_dspt import RHSE_DSPT
from paper_b.modules.rhse import pca_H
from paper_b.training.data import prepare_train
from paper_b.training.metrics import per_engine_rmse, rmse


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_windows(tr, seed=20261120):
    x = tr['x'].copy()
    rng = np.random.default_rng(seed)
    sel = rng.random(len(x)) < 0.30
    for b in np.where(sel)[0]:
        k = int(rng.integers(1, 3))
        chans = rng.choice(x.shape[-1], size=k, replace=False)
        for j in chans:
            if rng.random() < 0.5:
                x[b, :, j] += rng.normal(0.0, rng.uniform(0.5, 2.5), size=x.shape[1])
            else:
                x[b, :, j] = 0.0
    tr['x'] = x


def predict(model, va, dev, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(va['x']), bs):
            sl = slice(i, i + bs)
            o = model(torch.from_numpy(va['x'][sl]).to(dev),
                      mask=torch.from_numpy(va['mask'][sl]).to(dev),
                      baseline=torch.from_numpy(va['baseline'][sl]).to(dev),
                      cycle=torch.from_numpy(va['cycle'][sl]).to(dev),
                      cond=torch.from_numpy(va['cond'][sl]).to(dev))
            out.append(o['prediction'].reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arm', default='M1', choices=['M1', 'M2'])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha-meas', type=float, default=0.1)
    ap.add_argument('--alpha-mono', type=float, default=0.01)
    ap.add_argument('--data-dir', default='/home/user/rulb/data/CMAPSS')
    ap.add_argument('--out-root', default='/home/user/sprint24/experiments/phase2')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    set_seed(a.seed)
    dev = torch.device(a.device)
    ds = prepare_train(a.subset, a.data_dir, span=70, split_seed=7, fold=0,
                       val_frac=0.2, rul_cap=125.0, n_cond=6, cuts=(0.4, 0.55, 0.7, 0.85, 0.95))
    tr, va = ds['train'], ds['val']
    if a.arm == 'M2':
        augment_windows(tr)
    late = []
    for i in range(0, len(tr['x']), max(1, len(tr['x']) // 4000)):
        mask = tr['mask'][i].astype(bool)
        if mask.sum() >= 10:
            late.append(tr['x'][i][mask] - tr['baseline'][i])
    late = np.concatenate(late, 0)
    sd = late.std(0); sd[sd == 0] = 1.0
    H = pca_H(late / sd, 3)
    model = RHSE_DSPT(H).to(dev)
    groups = [
        dict(params=model.backbone.parameters(), lr=a.lr),
        dict(params=[model.H, model.log_s], lr=a.lr * 0.1),
        dict(params=model.hi_head.parameters(), lr=a.lr),
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    X = torch.from_numpy(tr['x']).to(dev); Y = torch.from_numpy(tr['y']).to(dev)
    MK = torch.from_numpy(tr['mask']).to(dev); BS = torch.from_numpy(tr['baseline']).to(dev)
    CY = torch.from_numpy(tr['cycle']).to(dev); CD = torch.from_numpy(tr['cond']).to(dev)
    best, best_state, best_ep, hist = float('inf'), None, -1, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(X), device=dev)
        tot, nb = 0.0, 0
        for idx in perm.split(a.batch):
            opt.zero_grad(set_to_none=True)
            out = model(X[idx], mask=MK[idx], baseline=BS[idx], cycle=CY[idx], cond=CD[idx])
            y = Y[idx].reshape(-1, 1)
            e = out['prediction'] - y
            loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()
            loss = loss + a.alpha_meas * (out['w'].detach() * out['r'].square()).mean()
            th = out['theta']
            step = (MK[idx][:, 1:] * MK[idx][:, :-1])
            mono = torch.relu(th[:, :-1] - th[:, 1:] - 1e-3).mean(-1)
            loss = loss + a.alpha_mono * (mono * step).sum() / step.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        p = predict(model, va, dev)
        vr = rmse(p, va['y'])
        em = per_engine_rmse(va['unit'], va['y'], p)
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), val_pooled=vr, val_engine=em))
        print(f'  ep{ep:03d} loss={tot/max(nb,1):.3f} val={em:.4f} pooled={vr:.4f}', flush=True)
        if em < best - 1e-4:
            best, best_ep = em, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= a.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    p = predict(model, va, dev)
    out_dir = Path(a.out_root) / f'phase2_{a.arm}_{a.subset}_s{a.seed}'
    (out_dir / 'ckpt').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / 'ckpt' / 'best.pt')
    metrics = dict(val_engine_mean_rmse=float(per_engine_rmse(va['unit'], va['y'], p)),
                   val_pooled_rmse=float(rmse(p, va['y'])),
                   best_ep=int(best_ep), epochs_run=len(hist),
                   params=int(sum(q.numel() for q in model.parameters())),
                   elapsed_s=round(time.time() - t0, 1))
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    (out_dir / 'config.json').write_text(json.dumps(vars(a), indent=1))
    print('DONE', json.dumps(metrics))


if __name__ == '__main__':
    main()
