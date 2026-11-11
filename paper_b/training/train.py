"""Unified trainer for Paper B (DSPT and Vanilla baseline).

Protocol (frozen):
  split by engine id, 80/20, split_seed=7, fold=0
  model seeds 101/202/303
  RUL cap 125, endpoint RUL=0 at the last training cycle
  early stopping on engine-level truncated validation RMSE
    (cuts 0.4/0.55/0.7/0.85/0.95, from paper_b/configs/protocol.json;
     the engine is the statistical unit -- see audit R2)
  no EMA selection (single, predetermined checkpoint rule)
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import prepare_train
from .metrics import per_engine_rmse, rmse, summarize
from ..models import build_dspt, build_vanilla

DEFAULTS = dict(
    model='dspt', subset='FD001', seed=101, span=70, epochs=40, patience=12,
    batch=512, lr=1e-3, weight_decay=1e-4, d_model=64, layers=2, dropout=0.2,
    mode='dual', ms_mode='full', sa_mode='off', hi_source='none', hi_mono='none',
    hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=True, use_pad_mask=True,
    input_mode='raw', aux_dual=True, instnorm=False, rul_cap=125.0,
    split_seed=7, fold=0, val_frac=0.2, n_cond=6, cuts=(0.4, 0.55, 0.7, 0.85, 0.95),
    legacy_rul=False, global_hi=False, legacy_ms_pad=False, legacy_ema=False,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build(cfg):
    if cfg['model'] == 'dspt':
        return build_dspt(num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
                          dropout=cfg['dropout'], max_span=cfg['span'],
                          instnorm=cfg['instnorm'], ms_mode=cfg['ms_mode'],
                          sa_mode=cfg['sa_mode'], hi_source=cfg['hi_source'],
                          hi_mono=cfg['hi_mono'], use_pad_mask=cfg['use_pad_mask'],
                          use_cycle=cfg['use_cycle'], n_cond=cfg['n_cond'],
                          legacy_ms_pad=cfg['legacy_ms_pad'], legacy_ema=cfg['legacy_ema'])
    return build_vanilla(num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
                         dropout=cfg['dropout'], max_span=cfg['span'],
                         input_mode=cfg['input_mode'], use_cycle=cfg['use_cycle'])


def _tensors(d, dev, keys):
    return [torch.from_numpy(np.ascontiguousarray(d[k])).to(dev) for k in keys]


def predict(net, d, dev, mode='dual', bs=512):
    net.eval()
    n = len(d['y'])
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                o = net(torch.from_numpy(d['x'][sl]).to(dev),
                    mask=torch.from_numpy(d['mask'][sl]).to(dev),
                    baseline=torch.from_numpy(d['baseline'][sl]).to(dev),
                    cycle=(torch.from_numpy(d['cycle'][sl]).to(dev) if net.use_cycle else None),
                    cond=torch.from_numpy(d['cond'][sl]).to(dev), mode=mode)
            out.append(o['prediction'].reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def train_one(cfg, device='cuda:0', out_root=None):
    cfg = {**DEFAULTS, **cfg}
    data_dir = Path(cfg['data_dir'])
    set_seed(cfg['seed'])
    t0 = time.time()

    ds = prepare_train(cfg['subset'], data_dir, span=cfg['span'],
                       split_seed=cfg['split_seed'], fold=cfg['fold'],
                       val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                       legacy_rul=cfg['legacy_rul'], global_hi=cfg['global_hi'])
    tr, va = ds['train'], ds['val']
    dev = torch.device(device)
    X, Y, MK, BS, CY, CD = _tensors(tr, dev, ('x', 'y', 'mask', 'baseline', 'cycle', 'cond'))
    HT = torch.from_numpy(tr['hi']).to(dev)
    VX, VY, VMK, VBS, VCY, VCD = _tensors(va, dev, ('x', 'y', 'mask', 'baseline', 'cycle', 'cond'))

    torch.backends.cudnn.benchmark = True
    net = _build(cfg).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-5)
    g = torch.Generator(device=dev).manual_seed(cfg['seed'])

    def val_metrics():
        """Return (pooled, engine_mean, predictions).

        Selection uses ``engine_mean``: the task book makes the validation
        engine the statistical unit. Every validation engine contributes
        exactly five windows, so the pooled value already weights engines
        equally (1/E each); the difference is the aggregation order --
        ``sqrt(mean MSE_e) >= mean sqrt(MSE_e)`` by Jensen -- and squaring
        amplifies the tail, so a few hard engines widen the gap. ``pooled`` is
        kept for comparison with historical numbers only.
        """
        p = predict(net, va, dev, cfg['mode'])
        return rmse(p, va['y']), per_engine_rmse(va['unit'], va['y'], p), p

    best, best_state, best_ep, history = float('inf'), None, -1, []
    for ep in range(1, cfg['epochs'] + 1):
        net.train()
        perm = torch.randperm(len(X), generator=g, device=dev)
        tot, nb = 0.0, 0
        for idx in perm.split(cfg['batch']):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = net(X[idx], mask=MK[idx], baseline=BS[idx],
                          cycle=(CY[idx] if net.use_cycle else None), cond=CD[idx], mode=cfg['mode'])
                y = Y[idx].reshape(-1, 1)
                e = out['prediction'] - y
                loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()
                if getattr(net, 'dual_span', False) and cfg['aux_dual']:
                    eA = out['y_A'] - y
                    eB = out['y_B'] - y
                    loss = loss + 0.25 * (eA.square().mean() + eB.square().mean())
                if cfg['hi_source'] != 'none' and cfg['hi_lambda'] > 0:
                    ht = HT[idx].reshape(-1)
                    loss = loss + cfg['hi_lambda'] * torch.mean((out['hi'].reshape(-1) - ht) ** 2)
                    if cfg['hi_mono'] != 'none' and cfg['hi_mono_coef'] > 0:
                        hs = out['hi_seq']
                        step = (MK[idx][:, 1:] * MK[idx][:, :-1])
                        diff = hs[:, 1:] - hs[:, :-1]
                        # HI is a HEALTH index: it decreases with life, so the
                        # correct monotone penalty punishes increases (diff > 0).
                        pen = torch.relu(diff) if cfg['hi_mono'] == 'correct' else torch.relu(-diff)
                        loss = loss + cfg['hi_mono_coef'] * (pen * step).sum() / step.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        pooled, vr, _ = val_metrics()
        history.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_rmse=vr,
                            val_pooled_rmse=pooled))
        if ep % 5 == 0 or ep == 1:
            print('  ep%03d loss=%.4f val=%.4f pooled=%.4f best=%.4f@%d'
                  % (ep, tot / max(nb, 1), vr, pooled, best, best_ep), flush=True)
        if vr < best - 1e-4:
            best, best_ep = vr, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= cfg['patience']:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    pv = predict(net, va, dev, cfg['mode'])
    vr = rmse(pv, va['y'])
    em = per_engine_rmse(va['unit'], va['y'], pv)
    m = summarize(pv, va['y'])
    # Main metric = engine-level mean (audit R2). ``val_engine_rmse`` is kept as
    # a legacy alias of the POOLED value because that is what the field always
    # contained; read ``val_engine_mean_rmse`` for the engine-level number.
    m.update(val_engine_rmse=vr, val_pooled_rmse=vr, val_engine_mean_rmse=em,
             best_val_rmse=best, best_ep=best_ep,
             params=int(nparam), epochs_run=len(history), elapsed_s=round(time.time() - t0, 1))
    per_cut = {}
    for c in np.unique(va['cut']):
        sel = va['cut'] == c
        per_cut[int(c)] = dict(n=int(sel.sum()), rmse=rmse(pv[sel], va['y'][sel]))
    m['per_cut'] = per_cut

    out = Path(out_root or 'paper_b/results/raw') / cfg['tag'] / cfg['subset'] / f"s{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / 'model.pt')
    (out / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    (out / 'metrics.json').write_text(json.dumps(m, indent=1))
    (out / 'history.json').write_text(json.dumps(history, indent=0))
    import pandas as pd
    pd.DataFrame(dict(unit=va['unit'], cut=va['cut'], y_true=va['y'], y_pred=pv)).to_csv(
        out / 'val_pred.csv', index=False)
    print('DONE %s %s s%d val=%.4f ep=%d params=%d %.1fs' % (
        cfg['tag'], cfg['subset'], cfg['seed'], vr, best_ep, nparam, m['elapsed_s']), flush=True)
    return m
