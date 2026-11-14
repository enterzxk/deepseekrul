"""S2: breakeven precision of a total-life auxiliary input (diagnostic only).

Frozen backbone = F_full_basic_hi1 (DSPT, ms=full / sa=basic / hi=hf / cycle on).
Treatment    = one extra scalar L*(1+eps) concatenated before the regression head.
Noise        = eps ~ N(0, sigma^2) drawn ONCE per engine per seed and shared by
               every window of that engine (per-window noise would average out
               and flatten the dose-response curve).
Evaluation   = validation engines at the five frozen cuts only. Official test is
               never opened in this script.

Usage:
  python train_s2.py --subset FD001 --seed 101 --sigma 0.10
  python train_s2.py --grid        # whole 100-run grid, resumable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get('S2_REPO', '/home/user/deepseek-rul'))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

from paper_b.models.dspt import build_dspt                      # noqa: E402
from paper_b.training.data import prepare_train                 # noqa: E402
from paper_b.training.metrics import nasa_score, per_engine_rmse, rmse  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import DSPTWithLife                                  # noqa: E402

DATA = Path(os.environ.get('S2_DATA', str(REPO / 'data' / 'CMAPSS')))
OUT = Path(os.environ.get('S2_OUT', '/home/user/s2/runs'))
PROTO = json.loads((REPO / 'paper_b/configs/protocol.json').read_text())

BASE_CFG = dict(
    model='dspt', span=70, epochs=40, patience=12, batch=512, lr=1e-3, weight_decay=1e-4,
    d_model=64, layers=2, dropout=0.2, mode='dual', ms_mode='full', sa_mode='basic',
    hi_source='hf', hi_mono='correct', hi_lambda=0.5, hi_mono_coef=0.1,
    use_cycle=True, use_pad_mask=True, input_mode='raw', aux_dual=True, instnorm=False,
    rul_cap=125, split_seed=7, fold=0, val_frac=0.2, n_cond=6,
    cuts=tuple(PROTO['val_cuts']), legacy_rul=False, global_hi=False,
    legacy_ms_pad=False, legacy_ema=False,
)
SEEDS = (101, 202, 303, 42, 43)
SIGMAS = (None, 0.05, 0.10, 0.20, 0.30)
SUBSETS = tuple(PROTO['subsets'])


def tag_of(sigma):
    return 'ctrl' if sigma is None else f's{int(round(sigma * 100)):03d}'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_net(cfg, use_life):
    kw = dict(num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
              dropout=cfg['dropout'], max_span=cfg['span'], instnorm=cfg['instnorm'],
              ms_mode=cfg['ms_mode'], sa_mode=cfg['sa_mode'], hi_source=cfg['hi_source'],
              hi_mono=cfg['hi_mono'], use_pad_mask=cfg['use_pad_mask'],
              use_cycle=cfg['use_cycle'], n_cond=cfg['n_cond'],
              legacy_ms_pad=cfg['legacy_ms_pad'], legacy_ema=cfg['legacy_ema'])
    return DSPTWithLife(**kw) if use_life else build_dspt(**kw)


def engine_life(frame):
    """True total lifetime per engine (cycles) — the oracle quantity."""
    return frame.groupby('unit')['cycle'].max().to_dict()


def train_units_vector(frame):
    """Unit id per training window.

    ``make_train_windows`` appends exactly ``len(g)`` windows per unit while
    iterating ``frame.groupby('unit', sort=True)``, so the unit sequence can be
    rebuilt from the frame without touching the frozen data module.
    """
    us = []
    for uid, g in frame.groupby('unit', sort=True):
        us.extend([int(uid)] * len(g))
    return np.asarray(us, np.int64)


def life_tensor(units, life_map, eps, mu, sd, dev):
    """L*(1+eps) standardised with training-fold statistics."""
    raw = np.array([life_map[int(u)] for u in units], np.float64)
    noisy = raw * (1.0 + eps)
    z = (noisy - mu) / sd
    return torch.from_numpy(z.astype(np.float32)).to(dev)


def run(subset, seed, sigma, device):
    use_life = sigma is not None
    tag = tag_of(sigma)
    run_dir = OUT / subset / f's{seed}' / tag
    if (run_dir / 'metrics.json').exists():
        return 'skip'
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(BASE_CFG, subset=subset, seed=seed)
    set_seed(seed)
    t0 = time.time()

    ds = prepare_train(subset, DATA, span=cfg['span'], split_seed=cfg['split_seed'],
                       fold=cfg['fold'], val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=cfg['cuts'])
    tr, va = ds['train'], ds['val']
    dev = torch.device(device)
    life_map = engine_life(ds['raw'])

    # one draw per engine, shared by all of that engine's windows
    rng = np.random.default_rng(seed * 100000 + int(round((sigma or 0) * 1000)) + 7)
    units = sorted(life_map)
    eps_map = {u: (float(rng.normal(0.0, sigma)) if use_life else 0.0) for u in units}
    lens = np.array([life_map[u] for u in units], np.float64)
    mu = float(lens.mean()) if use_life else 0.0
    sd = float(lens.std(ddof=1)) if use_life else 1.0

    def tensors(d, units=None):
        out = {k: torch.from_numpy(np.ascontiguousarray(d[k])).to(dev)
               for k in ('x', 'mask', 'baseline', 'cycle', 'cond')}
        if use_life:
            u = d['unit'] if 'unit' in d else units
            if u is None:
                raise KeyError('no unit ids for this split')
            eps = np.array([eps_map[int(x)] for x in u], np.float64)
            out['life'] = life_tensor(u, life_map, eps, mu, sd, dev)
        return out

    tr_units = train_units_vector(ds['tr'])
    TR, VA = tensors(tr, tr_units), tensors(va)
    HT = torch.from_numpy(tr['hi']).to(dev)
    net = build_net(cfg, use_life).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-5)
    g = torch.Generator(device=dev).manual_seed(seed)

    def forward(batch, idx, training):
        kw = dict(mask=batch['mask'][idx], baseline=batch['baseline'][idx],
                  cycle=(batch['cycle'][idx] if net.use_cycle else None),
                  cond=batch['cond'][idx], mode=cfg['mode'])
        if use_life:
            kw['life'] = batch['life'][idx]
        return net(batch['x'][idx], **kw)

    def val_predict():
        net.eval()
        with torch.no_grad():
            out = []
            n = len(VA['x'])
            for i in range(0, n, 512):
                sl = slice(i, i + 512)
                o = forward(VA, sl, False)
                out.append(o['prediction'].reshape(-1).float().cpu().numpy())
        p = np.concatenate(out)
        return p, per_engine_rmse(va['unit'], va['y'], p)

    best, best_state, best_ep, hist = float('inf'), None, -1, []
    for ep in range(1, cfg['epochs'] + 1):
        net.train()
        perm = torch.randperm(len(TR['x']), generator=g, device=dev)
        tot, nb = 0.0, 0
        for idx in perm.split(cfg['batch']):
            opt.zero_grad(set_to_none=True)
            out = forward(TR, idx, True)
            y = torch.from_numpy(tr['y']).to(dev)[idx].reshape(-1, 1)
            e = out['prediction'] - y
            loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()
            if cfg['aux_dual']:
                eA, eB = out['y_A'] - y, out['y_B'] - y
                loss = loss + 0.25 * (eA.square().mean() + eB.square().mean())
            ht = HT[idx].reshape(-1)
            loss = loss + cfg['hi_lambda'] * torch.mean((out['hi'].reshape(-1) - ht) ** 2)
            hs = out['hi_seq']
            step = (TR['mask'][idx][:, 1:] * TR['mask'][idx][:, :-1])
            diff = hs[:, 1:] - hs[:, :-1]
            pen = torch.relu(diff) if cfg['hi_mono'] == 'correct' else torch.relu(-diff)
            loss = loss + cfg['hi_mono_coef'] * (pen * step).sum() / step.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        p, vr = val_predict()
        hist.append(dict(epoch=ep, loss=tot / max(nb, 1), val_engine=vr))
        if vr < best - 1e-4:
            best, best_ep = vr, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= cfg['patience']:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    p, vr = val_predict()
    nasa_sum, nasa_mean = nasa_score(p, va['y'])
    metrics = dict(subset=subset, seed=seed, sigma=sigma, tag=tag,
                   val_engine_mean_rmse=float(vr), val_pooled_rmse=float(rmse(p, va['y'])),
                   nasa_sum=float(nasa_sum), nasa_mean=float(nasa_mean),
                   bias=float(np.mean(p - va['y'])),
                   best_ep=int(best_ep), epochs_run=len(hist), params=int(nparam),
                   elapsed_s=round(time.time() - t0, 1),
                   n_train_windows=int(len(tr['x'])), n_val_windows=int(len(va['x'])),
                   life_mu=mu, life_sd=sd,
                   eps_by_unit={int(k): v for k, v in eps_map.items()} if use_life else {})
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    import pandas as pd
    pd.DataFrame(dict(unit=va['unit'], cut=va['cut'], y_true=va['y'], y_pred=p)).to_csv(
        run_dir / 'val_pred.csv', index=False)
    print(f'{subset} s{seed} {tag}: val_engine {vr:.4f} ep{best_ep}/{len(hist)} '
          f'{metrics["elapsed_s"]}s', flush=True)
    return 'done'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--sigma', type=float)
    ap.add_argument('--grid', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    if a.grid:
        todo = [(s, sd, sg) for s in SUBSETS for sd in SEEDS for sg in SIGMAS]
        print(f'grid: {len(todo)} runs', flush=True)
        for i, (s, sd, sg) in enumerate(todo, 1):
            print(f'[{i}/{len(todo)}]', end=' ', flush=True)
            try:
                run(s, sd, sg, a.device)
            except Exception as e:                       # keep the queue alive
                print(f'FAILED {s} s{sd} {tag_of(sg)}: {type(e).__name__}: {e}', flush=True)
        return
    if a.subset is None or a.seed is None:
        ap.error('need --subset and --seed, or --grid')
    run(a.subset, a.seed, a.sigma, a.device)


if __name__ == '__main__':
    main()
