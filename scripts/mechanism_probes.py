"""No-training mechanism probes for the sensor-dropout arms.

Uses only existing checkpoints and forward passes. Three probes:

  P1  single-sensor ablation: mask one of the 14 sensors at test time and
      measure the rise in engine-level mean-of-roots RMSE. A model that has
      learned to use redundant sensors should degrade LESS.
  P2  cross-sensor pooling attention entropy: entropy of the softmax over the
      14 sensors that the model uses to pool sensor tokens (``sensor_score``).
      Higher entropy = more spread out = less reliance on individual sensors.
  P3  seed-to-seed prediction variance: the three seeds share the same
      validation engines and windows, so the variance of their predictions on
      the same window measures run-to-run instability.

Usage:
  python3 scripts/mechanism_probes.py <runs_root> --data-dir /path/CMAPSS \
      --split-seed 17 --out /tmp/mech.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from _cells import resolve                                    # noqa: E402
from paper_b.training.data import prepare_train                # noqa: E402
from paper_b.training.train import _build, predict             # noqa: E402

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)
ARMS = ('E0', 'E1a', 'E1b')
BASE = dict(
    model='dspt', span=70, epochs=40, patience=12, batch=512, lr=1e-3,
    weight_decay=1e-4, d_model=64, layers=2, dropout=0.2, mode='dual',
    ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct',
    hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=True, use_pad_mask=True,
    input_mode='raw', aux_dual=True, instnorm=False, rul_cap=125, fp32=True,
    split_seed=17, fold=0, val_frac=0.2, n_cond=6,
    cuts=[0.4, 0.55, 0.7, 0.85, 0.95],
    legacy_rul=False, global_hi=False, legacy_ms_pad=False, legacy_ema=False,
)


def mean_roots(unit, y, pred):
    acc = defaultdict(list)
    for u, t, p in zip(unit, y, pred):
        acc[int(u)].append((float(t), float(p)))
    per = [math.sqrt(sum((p - t) ** 2 for t, p in v) / len(v)) for v in acc.values()]
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--split-seed', type=int, default=17)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = {}
    res = dict(p1={}, p2={}, p3={}, baseline={})
    preds = defaultdict(dict)

    for subset in SUBSETS:
        ds = prepare_train(subset, Path(a.data_dir), span=BASE['span'],
                           split_seed=a.split_seed, fold=0, val_frac=0.2,
                           rul_cap=BASE['rul_cap'], n_cond=BASE['n_cond'],
                           cuts=tuple(BASE['cuts']), legacy_rul=False, global_hi=False)
        va = ds['val']
        n = len(va['y'])
        for arm in ARMS:
            for seed in SEEDS:
                cell = resolve(a.root, arm, subset, seed, a.split_seed)
                cfg = {**BASE, 'subset': subset, 'seed': seed, 'data_dir': str(a.data_dir)}
                net = _build(cfg).to(dev)
                net.load_state_dict(torch.load(Path(cell) / 'best.pt', map_location='cpu',
                                               weights_only=True))
                net.eval()

                # P2: capture the pooling logits once on the clean pass
                cap = {}
                h = net.sensor_score.register_forward_hook(
                    lambda m, i, o: cap.__setitem__('logits', o.detach()))
                with torch.no_grad():
                    base = predict(net, va, dev, cfg['mode'], fp32=True)
                h.remove()
                mr, _ = mean_roots(va['unit'], va['y'], base)
                res['baseline'][f'{arm}/{subset}/s{seed}'] = round(mr, 4)
                preds[(arm, subset)][seed] = base

                logits = cap['logits'].squeeze(-1)          # (B,L,N,1) -> (B,L,N)
                p = torch.softmax(logits.float(), dim=-1)
                ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean().item()
                res['p2'][f'{arm}/{subset}/s{seed}'] = round(float(ent), 4)

                deltas = []
                for j in range(14):
                    keep = np.ones((n, 14), dtype=bool)
                    keep[:, j] = False
                    with torch.no_grad():
                        pr = predict(net, va, dev, cfg['mode'], fp32=True, sensor_keep=keep)
                    mj, _ = mean_roots(va['unit'], va['y'], pr)
                    deltas.append(round(mj - mr, 4))
                res['p1'][f'{arm}/{subset}/s{seed}'] = dict(
                    mean_delta=round(float(np.mean(deltas)), 4),
                    max_delta=round(float(np.max(deltas)), 4),
                    argmax_sensor=int(np.argmax(deltas)),
                    deltas=deltas)
                print(f'{arm} {subset} s{seed}  base={mr:.4f}  '
                      f'p1_mean={np.mean(deltas):+.4f} p1_max={np.max(deltas):+.4f}  '
                      f'ent={ent:.4f}', flush=True)
                del net
                torch.cuda.empty_cache()

    # P3: cross-seed variance on the same windows
    for (arm, subset), d in preds.items():
        mats = [d[s] for s in SEEDS if s in d]
        if len(mats) < 2:
            continue
        M = np.stack(mats, axis=0)                          # (seeds, n)
        res['p3'][f'{arm}/{subset}'] = dict(
            mean_var=round(float(np.var(M, axis=0).mean()), 4),
            mean_sd=round(float(np.std(M, axis=0).mean()), 4))

    txt = json.dumps(res, indent=1)
    if a.out:
        open(a.out, 'w').write(txt)

    # compact summary
    for probe in ('baseline', 'p1', 'p2', 'p3'):
        print('\n==', probe)
        for arm in ARMS:
            vals = {k: v for k, v in res[probe].items() if k.startswith(arm + '/')}
            if not vals:
                continue
            if probe == 'p1':
                m = np.mean([v['mean_delta'] for v in vals.values()])
                mx = np.mean([v['max_delta'] for v in vals.values()])
                print(f'  {arm}: mean|Δ| per single-sensor ablation = {m:+.4f} '
                      f'(worst sensor {mx:+.4f})')
            elif probe == 'p3':
                mv = np.mean([v['mean_var'] for v in vals.values()])
                print(f'  {arm}: mean cross-seed prediction variance = {mv:.4f}')
            else:
                print(f'  {arm}: {np.mean(list(vals.values())):.4f}  (n={len(vals)})')


if __name__ == '__main__':
    main()
