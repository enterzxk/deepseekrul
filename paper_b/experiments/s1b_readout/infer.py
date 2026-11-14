"""S1-B inference: rebuild the frozen F_full_basic_hi1 predictions.

For every (subset, seed) this reproduces the stored evaluation points and adds
the shifted windows the multi-window readout needs:

  * validation: the five frozen cuts round(f*n), plus windows ending j cycles
    earlier for j = 1..J-1 (target RUL adjusted by +j).
  * official test: the final window, plus the same j shifts.

Nothing here trains or selects anything. The readout choice is made later on
validation only.

Usage:
  python infer.py --out preds.csv [--subsets FD001] [--seeds 101] [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Paths are overridable so the script runs from any checkout:
#   S1B_REPO     repo root containing paper_b/ and data/CMAPSS
#   S1B_WEIGHTS  directory with <subset>/s<seed>/model.pt (+ config.json)
#   S1B_DATA     C-MAPSS txt files (defaults to <repo>/data/CMAPSS)
import os
REPO = Path(os.environ.get('S1B_REPO', '/home/user/deepseek-rul'))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

from paper_b.training.data import (CYCLE_SCALE, _baseline_ref, _prepare_split,
                                   _window, prepare_test)
from paper_b.training.train import _build, predict

WEIGHTS = Path(os.environ.get('S1B_WEIGHTS', '/home/user/s1b/weights'))
DATA = Path(os.environ.get('S1B_DATA', str(REPO / 'data' / 'CMAPSS')))
PROTO = json.loads((REPO / 'paper_b/configs/protocol.json').read_text())
CUTS = tuple(PROTO['val_cuts'])
CAP = float(PROTO['rul_cap'])
SPAN = int(PROTO['span'])
J_MAX = 10


def shifted_val_windows(subset, cfg, j_max=J_MAX):
    """Frozen validation cuts plus j-cycle-earlier windows."""
    s = _prepare_split(subset, DATA, cfg['split_seed'], cfg['fold'], cfg['val_frac'], cfg['n_cond'])
    frame, values, clusters, pop = s['va'], s['va_s'], s['lv'], s['pop']
    rows = []
    for uid, g in frame.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], np.float32)
        cl = np.asarray(clusters[pos], np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        n = len(seq)
        for ci, f in enumerate(CUTS):
            c = int(min(n, max(1, round(f * n))))
            for j in range(j_max):
                t = (c - 1) - j
                if t < 0:
                    continue
                x, mask = _window(seq, t + 1, SPAN)
                rows.append(dict(
                    unit=int(uid), cut_idx=ci, cut_cycle=int(c), j=j,
                    y_true_raw=float(min(max(n - (t + 1), 0), CAP)),
                    x=x, mask=mask,
                    baseline=_baseline_ref(seq, cl, cyc, t, pop),
                    cycle=float(min(cyc[t] / CYCLE_SCALE, 1.0)),
                    cond=int(cl[t]),
                ))
    return rows


def shifted_test_windows(subset, cfg, j_max=J_MAX):
    """Official test final window plus j-cycle-earlier windows."""
    ds = prepare_test(subset, DATA, span=SPAN, split_seed=cfg['split_seed'], fold=cfg['fold'],
                      val_frac=cfg['val_frac'], rul_cap=CAP, n_cond=cfg['n_cond'])
    s = _prepare_split(subset, DATA, cfg['split_seed'], cfg['fold'], cfg['val_frac'], cfg['n_cond'])
    from paper_b.training.data import load_test
    test, rul_map = load_test(DATA, subset)
    test = test.sort_values(['unit', 'cycle']).reset_index(drop=True)
    ts = s['scaler'].transform(test)
    lt = s['scaler'].labels(test)
    rows = []
    for uid, g in test.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(ts[pos], np.float32)
        cl = np.asarray(lt[pos], np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        L = len(seq)
        base_rul = float(rul_map[int(uid)])
        for j in range(j_max):
            t = (L - 1) - j
            if t < 0:
                continue
            x, mask = _window(seq, t + 1, SPAN)
            rows.append(dict(
                unit=int(uid), cut_idx=0, cut_cycle=int(cyc[L - 1]), j=j,
                y_true_raw=float(min(base_rul + j, CAP)),
                x=x, mask=mask,
                baseline=_baseline_ref(seq, cl, cyc, t, s['pop']),
                cycle=float(min(cyc[t] / CYCLE_SCALE, 1.0)),
                cond=int(cl[t]),
            ))
    return rows


def predict_fp32(net, d, dev, mode='dual', bs=512):
    """Same forward as train.predict but without the BF16 autocast.

    The stored predictions were produced under autocast(bfloat16); that path is
    not bit-reproducible across processes, so re-inference for the readout runs
    in FP32 (the precision the current protocol mandates).
    """
    net.eval()
    n = len(d['y'])
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            o = net(torch.from_numpy(d['x'][sl]).to(dev),
                    mask=torch.from_numpy(d['mask'][sl]).to(dev),
                    baseline=torch.from_numpy(d['baseline'][sl]).to(dev),
                    cycle=(torch.from_numpy(d['cycle'][sl]).to(dev) if net.use_cycle else None),
                    cond=torch.from_numpy(d['cond'][sl]).to(dev), mode=mode)
            out.append(o['prediction'].reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def run_one(subset, seed, device, j_max=J_MAX, fp32=False):
    run_dir = WEIGHTS / subset / f's{seed}'
    cfg = json.loads((run_dir / 'config.json').read_text())
    net = _build(cfg).to(device)
    net.load_state_dict(torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=True))
    net.eval()
    out = []
    for split, builder in (('val', shifted_val_windows), ('test', shifted_test_windows)):
        rows = builder(subset, cfg, j_max)
        if not rows:
            continue
        # One forward pass per shift. Mixing shifts into one batch changes the
        # float reduction order and moved 1/100 test rows by 0.5 against the
        # stored file; per-shift batching reproduces j=0 bit-for-bit.
        for j in sorted({r['j'] for r in rows}):
            sub = [r for r in rows if r['j'] == j]
            d = {k: np.stack([r[k] for r in sub]) for k in
                 ('x', 'mask', 'baseline', 'cycle', 'cond')}
            d['x'] = d['x'].astype(np.float32)
            d['baseline'] = d['baseline'].astype(np.float32)
            d['cycle'] = d['cycle'].astype(np.float32)
            d['cond'] = d['cond'].astype(np.int64)
            d['mask'] = d['mask'].astype(bool) if d['mask'].dtype != bool else d['mask']
            d['y'] = np.asarray([r['y_true_raw'] for r in sub], np.float32)
            pred = (predict_fp32 if fp32 else predict)(net, d, device, mode=cfg['mode'])
            df = pd.DataFrame([{k: r[k] for k in ('unit', 'cut_idx', 'cut_cycle', 'j', 'y_true_raw')}
                               for r in sub])
            df['y_pred'] = pred
            df['subset'] = subset
            df['seed'] = seed
            df['split'] = split
            out.append(df)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--subsets', nargs='*', default=PROTO['subsets'])
    ap.add_argument('--seeds', nargs='*', type=int, default=PROTO['seeds'])
    ap.add_argument('--j-max', type=int, default=J_MAX)
    ap.add_argument('--check', action='store_true', help='compare j=0 against stored predictions')
    ap.add_argument('--fp32', action='store_true', help='run without BF16 autocast')
    a = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    frames = []
    for subset in a.subsets:
        for seed in a.seeds:
            df = run_one(subset, seed, device, a.j_max, a.fp32)
            frames.append(df)
            print(f'{subset} s{seed}: {len(df)} rows', flush=True)
    allp = pd.concat(frames, ignore_index=True)
    allp.to_csv(a.out, index=False)
    print('wrote', a.out, len(allp))

    if a.check:
        print('\n== reproduction check against stored val_pred / test_pred ==')
        worst = 0.0
        for subset in a.subsets:
            for seed in a.seeds:
                rd = WEIGHTS / subset / f's{seed}'
                mine = allp[(allp.subset == subset) & (allp.seed == seed)]
                v = pd.read_csv(rd / 'val_pred.csv')
                m0 = mine[(mine.split == 'val') & (mine.j == 0)].sort_values(['unit', 'cut_cycle'])
                ref = v.sort_values(['unit', 'cut'])
                if len(m0) == len(ref):
                    d = float(np.max(np.abs(m0.y_pred.to_numpy() - ref.y_pred.to_numpy())))
                    worst = max(worst, d)
                    print(f'  {subset} s{seed} val : n={len(m0)} max|dpred|={d:.3e}')
                else:
                    print(f'  {subset} s{seed} val : LENGTH MISMATCH {len(m0)} vs {len(ref)}')
                t = pd.read_csv(rd / 'test_pred.csv')
                m1 = mine[(mine.split == 'test') & (mine.j == 0)].sort_values('unit')
                if len(m1) == len(t):
                    d = float(np.max(np.abs(m1.y_pred.to_numpy() - t.sort_values("unit").y_pred.to_numpy())))
                    worst = max(worst, d)
                    print(f'  {subset} s{seed} test: n={len(m1)} max|dpred|={d:.3e}')
                else:
                    print(f'  {subset} s{seed} test: LENGTH MISMATCH {len(m1)} vs {len(t)}')
        print(f'WORST |dpred| over all = {worst:.3e}')


if __name__ == '__main__':
    main()

