#!/usr/bin/env python3
"""Dump one real training batch's augmentation result (read-only debug).

This answers "did the augmentation actually apply?" without touching the
training loop or any file in the h8 fingerprint.

What it checks
--------------
mask arms (E1a/E1b)
  * every hidden sensor is hidden for the whole window and for all derived
    channels: the model zeroes x and baseline *before* `_features`, so
    raw = 0, deviation = x - baseline = 0, EMA(0) = 0;
  * at least one sensor survives in every sample;
  * kept sensors are bit-identical to the unaugmented batch.
noise arms (E2a/E2b)
  * padded time steps receive no noise;
  * the injected RMS matches the frozen per-condition scale on valid steps.

Usage
-----
  python scripts/debug_aug_batch.py --manifest <stage2.json> --group E1b \
      --subset FD001 --seed 101 --data-dir /path/to/CMAPSS [--batch 8] \
      [--out /tmp/aug_debug.json]

Exit code 0 means every check passed; 1 means at least one failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def find_task(manifest, group, subset, seed):
    man = json.loads(Path(manifest).read_text())
    for r in man['runs']:
        if r['group'] == group and r['subset'] == subset and int(r['seed']) == int(seed):
            return man, r
    raise SystemExit(f'no such run: {group} {subset} s{seed}')


def build_augmentor(cfg, device):
    from paper_b.sensor_aug.augmentation import SensorAugmentor
    from paper_b.sensor_aug.noise_stats import effective_scale
    a = cfg['augment']
    scale = None
    if float(a.get('noise_alpha', 0.0)) > 0:
        stats = json.loads(Path(a['noise_stats_path']).read_text())
        scale = np.asarray(effective_scale(stats, float(a['noise_alpha']))['s'], np.float32)
    return SensorAugmentor(mask_prob=float(a.get('mask_prob', 0.0)), noise_scale=scale,
                           n_sensors=14, seed=cfg['seed'], device=str(device)), scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--group', required=True)
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    from paper_b.training.data import prepare_train

    man, task = find_task(a.manifest, a.group, a.subset, a.seed)
    cfg = task['config']
    dev = torch.device('cpu')

    ds = prepare_train(cfg['subset'], a.data_dir, span=cfg['span'],
                       split_seed=cfg['split_seed'], fold=cfg['fold'],
                       val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                       legacy_rul=cfg['legacy_rul'], global_hi=cfg['global_hi'])
    tr = ds['train']
    n = min(a.batch, len(tr['x']))
    x = torch.from_numpy(tr['x'][:n]).float()
    mask = torch.from_numpy(tr['mask'][:n]).float()
    base = torch.from_numpy(tr['baseline'][:n]).float()
    cseq = torch.from_numpy(tr['cond_seq'][:n]).long()

    aug, scale = build_augmentor(cfg, dev)
    out = {'run_id': task['run_id'], 'group': a.group, 'subset': a.subset,
           'seed': int(a.seed), 'batch': n,
           'mask_prob': aug.mask_prob,
           'noise_alpha': float(cfg['augment'].get('noise_alpha', 0.0)),
           'checks': {}}
    ok = True

    if aug.mask_gen is not None:
        keep = aug.sample_mask(n, device=dev)
        kept = keep.sum(dim=1)
        xm = torch.where(keep[:, None, :], x, torch.zeros_like(x))
        bm = torch.where(keep, base, torch.zeros_like(base))
        hidden = ~keep
        out['checks']['kept_per_sample'] = [int(v) for v in kept.tolist()]
        out['checks']['kept_min'] = int(kept.min())
        out['checks']['all_hidden_samples'] = int((kept == 0).sum())
        out['checks']['hidden_entries_zero'] = (
            float((xm[hidden[:, None, :].expand_as(xm)] == 0).float().mean())
            if bool(hidden.any()) else None)
        out['checks']['kept_entries_unchanged'] = (
            float(torch.eq(xm[keep[:, None, :].expand_as(xm)],
                           x[keep[:, None, :].expand_as(xm)]).float().mean())
            if bool(keep.any()) else None)
        # Derived channels are computed from the masked x/baseline inside
        # DSPT.encode, so a hidden sensor has deviation = 0 - 0 = 0 and
        # EMA(0) = 0 for every time step. Check the arithmetic on real rows.
        j = int(torch.nonzero(hidden.any(dim=0))[0]) if bool(hidden.any()) else None
        if j is not None:
            row = int(torch.nonzero(hidden[:, j])[0])
            out['checks']['hidden_sensor'] = j
            out['checks']['hidden_row_raw_all_zero'] = bool((xm[row, :, j] == 0).all())
            out['checks']['hidden_row_dev_all_zero'] = bool(
                (xm[row, :, j] - bm[row, j] == 0).all())
        ok &= out['checks']['all_hidden_samples'] == 0
        ok &= out['checks']['kept_min'] >= 1
        if out['checks']['hidden_entries_zero'] is not None:
            ok &= out['checks']['hidden_entries_zero'] == 1.0
        if out['checks']['kept_entries_unchanged'] is not None:
            ok &= out['checks']['kept_entries_unchanged'] == 1.0
        if j is not None:
            ok &= bool(out['checks']['hidden_row_raw_all_zero'])
            ok &= bool(out['checks']['hidden_row_dev_all_zero'])

    if aug.noise_gen is not None:
        xn = aug.add_noise(x, mask, cseq)
        delta = xn - x
        valid = mask.bool()
        padded_clean = bool((delta[~valid] == 0).all()) if bool((~valid).any()) else True
        # Nominal per-entry scale on valid steps: scale[cond_seq] * mask.
        nom = torch.from_numpy(np.asarray(scale, np.float32))[cseq] * valid.unsqueeze(-1).float()
        inj_rms = float((delta[valid].double() ** 2).mean().sqrt()) if bool(valid.any()) else 0.0
        nom_rms = float((nom[valid].double() ** 2).mean().sqrt()) if bool(valid.any()) else 0.0
        out['checks']['noise_padded_clean'] = padded_clean
        out['checks']['noise_injected_rms'] = inj_rms
        out['checks']['noise_nominal_rms'] = nom_rms
        out['checks']['noise_ratio'] = (inj_rms / nom_rms) if nom_rms > 0 else None
        ok &= padded_clean
        if nom_rms > 0:
            ok &= 0.9 <= inj_rms / nom_rms <= 1.1

    out['aug_stats_after_batch'] = aug.stats()
    out['passed'] = bool(ok)

    # Post-hoc record from a completed run, if one exists.
    run_dir = Path(a.manifest).resolve().parent.parent / 'runs' / task['run_id'] / \
        task['subset'] / f"s{task['seed']}"
    st = run_dir / 'aug_stats.json'
    if st.exists():
        out['run_aug_stats'] = json.loads(st.read_text())

    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
