#!/usr/bin/env python3
"""Estimate the noise scale from a given partition's TRAINING engines only.

Why this exists
---------------
`shared/noise_stats_<subset>.json` was produced by `build`, which is pinned to
`BASE['split_seed']=17`. The pre-registered rule says the noise scale is
estimated from the training engines only, so under split_seed=29 that means the
29 training engines -- reusing the 17 file would put 88-96% of the 29
validation engines inside the estimate, because a different split moves most
engines from train to val.

`noise_alpha` (0.25) stays frozen; only the per-partition scale is re-derived,
which is what the rule specifies. The 17 files are left untouched, and the new
file is passed to the runner through `cfg['augment']['noise_stats_path']`, so
neither the runner nor the h8 fingerprint changes.

Usage
-----
    python scripts/make_noise_stats_split.py --data-dir <dir> --split-seed 29
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_b.sensor_aug.noise_stats import (  # noqa: E402
    compute_noise_stats, effective_scale, save_stats,
)
from paper_b.training.data import _prepare_split  # noqa: E402
from scripts.run_sensor_aug import BASE, SHARED  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--split-seed', type=int, default=29)
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    for subset in args.subsets.split(','):
        s = _prepare_split(subset, data_dir, args.split_seed, BASE['fold'],
                           BASE['val_frac'], BASE['n_cond'])
        tr = set(int(x) for x in s['train_units'])
        va = set(int(x) for x in s['val_units'])
        if tr & va:
            raise SystemExit(f'{subset}: train/val overlap {sorted(tr & va)[:5]}')
        stats = compute_noise_stats(s['tr'], s['tr_s'], s['lt'], n_cond=BASE['n_cond'])
        stats.update(subset=subset, split_seed=args.split_seed, fold=BASE['fold'],
                     val_frac=BASE['val_frac'],
                     train_units=[int(x) for x in s['train_units']],
                     val_units=[int(x) for x in s['val_units']])
        stats['alpha_scales'] = {f'{a:g}': effective_scale(stats, a) for a in (0.25, 0.50)}
        dst = SHARED / f'noise_stats_{subset}_split{args.split_seed}.json'
        save_stats(stats, dst)
        old = json.loads((SHARED / f'noise_stats_{subset}.json').read_text())
        print(f'{subset}: split{args.split_seed} n_train={len(tr)} n_val={len(va)} '
              f'overlap_train_val={len(tr & va)} | new_hash={stats["hash"]} '
              f'(17 version {old["hash"]}) cap25={stats["alpha_scales"]["0.25"]["cap_fraction"]:.3f}')
    print(f'[done] wrote noise_stats_*_split{args.split_seed}.json into {SHARED}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
