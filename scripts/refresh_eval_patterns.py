#!/usr/bin/env python3
"""Rebuild the evaluation missing-pattern bank for the confirmation partition.

Why this is needed
------------------
`shared/missing_patterns_val_<subset>.json` is keyed by engine id:
`patterns[str(unit)][pattern_id][:k]`. It is produced by
`make_missing_patterns(s['val_units'])` for whatever partition `build` used,
and `build` is pinned to `BASE['split_seed']=17`. Under split_seed=29 the
validation engines are a different set, so `evaluation.missing_keep` raises
`KeyError: '8'` on the first engine it cannot find.

This is an **evaluation-only** artifact: the training-time mask draws random
channels per sample from `mask_prob` and never reads this bank. Rebuilding it
on the confirmation partition therefore does not change the arm; it only makes
the robustness evaluation well-defined on that partition.

What is deliberately NOT rebuilt
--------------------------------
`shared/noise_stats_<subset>.json` stays on split_seed=17. Its scale is
injected during **training**, so it is part of the frozen arm definition;
re-estimating it on the confirmation partition would change the arm rather
than confirm it. `sigma` is a per-sensor vector, not keyed by engine id, so it
does not raise on the new partition.

Usage
-----
    python scripts/refresh_eval_patterns.py --data-dir <dir> --split-seed 29
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_b.sensor_aug.evaluation import make_missing_patterns, patterns_hash  # noqa: E402
from paper_b.training.data import _prepare_split  # noqa: E402
from scripts.run_sensor_aug import BASE, SHARED  # noqa: E402

PATTERN_SEED = 20261114
N_PATTERNS = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--split-seed', type=int, default=29)
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    ap.add_argument('--apply', action='store_true', help='write; otherwise dry run')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    archive = SHARED / f'archive_split{BASE["split_seed"]}'
    archive.mkdir(parents=True, exist_ok=True)

    for subset in args.subsets.split(','):
        s = _prepare_split(subset, data_dir, args.split_seed, BASE['fold'],
                           BASE['val_frac'], BASE['n_cond'])
        pat = make_missing_patterns(s['val_units'])
        h = patterns_hash(pat)
        dst = SHARED / f'missing_patterns_val_{subset}.json'
        old = json.loads(dst.read_text()) if dst.exists() else None
        old_units = sorted(old['patterns'], key=lambda x: int(x)) if old else []
        new_units = sorted(pat, key=lambda x: int(x))
        print(f'{subset}: val_units(split {args.split_seed})={len(new_units)} '
              f'old_hash={old["hash"] if old else None} new_hash={h} '
              f'units_changed={sum(1 for u in new_units if u not in old_units)}/{len(new_units)}')
        if not args.apply:
            continue
        if old is not None:
            shutil.copy2(dst, archive / dst.name)
        dst.write_text(json.dumps({'subset': subset, 'split': 'val',
                                   'split_seed': args.split_seed,
                                   'seed': PATTERN_SEED, 'n_patterns': N_PATTERNS,
                                   'hash': h, 'patterns': pat}, indent=1))
    if args.apply:
        print(f'[apply] wrote {args.subsets} at split_seed={args.split_seed}; '
              f'split-{BASE["split_seed"]} copies archived to {archive}')
    else:
        print('[dry-run] nothing written; pass --apply')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
