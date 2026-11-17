#!/usr/bin/env python3
"""Regenerate the shared evaluation-only missing-sensor patterns for ONE split.

Why this exists
---------------
`run_sensor_aug.py:390` reads a single global
`shared/missing_patterns_val_<subset>.json`, and `build()` writes it from
*that build's* `val_units` (`:241`). So the file belongs to whichever partition
last ran `build()`. A batch that mixes two partitions in one manifest cannot
work: `evaluate.py:88` does `patterns[str(int(u))]` and raises
`KeyError(<unit>)` for every validation engine that belongs to the other split.

That is exactly what stopped the 132-cell batch at 86/132 with `KeyError('8')`:
the file on disk was split-17, while the split-29 cells' validation engines
(unit 8 among them) are not keys in it.

The fix that does NOT change h8: regenerate the file per partition and run the
partitions **sequentially**, never concurrently. This script produces the file
for a given `--split` and prints its hash so the run log records which pattern
set was live.

Usage
-----
    python scripts/make_eval_patterns_split.py --root <root> --data-dir <CMAPSS> \
        --split 29 [--dry-run] [--out <path>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from paper_b.sensor_aug.evaluation import make_missing_patterns, patterns_hash  # noqa: E402

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')


def split_units(units, split_seed, fold=0, val_frac=0.2):
    """Identical to paper_b/training/data.py:split_units."""
    units = np.sort(np.asarray(units))
    rng = np.random.default_rng(split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size * val_frac)))
    return sorted(int(x) for x in sh[fold * n_val:(fold + 1) * n_val])


def lifetimes(path: Path) -> dict[int, int]:
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 2:
                u, c = int(float(p[0])), int(float(p[1]))
                out[u] = max(out.get(u, 0), c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', default=None, help='override output path (staging)')
    a = ap.parse_args()
    root, data = Path(a.root), Path(a.data_dir)

    for subset in SUBSETS:
        life = lifetimes(data / f'train_{subset}.txt')
        val = split_units(sorted(life), a.split, a.fold)
        pat = make_missing_patterns(val)
        payload = dict(subset=subset, split_seed=a.split, fold=a.fold,
                       n_patterns=10, hash=patterns_hash(pat), patterns=pat,
                       written_by='scripts/make_eval_patterns_split.py')
        dest = Path(a.out) if a.out else root / 'shared' / f'missing_patterns_val_{subset}.json'
        if not a.dry_run:
            dest.write_text(json.dumps(payload, indent=1), encoding='utf-8')
        blob = json.dumps(payload, indent=1).encode()
        print(f'{subset}: split={a.split} val_units={len(val)} '
              f'patterns_hash={payload["hash"]} file_sha256='
              f'{hashlib.sha256(blob).hexdigest()[:16]} '
              f'{"(dry-run)" if a.dry_run else "-> " + str(dest)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
