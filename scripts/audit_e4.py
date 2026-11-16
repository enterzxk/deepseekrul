"""E4 identity arm vs E0: paired engine-level recompute.

E4 replaces the sensor-attention block with the existing ``sa_mode='identity'``
switch (an existing code path, not a contribution). This script only measures
it as a control arm.

Metric: mean_u RMSE_u (primary). Delta = E4 - E0, paired per engine inside
(subset, seed); subsets merged with equal weight; 95% CI from an engine-level
bootstrap (>=10000 draws). Also reports the seed-level sd and the merged Delta
without FD001, matching the tables published for E1/E2.

Usage:
  python3 scripts/audit_e4.py <runs_root> [--split-seed 17] [--boot 10000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cells import resolve  # noqa: E402

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)


def roots(path):
    acc = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            acc[int(r.get('unit') or r.get('engine_id'))].append(
                (float(r['y_true']), float(r['y_pred'])))
    return {u: math.sqrt(sum((p - t) ** 2 for t, p in v) / len(v)) for u, v in acc.items()}


BOOT_SEED = 20261116


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--split-seed', type=int, default=17)
    ap.add_argument('--boot', type=int, default=10000)
    a = ap.parse_args()

    d_by_subset, per_cell, signs = {}, {}, {}
    for s in SUBSETS:
        ds = []
        for sd in SEEDS:
            ca = Path(resolve(a.root, 'E4', s, sd, a.split_seed))
            c0 = Path(resolve(a.root, 'E0', s, sd, a.split_seed))
            ra = roots(ca / 'val_pred.csv')
            r0 = roots(c0 / 'val_pred.csv')
            eng = sorted(set(ra) & set(r0))
            d = np.array([ra[u] - r0[u] for u in eng])
            ds.append(d)
            per_cell[f'{s}/s{sd}'] = dict(delta=round(float(d.mean()), 4),
                                          n_engines=len(eng),
                                          e4=round(float(np.mean([ra[u] for u in eng])), 4),
                                          e0=round(float(np.mean([r0[u] for u in eng])), 4))
            signs.setdefault(s, []).append(int((d < 0).sum()))
        d_by_subset[s] = np.mean(np.stack(ds, axis=0), axis=0)

    keys = list(d_by_subset)
    merged = float(np.mean([d_by_subset[k].mean() for k in keys]))
    rng = np.random.default_rng(BOOT_SEED)   # own stream: main CI
    draws = np.empty(a.boot)
    for i in range(a.boot):
        draws[i] = float(np.mean([d_by_subset[k][rng.integers(0, len(d_by_subset[k]),
                                                               len(d_by_subset[k]))].mean()
                                  for k in keys]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    nneg = int(sum(1 for k in keys if d_by_subset[k].mean() < 0))

    keep = {k: v for k, v in d_by_subset.items() if k != 'FD001'}
    m2 = float(np.mean([v.mean() for v in keep.values()]))
    rng2 = np.random.default_rng(BOOT_SEED)   # own stream: FD001-LOO CI
    d2 = np.empty(a.boot)
    for i in range(a.boot):
        d2[i] = float(np.mean([keep[k][rng2.integers(0, len(keep[k]), len(keep[k]))].mean()
                               for k in keep]))
    lo2, hi2 = np.percentile(d2, [2.5, 97.5])

    seed_means = {}
    for sd in SEEDS:
        v = [per_cell[f'{s}/s{sd}']['delta'] for s in SUBSETS]
        seed_means[sd] = float(np.mean(v))

    out = dict(
        arm='E4 (sa_mode=identity)', split_seed=a.split_seed,
        merged_delta=round(merged, 4), ci95=[round(float(lo), 4), round(float(hi), 4)],
        n_negative=nneg, n_subsets=len(keys),
        per_subset={k: round(float(v.mean()), 4) for k, v in d_by_subset.items()},
        per_cell=per_cell, engine_signs=signs,
        seed_level=dict(means={str(k): round(v, 4) for k, v in seed_means.items()},
                        sd=round(float(np.std(list(seed_means.values()), ddof=1)), 4)),
        loo_fd001=dict(merged=round(m2, 4), ci95=[round(float(lo2), 4), round(float(hi2), 4)]),
        n_boot=a.boot,
    )
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
