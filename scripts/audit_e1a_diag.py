"""Diagnostics for one arm of the sensor-augmentation grid.

Answers three questions that a merged Delta alone cannot:
  1. how large is the effect relative to seed-to-seed fluctuation
  2. what is the merged Delta without the subset that contributes most
  3. what drives the single cell whose sign disagrees with the rest

All numbers are recomputed from ``val_pred.csv``; nothing is read from a
stored metric.

Usage:
  python3 scripts/audit_e1a_diag.py <runs_root> --arm E1a [--exclude FD001]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)


def roots_of(cell):
    acc = defaultdict(list)
    with open(cell / 'val_pred.csv', newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            acc[int(r['unit'])].append((float(r['y_true']), float(r['y_pred'])))
    return {u: math.sqrt(sum((p - t) ** 2 for t, p in pts) / len(pts)) for u, pts in acc.items()}


SPLIT_SEED = 17   # discovery split


def cell(root, arm, subset, seed, split_seed=SPLIT_SEED):
    """Resolve by config.json split_seed; refuse ambiguity (see _cells.py)."""
    hits = sorted(root.glob(f'{arm}-{subset}-s{seed}-*'))
    ok = []
    for h in hits:
        cfgp = h / subset / f's{seed}' / 'config.json'
        if not cfgp.exists():
            continue
        if int(json.loads(cfgp.read_text()).get('split_seed', -1)) == split_seed:
            ok.append(h / subset / f's{seed}')
    if len(ok) != 1:
        return None
    c = ok[0]
    return c if (c / 'val_pred.csv').exists() else None


def boot(d_by_subset, n_boot, rng):
    keys = list(d_by_subset)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        vals = [d_by_subset[k][rng.integers(0, len(d_by_subset[k]), len(d_by_subset[k]))].mean()
                for k in keys]
        draws[i] = float(np.mean(vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    merged = float(np.mean([d_by_subset[k].mean() for k in keys]))
    return merged, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--arm', default='E1a')
    ap.add_argument('--exclude', default='')
    ap.add_argument('--boot', type=int, default=10000)
    a = ap.parse_args()
    root = Path(a.root)
    rng = np.random.default_rng(20261115)

    per_cell, d_by_subset = {}, {}
    for s in SUBSETS:
        for sd in SEEDS:
            ca, c0 = cell(root, a.arm, s, sd), cell(root, 'E0', s, sd)
            if ca is None or c0 is None:
                continue
            ra, r0 = roots_of(ca), roots_of(c0)
            eng = sorted(set(ra) & set(r0))
            d = np.array([ra[u] - r0[u] for u in eng])
            per_cell[(s, sd)] = dict(delta=float(d.mean()), arm=float(np.mean([ra[u] for u in eng])),
                                     e0=float(np.mean([r0[u] for u in eng])),
                                     d=d, engines=eng, roots_arm=ra, roots_e0=r0)
            d_by_subset.setdefault(s, []).append(d)
    for s in d_by_subset:
        d_by_subset[s] = np.mean(np.stack(d_by_subset[s], axis=0), axis=0)

    all_cells = [v['delta'] for v in per_cell.values()]
    out = dict(arm=a.arm, n_cells=len(all_cells),
               cell_mean=float(np.mean(all_cells)),
               cell_sd=float(np.std(all_cells, ddof=1)) if len(all_cells) > 1 else None,
               cell_range=[float(min(all_cells)), float(max(all_cells))])

    out['per_subset'] = {}
    for s in SUBSETS:
        v = [per_cell[(s, sd)]['delta'] for sd in SEEDS if (s, sd) in per_cell]
        if v:
            out['per_subset'][s] = dict(mean=float(np.mean(v)),
                                        sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                        cells=[round(x, 4) for x in v],
                                        n_engines=len(per_cell[(s, SEEDS[0])]['engines']))

    # seed-level: mean over subsets for each seed -> the quantity the study
    # averages, so its spread is the seed-to-seed fluctuation of the headline
    seed_means = {}
    for sd in SEEDS:
        v = [per_cell[(s, sd)]['delta'] for s in SUBSETS if (s, sd) in per_cell]
        if v:
            seed_means[sd] = float(np.mean(v))
    out['seed_level'] = dict(means=seed_means,
                             sd=(float(np.std(list(seed_means.values()), ddof=1))
                                 if len(seed_means) > 1 else None),
                             range=[float(min(seed_means.values())), float(max(seed_means.values()))]
                             if seed_means else None)

    excl = [x for x in a.exclude.split(',') if x]
    keep = {k: v for k, v in d_by_subset.items() if k not in excl}
    if keep:
        m, lo, hi = boot(keep, a.boot, rng)
        out['merged_all'] = dict(delta=float(np.mean([v.mean() for v in d_by_subset.values()])),
                                 subsets=list(d_by_subset))
        out['merged_excluding'] = dict(excluded=excl, delta=m, ci95=[lo, hi],
                                       n_subsets=len(keep),
                                       n_negative=int(sum(1 for v in keep.values() if v.mean() < 0)))

    # the disagreeing cell
    bad = sorted(((v['delta'], k) for k, v in per_cell.items()))[-1]
    s, sd = bad[1]
    v = per_cell[(s, sd)]
    d = v['d']
    order = np.argsort(-np.abs(d))[:3]
    out['worst_cell'] = dict(
        subset=s, seed=sd, delta=v['delta'], arm=v['arm'], e0=v['e0'],
        n_engines=len(d), n_negative=int((d < 0).sum()),
        engine_delta_min=float(d.min()), engine_delta_max=float(d.max()),
        engine_delta_sd=float(d.std(ddof=1)),
        top3=[dict(engine=int(v['engines'][i]), delta=float(d[i]),
                   arm=round(float(v['roots_arm'][v['engines'][i]]), 3),
                   e0=round(float(v['roots_e0'][v['engines'][i]]), 3)) for i in order],
    )
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
