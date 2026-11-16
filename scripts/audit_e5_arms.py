"""E5 four-arm recompute against the internal `none` baseline.

E5's baseline is the `none` arm retrained under the same new h8, never E0:
E0 lives on a different fingerprint and is only a magnitude reference.

Delta = arm - none, paired per engine inside (subset, seed); four subsets merged
with equal weight; 95% CI from an engine-level bootstrap. Also reports the
per-subset values, direction count, seed-level sd and the leave-FD001-out merge,
so the table can be read exactly like the E1/E2 ones.

Usage:
  python3 scripts/audit_e5_arms.py <runs_root> [--boot 10000]
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
ARMS = {'lr4': 'E5lr4', 'lr14': 'E5lr14', 'mlp': 'E5mlp'}
BASE = 'E5none'


def roots(path):
    acc = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            acc[int(r.get('unit') or r.get('engine_id'))].append(
                (float(r['y_true']), float(r['y_pred'])))
    return {u: math.sqrt(sum((p - t) ** 2 for t, p in v) / len(v)) for u, v in acc.items()}


BOOT_SEED = 20261116


def boot(d_by_subset, n_boot):
    # Own RNG per call: the main CI, the FD001-LOO CI and lr4-mlp are three
    # different statistics and must not consume one another's stream.
    rng = np.random.default_rng(BOOT_SEED)
    keys = list(d_by_subset)
    merged = float(np.mean([d_by_subset[k].mean() for k in keys]))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = float(np.mean([d_by_subset[k][rng.integers(0, len(d_by_subset[k]),
                                                               len(d_by_subset[k]))].mean()
                                  for k in keys]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return merged, float(lo), float(hi), int(sum(1 for k in keys if d_by_subset[k].mean() < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    out = {}
    for name, prefix in ARMS.items():
        dsub, per_cell, seed_means = {}, {}, {}
        ok = True
        for s in SUBSETS:
            ds = []
            for sd in SEEDS:
                try:
                    ca = Path(resolve(a.root, prefix, s, sd, 17))
                    c0 = Path(resolve(a.root, BASE, s, sd, 17))
                except RuntimeError:
                    ok = False
                    continue
                ra, r0 = roots(ca / 'val_pred.csv'), roots(c0 / 'val_pred.csv')
                eng = sorted(set(ra) & set(r0))
                d = np.array([ra[u] - r0[u] for u in eng])
                ds.append(d)
                per_cell[f'{s}/s{sd}'] = dict(delta=round(float(d.mean()), 4),
                                              n_engines=len(eng),
                                              arm=round(float(np.mean([ra[u] for u in eng])), 4),
                                              none=round(float(np.mean([r0[u] for u in eng])), 4))
            if ds:
                dsub[s] = np.mean(np.stack(ds, axis=0), axis=0)
        if not dsub:
            out[name] = dict(complete=False)
            continue
        merged, lo, hi, nneg = boot(dsub, a.boot)
        keep = {k: v for k, v in dsub.items() if k != 'FD001'}
        m2, l2, h2, n2 = boot(keep, a.boot)
        for sd in SEEDS:
            v = [per_cell[f'{s}/s{sd}']['delta'] for s in SUBSETS if f'{s}/s{sd}' in per_cell]
            if v:
                seed_means[sd] = round(float(np.mean(v)), 4)
        out[name] = dict(
            complete=ok, merged=round(merged, 4), ci95=[round(lo, 4), round(hi, 4)],
            n_negative=nneg, n_subsets=len(dsub),
            per_subset={k: round(float(v.mean()), 4) for k, v in dsub.items()},
            per_cell=per_cell,
            seed_level=dict(means={str(k): v for k, v in seed_means.items()},
                            sd=round(float(np.std(list(seed_means.values()), ddof=1)), 4)
                            if len(seed_means) > 1 else None),
            loo_fd001=dict(merged=round(m2, 4), ci95=[round(l2, 4), round(h2, 4)]),
        )
        print('%-5s merged %+.4f CI[%+.4f,%+.4f] neg %d/%d  per-subset %s'
              % (name, merged, lo, hi, nneg, len(dsub), out[name]['per_subset']), flush=True)

    # report-only (does not change any frozen criterion): the direct
    # lr4 - mlp paired comparison, because "mlp CI contains 0" is an absence
    # of evidence, not evidence of absence.
    dsub, per_cell = {}, {}
    for s in SUBSETS:
        ds = []
        for sd in SEEDS:
            try:
                ca = Path(resolve(a.root, 'E5lr4', s, sd, 17))
                cb = Path(resolve(a.root, 'E5mlp', s, sd, 17))
            except RuntimeError:
                continue
            ra, rb = roots(ca / 'val_pred.csv'), roots(cb / 'val_pred.csv')
            eng = sorted(set(ra) & set(rb))
            d = np.array([ra[u] - rb[u] for u in eng])
            ds.append(d)
            per_cell[f'{s}/s{sd}'] = round(float(d.mean()), 4)
        if ds:
            dsub[s] = np.mean(np.stack(ds, axis=0), axis=0)
    if dsub:
        m, lo, hi, nneg = boot(dsub, a.boot)
        out['lr4_minus_mlp'] = dict(merged=round(m, 4), ci95=[round(lo, 4), round(hi, 4)],
                                    n_negative=nneg, n_subsets=len(dsub),
                                    per_subset={k: round(float(v.mean()), 4)
                                                for k, v in dsub.items()},
                                    per_cell=per_cell)
        print('lr4-mlp merged %+.4f CI[%+.4f,%+.4f] neg %d/%d' % (m, lo, hi, nneg, len(dsub)))

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    else:
        print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
