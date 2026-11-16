"""Independent recompute of the split_seed=29 confirmation set (E0 vs every arm).

Cells are discovered by reading config.json (split_seed == 29), never by
directory name, so a renamed tag cannot silently drop a cell.

Per arm, against the SAME E0 baseline:
  * Delta = mean_u RMSE_u(arm) - mean_u RMSE_u(E0), paired at engine level
  * merged over subsets with equal weight, engine bootstrap 95% CI
  * direction count, per-subset values, per-seed engine sign counts
  * verdict against the frozen rule: merged <= -0.3 AND CI excludes 0 AND >=3/4
    same sign

Shared checks: every engine has exactly 5 cut points; every arm and E0 cover
the same engine set and cut points per (subset, seed); per-cell fingerprint.

Usage:
  python3 scripts/audit_confirm.py <runs_root> [--split-seed 29] [--boot 10000]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303, 42, 43)


def arm_of(cfg):
    a = cfg.get('augment') or {}
    p = float(a.get('mask_prob', 0.0) or 0.0)
    n = float(a.get('noise_alpha', 0.0) or 0.0)
    if p == 0.0 and n == 0.0:
        return 'E0'
    if n == 0.0 and abs(p - 0.10) < 1e-9:
        return 'E1a'
    if n == 0.0 and abs(p - 0.20) < 1e-9:
        return 'E1b'
    if p == 0.0 and abs(n - 0.25) < 1e-9:
        return 'E2a'
    if p == 0.0 and abs(n - 0.50) < 1e-9:
        return 'E2b'
    return 'other(p=%g,n=%g)' % (p, n)


def discover(root, split_seed):
    cells = {}
    for cfgp in glob.glob(os.path.join(root, '*', '*', 's*', 'config.json')):
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            continue
        if int(cfg.get('split_seed', -1)) != int(split_seed):
            continue
        c = os.path.dirname(cfgp)
        if not os.path.exists(os.path.join(c, 'val_pred.csv')):
            continue
        cells[(arm_of(cfg), cfg['subset'], int(cfg['seed']))] = dict(dir=c, cfg=cfg)
    return cells


def roots_and_cuts(path):
    acc = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            acc[int(r.get('engine_id') or r.get('unit'))].append(
                (float(r['y_true']), float(r['y_pred']), int(float(r['cutoff_cycle']))))
    out = {}
    for u, pts in acc.items():
        out[u] = dict(rmse=math.sqrt(sum((p - t) ** 2 for t, p, _ in pts) / len(pts)),
                      n=len(pts), cuts=tuple(sorted(c for _, _, c in pts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--split-seed', type=int, default=29)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261116)

    cells = discover(a.root, a.split_seed)
    arms = sorted({k[0] for k in cells} - {'E0'})
    print('discovered cells', len(cells), 'arms', ['E0'] + arms)

    problems = []
    for k, v in list(cells.items()):
        v['roots'] = roots_and_cuts(os.path.join(v['dir'], 'val_pred.csv'))
        bad = [u for u, r in v['roots'].items() if r['n'] != 5]
        if bad:
            problems.append(dict(kind='cuts', arm=k[0], subset=k[1], seed=k[2], engines=bad[:5]))
    print('cells with !=5 cuts', sum(1 for p in problems if p['kind'] == 'cuts'))

    missing = [dict(arm=x, subset=s, seed=sd) for x in ['E0'] + arms
               for s in SUBSETS for sd in SEEDS if (x, s, sd) not in cells]
    print('missing cells', len(missing))

    pairing = []
    for arm in arms:
        for s in SUBSETS:
            for sd in SEEDS:
                ka, k0 = (arm, s, sd), ('E0', s, sd)
                if ka not in cells or k0 not in cells:
                    continue
                ea, e0 = cells[ka]['roots'], cells[k0]['roots']
                same = set(ea) == set(e0) and all(ea[u]['cuts'] == e0[u]['cuts'] for u in ea)
                pairing.append(dict(arm=arm, subset=s, seed=sd, same=same, n_engines=len(ea)))
                if not same:
                    problems.append(dict(kind='pairing', arm=arm, subset=s, seed=sd))
    print('pairing ok', sum(1 for p in pairing if p['same']), '/', len(pairing))

    fps = defaultdict(int)
    for v in cells.values():
        p = os.path.join(v['dir'], 'provenance.json')
        if os.path.exists(p):
            fps[json.load(open(p)).get('code_fingerprint')] += 1
    print('fingerprints', dict(fps))

    out_arms = {}
    for arm in arms:
        d_by_subset, per_cell, signs = {}, {}, {}
        for s in SUBSETS:
            ds = []
            for sd in SEEDS:
                ka, k0 = (arm, s, sd), ('E0', s, sd)
                if ka not in cells or k0 not in cells:
                    continue
                ra, r0 = cells[ka]['roots'], cells[k0]['roots']
                eng = sorted(set(ra) & set(r0))
                d = np.array([ra[u]['rmse'] - r0[u]['rmse'] for u in eng])
                ds.append(d)
                per_cell[f'{s}/{sd}'] = round(float(d.mean()), 4)
                signs.setdefault(s, []).append(int((d < 0).sum()))
            if ds:
                d_by_subset[s] = np.mean(np.stack(ds, axis=0), axis=0)
        if not d_by_subset:
            continue
        keys = list(d_by_subset)
        merged = float(np.mean([d_by_subset[k].mean() for k in keys]))
        draws = np.empty(a.boot)
        for i in range(a.boot):
            draws[i] = float(np.mean([d_by_subset[k][rng.integers(0, len(d_by_subset[k]),
                                                                   len(d_by_subset[k]))].mean()
                                      for k in keys]))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        nneg = int(sum(1 for k in keys if d_by_subset[k].mean() < 0))
        c1, c2, c3 = merged <= -0.3, (hi < 0), nneg >= 3
        ok = c1 and c2 and c3
        out_arms[arm] = dict(
            merged=merged, ci95=[float(lo), float(hi)], n_negative=nneg, n_subsets=len(keys),
            per_subset={k: round(float(v.mean()), 4) for k, v in d_by_subset.items()},
            per_cell=per_cell, engine_signs=signs,
            criteria=dict(merged_le_-0.3=bool(c1), ci_excludes_0=bool(c2),
                          at_least_3_of_4_same_sign=bool(c3)),
            verdict='REPLICATED' if ok else 'NOT REPLICATED')
        print()
        print('%s merged %+.4f  95%% CI [%+.4f, %+.4f]  direction %d/%d  %s'
              % (arm, merged, lo, hi, nneg, len(keys), out_arms[arm]['verdict']))
        print('   per-subset', out_arms[arm]['per_subset'])
        print('   criteria', out_arms[arm]['criteria'])
        print('   engine signs', signs)
        print('   per-cell', per_cell)

    out = dict(split_seed=a.split_seed, cells=len(cells), arms=['E0'] + arms,
               missing=missing, pairing=pairing, problems=problems,
               fingerprints=dict(fps), results=out_arms, n_boot=a.boot)
    if a.json:
        open(a.json, 'w').write(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
