"""Final four-arm table for the sensor-augmentation stage (E1a/E1b/E2a/E2b vs E0).

Everything is recomputed from the saved predictions. Nothing is taken from a
stored metric except the engine-equal percentages, which are read from the
harness's own ``robustness_val.json`` so the comparison with the old study uses
the same aggregation the old study reported.

Outputs, per arm:
  clean      merged Delta (mean-of-roots), 95% CI (engine bootstrap), direction
             count, per-subset Delta, per-engine sign counts
  seed level mean over subsets per seed -> sd of the headline
  LOO        merged Delta without FD001
  noise      same at beta = 0.25 / 0.5 / 1.0
  pct        relative improvement in engine-equal RMSE vs E0 at each beta

Usage:
  python3 scripts/audit_final_table.py <runs_root> --boot 10000 --json out.json
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
SEEDS = (101, 202, 303)
ARMS = ('E1a', 'E1b', 'E2a', 'E2b')
BETAS = (0.25, 0.5, 1.0)


def cell_dir(root, arm, sub, seed):
    hits = sorted(glob.glob(os.path.join(root, f'{arm}-{sub}-s{seed}-*')))
    if len(hits) != 1:
        return None
    c = os.path.join(hits[0], sub, f's{seed}')
    return c if os.path.exists(os.path.join(c, 'val_pred.csv')) else None


def per_engine_roots(path, col='y_pred'):
    acc = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key = r.get('engine_id') or r.get('unit')
            acc[int(key)].append((float(r['y_true']), float(r[col])))
    return {u: math.sqrt(sum((p - t) ** 2 for t, p in pts) / len(pts)) for u, pts in acc.items()}


def noise_by_beta(cell):
    rows = list(csv.DictReader(open(os.path.join(cell, 'val_noise_predictions.csv'),
                                    newline='', encoding='utf-8-sig')))
    rb = json.loads(open(os.path.join(cell, 'robustness_val.json')).read())
    pf = rb['noise']['per_field']
    betas = sorted({float(r['corruption_level']) for r in rows})
    nf = len(pf) // len(betas)
    nb = len(rows) // (nf * len(betas))
    order = [(fid, b) for fid in range(nf) for b in betas]
    out = {}
    for i, (fid, b) in enumerate(order):
        blk = rows[i * nb:(i + 1) * nb]
        acc = defaultdict(list)
        for r in blk:
            acc[int(r['engine_id'])].append((float(r['y_true']), float(r['y_pred_evaluated'])))
        out.setdefault(b, {})
        for u, pts in acc.items():
            out[b].setdefault(u, []).extend(pts)
    return {b: {u: math.sqrt(sum((p - t) ** 2 for t, p in pts) / len(pts))
                for u, pts in d.items()} for b, d in out.items()}


def boot_merge(d_by_subset, n_boot, rng):
    keys = list(d_by_subset)
    merged = float(np.mean([d_by_subset[k].mean() for k in keys]))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = float(np.mean([d_by_subset[k][rng.integers(0, len(d_by_subset[k]),
                                                               len(d_by_subset[k]))].mean()
                                  for k in keys]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return merged, float(lo), float(hi), int(sum(1 for k in keys if d_by_subset[k].mean() < 0)), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261115)

    clean, noise, eqpct = {}, {}, {}
    for arm in ('E0',) + ARMS:
        for s in SUBSETS:
            for sd in SEEDS:
                c = cell_dir(a.root, arm, s, sd)
                if c is None:
                    continue
                clean[(arm, s, sd)] = per_engine_roots(os.path.join(c, 'val_pred.csv'))
                noise[(arm, s, sd)] = noise_by_beta(c)
                rb = json.loads(open(os.path.join(c, 'robustness_val.json')).read())
                eqpct[(arm, s, sd)] = {float(k): v['engine_equal_rmse']
                                       for k, v in rb['noise']['aggregate'].items()}

    report = {}
    for arm in ARMS:
        arm_out = {}
        # ---------------- clean
        dsub, cell_d, sign = {}, {}, {}
        for s in SUBSETS:
            ds, signs = [], []
            for sd in SEEDS:
                ka, k0 = (arm, s, sd), ('E0', s, sd)
                if ka not in clean or k0 not in clean:
                    continue
                ra, r0 = clean[ka], clean[k0]
                eng = sorted(set(ra) & set(r0))
                d = np.array([ra[u] - r0[u] for u in eng])
                ds.append(d)
                signs.append(int((d < 0).sum()))
                cell_d[f'{s}/{sd}'] = round(float(d.mean()), 4)
            if ds:
                dsub[s] = np.mean(np.stack(ds, axis=0), axis=0)
                sign[s] = signs
        merged, lo, hi, nneg, nsub = boot_merge(dsub, a.boot, rng)
        keep = {k: v for k, v in dsub.items() if k != 'FD001'}
        m2, l2, h2, n2, _ = boot_merge(keep, a.boot, rng)
        seed_means = {}
        for sd in SEEDS:
            v = [float(np.mean([clean[(arm, s, sd)][u] - clean[('E0', s, sd)][u]
                                for u in set(clean[(arm, s, sd)]) & set(clean[('E0', s, sd)])]))
                 for s in SUBSETS if (arm, s, sd) in clean]
            if v:
                seed_means[sd] = float(np.mean(v))
        arm_out['clean'] = dict(
            merged=merged, ci95=[lo, hi], n_negative=nneg, n_subsets=nsub,
            per_subset={k: round(float(v.mean()), 4) for k, v in dsub.items()},
            per_cell=cell_d, engine_signs=sign,
            seed_level=dict(means=seed_means,
                            sd=(float(np.std(list(seed_means.values()), ddof=1))
                                if len(seed_means) > 1 else None)),
            loo_fd001=dict(merged=m2, ci95=[l2, h2], n_negative=n2, n_subsets=len(keep)),
        )
        # ---------------- noise
        nz = {}
        for b in BETAS:
            dsub_b, pcts = {}, []
            for s in SUBSETS:
                ds = []
                for sd in SEEDS:
                    ka, k0 = (arm, s, sd), ('E0', s, sd)
                    if ka not in noise or k0 not in noise:
                        continue
                    ra, r0 = noise[ka].get(b, {}), noise[k0].get(b, {})
                    eng = sorted(set(ra) & set(r0))
                    if not eng:
                        continue
                    ds.append(np.array([ra[u] - r0[u] for u in eng]))
                if ds:
                    dsub_b[s] = np.mean(np.stack(ds, axis=0), axis=0)
            if not dsub_b:
                continue
            m, l, h, nn, ns = boot_merge(dsub_b, a.boot, rng)
            for s in SUBSETS:
                ra = [eqpct[(arm, s, sd)][b] for sd in SEEDS if (arm, s, sd) in eqpct]
                r0 = [eqpct[('E0', s, sd)][b] for sd in SEEDS if ('E0', s, sd) in eqpct]
                if ra and r0:
                    pcts.append((float(np.mean(r0)) - float(np.mean(ra))) / float(np.mean(r0)) * 100)
            nz[f'{b:g}'] = dict(merged=m, ci95=[l, h], n_negative=nn, n_subsets=ns,
                                per_subset={k: round(float(v.mean()), 4) for k, v in dsub_b.items()},
                                pct_improvement_engine_equal=round(float(np.mean(pcts)), 3) if pcts else None,
                                pct_per_subset=None)
        arm_out['noise'] = nz
        report[arm] = arm_out

    out = dict(n_boot=a.boot, boot_seed=20261115, arms=report)
    txt = json.dumps(out, indent=1)
    if a.json:
        open(a.json, 'w').write(txt)

    # compact printout
    for arm, v in report.items():
        c = v['clean']
        print('%-4s clean merged=%+.4f CI[%+.4f,%+.4f] neg=%d/%d  LOO(FD001)=%+.4f CI[%+.4f,%+.4f]'
              % (arm, c['merged'], c['ci95'][0], c['ci95'][1], c['n_negative'], c['n_subsets'],
                 c['loo_fd001']['merged'], c['loo_fd001']['ci95'][0], c['loo_fd001']['ci95'][1]))
        print('     per-subset', c['per_subset'])
        print('     engine signs', c['engine_signs'])
        print('     seed means', {k: round(x, 4) for k, x in c['seed_level']['means'].items()},
              'sd', round(c['seed_level']['sd'], 4) if c['seed_level']['sd'] else None)
        for b, w in v['noise'].items():
            print('     beta=%-4s merged=%+.4f CI[%+.4f,%+.4f] neg=%d/%d  pct(eq)=%+.2f%%  per-subset %s'
                  % (b, w['merged'], w['ci95'][0], w['ci95'][1], w['n_negative'], w['n_subsets'],
                     w['pct_improvement_engine_equal'], w['per_subset']))


if __name__ == '__main__':
    main()
