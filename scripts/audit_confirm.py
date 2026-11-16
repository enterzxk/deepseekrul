"""Independent recompute of the split_seed=29 confirmation set (E0 vs every arm).

Cells are taken from the frozen manifests, not by scanning config.json: a cell
that died during training never writes config.json, so a config-based scan
would silently drop exactly the cells that went wrong.

Per arm, against the SAME E0 baseline:
  * Delta = mean_u RMSE_u(arm) - mean_u RMSE_u(E0), paired at engine level
  * merged over subsets with equal weight, engine bootstrap 95% CI
  * direction count, per-subset values, per-seed engine sign counts
  * verdict against the frozen rule: merged <= -0.3 AND CI excludes 0 AND >=3/4
    same sign

Shared checks: every engine has exactly 5 cut points; every arm and E0 cover
the same engine set and cut points per (subset, seed); per-cell status and the
seven artifacts.

Usage:
  python3 scripts/audit_confirm.py <runs_root> \
      --manifest .../confirm29.json --manifest .../confirm29_e1b.json \
      --manifest .../confirm29_e2a.json [--boot 10000]
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
ARTIFACTS = ('val_pred.csv', 'val_noise_predictions.csv', 'val_missing_predictions.csv',
             'robustness_val.json', 'metrics.json', 'config.json', 'best.pt')


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


def expected_from_manifests(root, manifests):
    out = []
    for mf in manifests:
        d = json.load(open(mf))
        for row in d['runs']:
            c = row['config']
            out.append(dict(arm=arm_of(c), subset=c['subset'], seed=int(c['seed']),
                            run_id=row['run_id'], split_seed=int(c.get('split_seed', -1)),
                            dir=os.path.join(root, row['run_id'], c['subset'], f"s{int(c['seed'])}")))
    return out


def expected_from_scan(root, split_seed):
    out = []
    for cfgp in glob.glob(os.path.join(root, '*', '*', 's*', 'config.json')):
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            continue
        if int(cfg.get('split_seed', -1)) != int(split_seed):
            continue
        c = os.path.dirname(cfgp)
        out.append(dict(arm=arm_of(cfg), subset=cfg['subset'], seed=int(cfg['seed']),
                        run_id=os.path.basename(os.path.dirname(os.path.dirname(c))),
                        split_seed=int(cfg['split_seed']), dir=c))
    return out


def pred_file(d):
    """Prefer the full dump (it carries cutoff_cycle) over the minimal one."""
    for f in ('val_predictions.csv', 'val_pred.csv'):
        p = os.path.join(d, f)
        if os.path.exists(p):
            return p
    return None


def roots_and_cuts(path):
    acc = defaultdict(list)
    with open(path, newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        pcol = next((c for c in ('y_pred', 'y_pred_evaluated', 'y_pred_raw') if c in cols), None)
        if pcol is None or 'y_true' not in cols:
            raise KeyError(f'no prediction column in {path}: {cols}')
        for r in rd:
            cut = r.get('cutoff_cycle')
            acc[int(r.get('engine_id') or r.get('unit'))].append(
                (float(r['y_true']), float(r[pcol]),
                 int(float(cut)) if cut not in (None, '') else None))
    out = {}
    for u, pts in acc.items():
        cuts = tuple(sorted(p[2] for p in pts)) if all(p[2] is not None for p in pts) \
            else tuple(range(len(pts)))
        out[u] = dict(rmse=math.sqrt(sum((p - t) ** 2 for t, p, _ in pts) / len(pts)),
                      n=len(pts), cuts=cuts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--manifest', action='append', default=[])
    ap.add_argument('--split-seed', type=int, default=29)
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(20261116)

    exp = (expected_from_manifests(a.root, a.manifest) if a.manifest
           else expected_from_scan(a.root, a.split_seed))
    print('expected cells', len(exp))

    cells, status_tab, art_missing = {}, defaultdict(int), []
    for e in exp:
        st = os.path.join(e['dir'], 'status.json')
        s = 'NO_DIR'
        if os.path.isdir(e['dir']):
            s = json.load(open(st)).get('status') if os.path.exists(st) else 'NO_STATUS'
        status_tab[s] += 1
        miss = [f for f in ARTIFACTS if not os.path.exists(os.path.join(e['dir'], f))]
        if miss:
            art_missing.append(dict(run_id=e['run_id'], arm=e['arm'], subset=e['subset'],
                                    seed=e['seed'], status=s, missing=miss))
        # A cell only enters the analysis when it is terminal AND complete:
        # an EVAL_FAILED cell still has val_pred.csv + metrics.json, and
        # counting it would hide exactly the shortfall this audit exists for.
        if s == 'COMPLETE' and not miss and os.path.exists(os.path.join(e['dir'], 'val_pred.csv')):
            pf = pred_file(e['dir'])
            e['roots'] = roots_and_cuts(pf) if pf else {}
            e['pred_file'] = pf
            cells[(e['arm'], e['subset'], e['seed'])] = e

    print('status', dict(status_tab))
    per_arm = defaultdict(lambda: [0, 0])
    for e in exp:
        per_arm[e['arm']][1] += 1
        if (e['arm'], e['subset'], e['seed']) in cells:
            per_arm[e['arm']][0] += 1
    print('per-arm usable/expected', {k: '%d/%d' % tuple(v) for k, v in per_arm.items()})
    print('cells with artifacts missing', len(art_missing))
    for m in art_missing[:12]:
        print('   ', m['run_id'], m['status'], 'missing', m['missing'])

    problems = []
    for k, v in cells.items():
        bad = [u for u, r in v['roots'].items() if r['n'] != 5]
        if bad:
            problems.append(dict(kind='cuts', arm=k[0], subset=k[1], seed=k[2], engines=bad[:5]))
    print('cells with !=5 cuts', sum(1 for p in problems if p['kind'] == 'cuts'))

    arms = sorted({k[0] for k in cells} - {'E0'})
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
        complete = all((arm, s, sd) in cells for s in SUBSETS for sd in SEEDS) and \
            all(('E0', s, sd) in cells for s in SUBSETS for sd in SEEDS)
        out_arms[arm] = dict(
            complete=complete, merged=merged, ci95=[float(lo), float(hi)],
            n_negative=nneg, n_subsets=len(keys),
            per_subset={k: round(float(v.mean()), 4) for k, v in d_by_subset.items()},
            per_cell=per_cell, engine_signs=signs,
            criteria={'merged_le_neg_0p3': bool(c1), 'ci_excludes_0': bool(c2),
                      'at_least_3_of_4_same_sign': bool(c3)},
            verdict=('REPLICATED' if (c1 and c2 and c3) else 'NOT REPLICATED')
            if complete else 'INCOMPLETE')
        print()
        print('%s merged %+.4f  95%% CI [%+.4f, %+.4f]  direction %d/%d  %s%s'
              % (arm, merged, lo, hi, nneg, len(keys), out_arms[arm]['verdict'],
                 '' if complete else ' (arm not fully landed)'))
        print('   per-subset', out_arms[arm]['per_subset'])
        print('   criteria', out_arms[arm]['criteria'])
        print('   engine signs', signs)
        print('   per-cell', per_cell)

    out = dict(cells=len(cells), expected=len(exp), status=dict(status_tab),
               per_arm={k: '%d/%d' % tuple(v) for k, v in per_arm.items()},
               artifacts_missing=art_missing, pairing=pairing, problems=problems,
               fingerprints=dict(fps), results=out_arms, n_boot=a.boot)
    if a.json:
        open(a.json, 'w').write(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
