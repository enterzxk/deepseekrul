"""Independent recompute of the E1/E2 grid against the E0 baseline.

Reads only the per-run prediction dumps; never trusts a stored metric.

  * recomputes the primary metric  mean_u RMSE_u  (val_engine_mean_roots_rmse)
    per cell and reports the max deviation from the recorded value
  * checks that every arm cell and its E0 partner cover the SAME validation
    engines and the SAME five cut points (paired design)
  * per arm: Delta vs E0 per subset, equal-weight merged Delta, engine-level
    paired bootstrap 95% CI (>=10000 draws), and the direction count

The bootstrap unit is the engine, never the window: d_u = mean over seeds of
(RMSE_u[arm] - RMSE_u[E0]); Delta = mean_u d_u. The merged statistic resamples
engines independently inside each subset and averages the four subset Deltas
with equal weight.

Usage:
  python3 scripts/audit_e1e2.py <runs_root> [--boot 10000] [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ARMS = ('E1a', 'E1b', 'E2a', 'E2b')
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)


def read_pred(path):
    acc, n = {}, 0
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            u = int(r['unit'])
            acc.setdefault(u, []).append((float(r['y_true']), float(r['y_pred'])))
            n += 1
    return acc, n


def cell_metrics(cell):
    acc, n = read_pred(cell / 'val_pred.csv')
    per = {}
    for u, pts in acc.items():
        per[u] = dict(mse=sum((p - t) ** 2 for t, p in pts) / len(pts),
                      n=len(pts),
                      cuts=tuple(sorted(round(t, 6) for t, _ in pts)))
    roots = {u: math.sqrt(v['mse']) for u, v in per.items()}
    mean_roots = sum(roots.values()) / len(roots)
    engine_equal = math.sqrt(sum(v['mse'] for v in per.values()) / len(per))
    pooled = math.sqrt(sum((p - t) ** 2 for pts in acc.values() for t, p in pts) / n)
    m = json.loads((cell / 'metrics.json').read_text(encoding='utf-8'))
    return dict(per=per, roots=roots, n_rows=n,
                mean_roots=mean_roots, engine_equal=engine_equal, pooled=pooled,
                rec=m)


SPLIT_SEED = 17   # discovery split; the confirmation set has its own script


def find_cell(root, arm, subset, seed, split_seed=SPLIT_SEED):
    """Resolve by config.json split_seed; refuse ambiguity.

    runs/ holds every split side by side under the same prefix, so a bare
    prefix glob is not enough -- see scripts/_cells.py for the incident.
    """
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
    return c if (c / 'val_pred.csv').exists() and (c / 'metrics.json').exists() else None


def boot_ci(d_by_subset, n_boot, rng):
    """d_by_subset: {subset: np.array over engines}. Returns (delta, lo, hi, merged)."""
    per_sub = {}
    for s, d in d_by_subset.items():
        per_sub[s] = (float(d.mean()), len(d))
    merged = float(np.mean([v[0] for v in per_sub.values()]))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for s, d in d_by_subset.items():
            idx = rng.integers(0, len(d), len(d))
            vals.append(d[idx].mean())
        draws[b] = float(np.mean(vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return per_sub, merged, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--json', default=None)
    ap.add_argument('--arms', default=','.join(ARMS),
                    help='comma-separated arms to analyse; incomplete arms are '
                         'skipped so a partial grid cannot look like a final one')
    a = ap.parse_args()
    root = Path(a.root)
    rng = np.random.default_rng(20261115)

    cells, deviations, structural = {}, [], []
    arms_wanted = tuple(x.strip() for x in a.arms.split(',') if x.strip())
    for arm in ('E0',) + arms_wanted:
        for subset in SUBSETS:
            for seed in SEEDS:
                c = find_cell(root, arm, subset, seed)
                if c is None:
                    structural.append(dict(arm=arm, subset=subset, seed=seed, state='MISSING'))
                    continue
                m = cell_metrics(c)
                cells[(arm, subset, seed)] = m
                rec = m['rec'].get('val_engine_mean_roots_rmse')
                deviations.append(dict(
                    arm=arm, subset=subset, seed=seed,
                    recomputed=m['mean_roots'], recorded=rec,
                    diff=(abs(m['mean_roots'] - rec) if rec is not None else None),
                    n_units=len(m['per']),
                    cuts5=all(v['n'] == 5 for v in m['per'].values()),
                ))

    # ---- structural: paired engine sets and cut points ----------------------
    pairing = []
    for arm in ARMS:
        for subset in SUBSETS:
            for seed in SEEDS:
                k_a, k_0 = (arm, subset, seed), ('E0', subset, seed)
                if k_a not in cells or k_0 not in cells:
                    continue
                ea, e0 = set(cells[k_a]['per']), set(cells[k_0]['per'])
                cuts_same = all(cells[k_a]['per'][u]['cuts'] == cells[k_0]['per'][u]['cuts']
                                for u in ea & e0)
                pairing.append(dict(arm=arm, subset=subset, seed=seed,
                                    same_engines=ea == e0, n_engines=len(ea),
                                    cuts_match=cuts_same))

    # ---- deltas ------------------------------------------------------------
    report = {}
    for arm in arms_wanted:
        have = sum(1 for s in SUBSETS for sd in SEEDS if (arm, s, sd) in cells)
        if have < len(SUBSETS) * len(SEEDS):
            report[arm] = dict(incomplete=True, cells_present=have,
                               cells_expected=len(SUBSETS) * len(SEEDS),
                               note='arm aggregate withheld until all cells land')
            continue
        d_by_subset, per_sub_cell = {}, {}
        for subset in SUBSETS:
            ds = []
            for seed in SEEDS:
                k_a, k_0 = (arm, subset, seed), ('E0', subset, seed)
                if k_a not in cells or k_0 not in cells:
                    continue
                roots_a, roots_0 = cells[k_a]['roots'], cells[k_0]['roots']
                engines = sorted(set(roots_a) & set(roots_0))
                d = np.array([roots_a[u] - roots_0[u] for u in engines])
                ds.append(d)
                per_sub_cell.setdefault(subset, {})[seed] = dict(
                    arm=float(np.mean([roots_a[u] for u in engines])),
                    e0=float(np.mean([roots_0[u] for u in engines])),
                    delta=float(d.mean()), n_engines=len(engines))
            if ds:
                d_by_subset[subset] = np.mean(np.stack(ds, axis=0), axis=0)
        if not d_by_subset:
            continue
        per_sub, merged, lo, hi = boot_ci(d_by_subset, a.boot, rng)
        report[arm] = dict(
            per_subset={s: dict(delta=per_sub[s][0], n_engines=per_sub[s][1])
                        for s in per_sub},
            merged_delta=merged, merged_ci95=[lo, hi],
            n_subsets_negative=int(sum(1 for s in per_sub if per_sub[s][0] < 0)),
            n_subsets=len(per_sub), per_cell=per_sub_cell,
        )

    devs = [d['diff'] for d in deviations if d['diff'] is not None]
    summary = dict(
        n_cells=len(cells), expected_cells=5 * 4 * 3,
        missing=structural,
        max_abs_diff_mean_roots=(max(devs) if devs else None),
        all_cuts5=all(d['cuts5'] for d in deviations),
        pairing_all_ok=all(p['same_engines'] and p['cuts_match'] for p in pairing),
        pairing=pairing, deviations=deviations, arms=report,
        n_boot=a.boot, boot_seed=20261115,
    )
    txt = json.dumps(summary, indent=1)
    if a.json:
        Path(a.json).write_text(txt, encoding='utf-8')
    print(txt)


if __name__ == '__main__':
    main()
