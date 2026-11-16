"""Independent recompute of the noise-stress protocol (E1/E2 vs E0).

The noise-stress rows are written to one CSV without a field column, so the
field blocks are recovered by replaying the generator order used by
``evaluation.eval_noise`` (for fid: for beta: rows). The reconstruction is
validated against ``robustness_val.json::noise.per_field`` before any Delta is
computed; if it does not reproduce that file to <1e-6 the script refuses to
report.

Two readings are produced:
  * ``official``: mean over fields of engine-equal RMSE (pooled over engines) -
    the quantity the harness already reports, used only as a cross-check
  * ``roots``:    mean_u RMSE_u per beta, pooled over the five fields - the
    study's primary aggregation, used for the paired Delta

Delta is paired at the engine level: d_u = RMSE_u[arm] - RMSE_u[E0], then
bootstrap over engines (>=10000 draws), subsets merged with equal weight.

Usage:
  python3 scripts/audit_noise_pressure.py <runs_root> --arms E1a --boot 10000
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


def load_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def engine_mse(rows):
    acc = defaultdict(list)
    for r in rows:
        acc[int(r['engine_id'])].append((float(r['y_true']), float(r['y_pred_evaluated'])))
    out = {}
    for u, pts in acc.items():
        out[u] = (sum((p - t) ** 2 for t, p in pts) / len(pts), len(pts))
    return out


def mean_roots(per):
    return sum(math.sqrt(v[0]) for v in per.values()) / len(per)


def engine_equal(per):
    return math.sqrt(sum(v[0] for v in per.values()) / len(per))


SPLIT_SEED = 17   # discovery split


def find_cell(root, arm, subset, seed, split_seed=SPLIT_SEED):
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
    return c if (c / 'val_noise_predictions.csv').exists() else None


def blocks_for(cell):
    """Return {(fid, beta): rows} plus the validation against robustness_val.json."""
    rows = load_csv(cell / 'val_noise_predictions.csv')
    rb = json.loads((cell / 'robustness_val.json').read_text())
    per_field = rb['noise']['per_field']
    betas = sorted({float(r['corruption_level']) for r in rows})
    n_fields = len(per_field) // len(betas)
    n_block = len(rows) // (n_fields * len(betas))
    order = []
    for fid in range(n_fields):
        for b in betas:
            order.append((fid, b))
    blocks = {}
    for i, key in enumerate(order):
        blocks[key] = rows[i * n_block:(i + 1) * n_block]
    # validate against the harness's own per-field numbers
    worst = 0.0
    for i, key in enumerate(order):
        per = engine_mse(blocks[key])
        worst = max(worst, abs(engine_equal(per) - float(per_field[i]['engine_equal_rmse'])))
    return blocks, betas, n_fields, worst


BOOT_SEED = 20261115


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--arms', default='E1a,E1b,E2a,E2b')
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root = Path(a.root)
    arms = [x.strip() for x in a.arms.split(',') if x.strip()]

    cache, worst_all = {}, 0.0
    for arm in ('E0',) + tuple(arms):
        for s in SUBSETS:
            for sd in SEEDS:
                c = find_cell(root, arm, s, sd)
                if c is None:
                    continue
                blocks, betas, nf, worst = blocks_for(c)
                worst_all = max(worst_all, worst)
                cache[(arm, s, sd)] = dict(blocks=blocks, betas=betas, n_fields=nf)

    out = dict(n_cells=len(cache), reconstruction_max_abs_err=worst_all,
               n_fields=(next(iter(cache.values()))['n_fields'] if cache else None),
               betas=(next(iter(cache.values()))['betas'] if cache else None), arms={})

    if worst_all > 1e-6:
        out['error'] = 'field reconstruction does not reproduce robustness_val.json'
        print(json.dumps(out, indent=1))
        return 1

    betas = next(iter(cache.values()))['betas']
    for arm in arms:
        have = sum(1 for s in SUBSETS for sd in SEEDS if (arm, s, sd) in cache)
        if have < len(SUBSETS) * len(SEEDS):
            out['arms'][arm] = dict(incomplete=True, cells_present=have,
                                    cells_expected=len(SUBSETS) * len(SEEDS))
            continue
        res = {}
        for b in betas:
            d_by_subset, official = {}, {}
            for s in SUBSETS:
                ds, offs = [], []
                for sd in SEEDS:
                    ka, k0 = (arm, s, sd), ('E0', s, sd)
                    if ka not in cache or k0 not in cache:
                        continue
                    ra = [r for k, blk in cache[ka]['blocks'].items() if k[1] == b for r in blk]
                    r0 = [r for k, blk in cache[k0]['blocks'].items() if k[1] == b for r in blk]
                    pa, p0 = engine_mse(ra), engine_mse(r0)
                    eng = sorted(set(pa) & set(p0))
                    ds.append(np.array([math.sqrt(pa[u][0]) - math.sqrt(p0[u][0]) for u in eng]))
                    offs.append((mean_roots(pa), mean_roots(p0)))
                if ds:
                    d_by_subset[s] = np.mean(np.stack(ds, axis=0), axis=0)
                    official[s] = dict(arm=float(np.mean([x[0] for x in offs])),
                                       e0=float(np.mean([x[1] for x in offs])))
            if not d_by_subset:
                continue
            merged = float(np.mean([v.mean() for v in d_by_subset.values()]))
            rng = np.random.default_rng(BOOT_SEED)   # own stream per (arm, beta)
            draws = np.empty(a.boot)
            keys = list(d_by_subset)
            for i in range(a.boot):
                vals = [d_by_subset[k][rng.integers(0, len(d_by_subset[k]), len(d_by_subset[k]))].mean()
                        for k in keys]
                draws[i] = float(np.mean(vals))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            res[f'{b:g}'] = dict(
                per_subset={k: dict(delta=float(v.mean()), n_engines=int(len(v)))
                            for k, v in d_by_subset.items()},
                merged_delta=merged, ci95=[float(lo), float(hi)],
                n_negative=int(sum(1 for v in d_by_subset.values() if v.mean() < 0)),
                n_subsets=len(d_by_subset), absolute=official)
        out['arms'][arm] = res

    txt = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(txt, encoding='utf-8')
    print(txt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
