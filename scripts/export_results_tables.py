#!/usr/bin/env python3
"""Export the two result tables reviewers need to recompute everything.

SENSOR_BASELINE_RESULTS.csv      one row per E0 cell (discovery + confirmation)
SENSOR_AUGMENTATION_RESULTS.csv  one row per augmented cell, with its paired E0
                                 baseline and the arm-level statistics repeated
                                 on every row of that arm

Everything is recomputed from the per-engine predictions, not copied from summary
JSONs, except the fields that only exist in metadata (`best_ep`, `elapsed_s`,
`code_fingerprint`).

Paired-difference aggregation -- deliberately identical to the scripts the verdicts
were made on (`audit_e1e2.py` for split 17, `audit_confirm.py` for split 29):

    per (subset, seed):  d_u = RMSE_u(arm) - RMSE_u(E0)   over the common engines
    per subset:          mean over the seeds of those vectors, elementwise
    merged:              equal-weight mean of the four subset means
    CI:                  10000 engine-level bootstrap draws, one rng created once
                         per split and carried across arms in sorted arm order

Averaging the seeds *before* resampling treats the three seeds as repeated
measurements of the same engine. Pooling them instead (3x the units) narrows the
interval by construction; that mistake was made once here and corrected.

Bootstrap seeds follow the run that produced the verdicts:
    split_seed=17 -> 20261115   (audit_e1e2.py)
    split_seed=29 -> 20261116   (audit_confirm.py)
Each arm is additionally bootstrapped under the other seed so the Monte-Carlo
sensitivity of the interval endpoints can be reported. That is a measurement of MC
error, not seed shopping: the canonical column stays the verdict-time one.

Caliber warning, also in SENSOR_RESULTS_README.md:
    `val_engine_mean_roots_rmse` / `clean_mean_roots` come from the clean evaluation
    path (val_pred.csv). The `noise_beta*` columns come from the noise-stress path
    (val_noise_predictions.csv, column y_pred_evaluated) and are NOT comparable with
    the clean numbers. Do not put beta=0 against a clean baseline.

Cells are resolved through `_cells.py` (split_seed filter), never by name prefix.
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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._cells import try_resolve  # noqa: E402

SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
DISCOVERY_SEEDS = [101, 202, 303]
CONFIRM_SEEDS = [101, 202, 303, 42, 43]
AUG_ARMS = ['E1a', 'E1b', 'E2a', 'E2b']
CONFIRM_AUG_ARMS = ['E1a', 'E1b', 'E2a']
BETAS = (0.25, 0.5, 1.0)
CANON_BOOT_SEED = {17: 20261115, 29: 20261116}


def read_rows(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def per_engine_roots(rows, pred_col='y_pred'):
    acc = defaultdict(list)
    for r in rows:
        key = r.get('engine_id') or r.get('unit')
        acc[int(key)].append((float(r['y_true']), float(r[pred_col])))
    return {u: math.sqrt(sum((p - t) ** 2 for t, p in pts) / len(pts))
            for u, pts in acc.items()}


def mean_of_roots(per_engine):
    return float(np.mean(list(per_engine.values())))


def cell_facts(cell: Path):
    cfg = json.loads((cell / 'config.json').read_text())
    met = json.loads((cell / 'metrics.json').read_text())
    out = {
        'split_seed': cfg['split_seed'],
        'best_ep': met.get('best_ep'),
        'metrics_mean_roots': met.get('val_engine_mean_roots_rmse'),
        'elapsed_s': None,
        'code_fingerprint': None,
    }
    tc = cell / 'train_complete.json'
    if tc.exists():
        out['elapsed_s'] = json.loads(tc.read_text()).get('elapsed_s')
    pv = cell / 'provenance.json'
    if pv.exists():
        out['code_fingerprint'] = json.loads(pv.read_text()).get('code_fingerprint')
    clean = per_engine_roots(read_rows(cell / 'val_pred.csv'))
    out['clean_per_engine'] = clean
    out['clean_mean_roots'] = mean_of_roots(clean)
    nz = cell / 'val_noise_predictions.csv'
    if nz.exists():
        rows = read_rows(nz)
        for b in BETAS:
            sel = [r for r in rows if abs(float(r['corruption_level']) - b) < 1e-9]
            if sel:
                pe = per_engine_roots(sel, 'y_pred_evaluated')
                out[f'noise_beta{b:g}_per_engine'] = pe
                out[f'noise_beta{b:g}_mean_roots'] = mean_of_roots(pe)
    return out


def bootstrap_merged(d_by_subset, n_boot, rng):
    """d_by_subset: {subset: array over engines, already averaged across seeds}."""
    keys = list(d_by_subset)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        vals = [d_by_subset[k][rng.integers(0, len(d_by_subset[k]),
                                            len(d_by_subset[k]))].mean()
                for k in keys]
        draws[i] = float(np.mean(vals))
    point = float(np.mean([d_by_subset[k].mean() for k in keys]))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def all_arm_cis(gathered, arms, n_boot, seed):
    """One fresh rng per arm, seeded from (seed, arm ordinal).

    The verdict-time scripts created one rng and carried it across arms, so an arm's
    endpoints depended on how many arms had been bootstrapped before it -- i.e. on the
    command that happened to be run. Measured: E1b/discovery came out as [-0.992,
    -0.367] in the verdict run, [-1.0000, -0.3758] with one shared stream over four
    arms, and [-1.0033, -0.3723] with a fresh stream, all from identical per-cell
    deltas (verified 12/12) and an identical bootstrap function (verified bit-equal on
    the same input). A number that depends on invocation history is not reproducible.

    Seeding per arm removes that dependency: the same arm always gets the same stream,
    no matter which other arms are present. The cost is that endpoints move by about
    the measured MC drift (<=0.015); `arm_mc_endpoint_drift_max` reports that, and the
    verdict criteria are re-checked in the summary.
    """
    out = {}
    for idx, arm in enumerate(sorted(a for a in arms if a in gathered)):
        rng = np.random.default_rng([seed, idx])
        out[arm] = bootstrap_merged(gathered[arm][2], n_boot, rng)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--boot', type=int, default=10000)
    args = ap.parse_args()

    runs = Path(args.runs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- baseline
    base_rows = []
    base_index = {}
    for split, seeds in ((17, DISCOVERY_SEEDS), (29, CONFIRM_SEEDS)):
        for subset in SUBSETS:
            for seed in seeds:
                cell = try_resolve(runs, 'E0', subset, seed, split)
                if cell is None:
                    continue
                f = cell_facts(Path(cell))
                base_index[(subset, seed, split)] = f
                base_rows.append({
                    'run_id': Path(cell).parent.parent.name,
                    'subset': subset, 'seed': seed, 'split_seed': split,
                    'val_engine_mean_roots_rmse': round(f['clean_mean_roots'], 6),
                    'val_engine_mean_roots_rmse_recorded': f['metrics_mean_roots'],
                    'best_ep': f['best_ep'], 'elapsed_s': f['elapsed_s'],
                    'code_fingerprint': f['code_fingerprint'],
                    'n_engines': len(f['clean_per_engine']),
                })
    with open(out_dir / 'SENSOR_BASELINE_RESULTS.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(base_rows[0]))
        w.writeheader()
        w.writerows(base_rows)

    # ------------------------------------------------------------ augmentation
    aug_rows = []
    for split, seeds, arms in ((17, DISCOVERY_SEEDS, AUG_ARMS),
                               (29, CONFIRM_SEEDS, CONFIRM_AUG_ARMS)):
        gathered = {}
        for arm in arms:
            per_seed = defaultdict(dict)
            arm_facts = {}
            for subset in SUBSETS:
                for seed in seeds:
                    cell = try_resolve(runs, arm, subset, seed, split)
                    base = base_index.get((subset, seed, split))
                    if cell is None or base is None:
                        continue
                    f = cell_facts(Path(cell))
                    arm_facts[(subset, seed)] = (Path(cell), f)
                    eng = sorted(set(f['clean_per_engine']) & set(base['clean_per_engine']))
                    per_seed[subset][seed] = np.array(
                        [f['clean_per_engine'][u] - base['clean_per_engine'][u]
                         for u in eng])
            if not per_seed:
                continue
            d_by_subset = {}
            for subset, by_seed in per_seed.items():
                arrs = [by_seed[sd] for sd in seeds if sd in by_seed]
                if not arrs:
                    continue
                if len({len(a) for a in arrs}) != 1:
                    raise SystemExit(f'{arm} {subset}: seeds disagree on engine count')
                d_by_subset[subset] = np.mean(np.stack(arrs, axis=0), axis=0)
            gathered[arm] = (per_seed, arm_facts, d_by_subset)

        canon_seed = CANON_BOOT_SEED[split]
        other_seed = 20261115 if canon_seed == 20261116 else 20261116
        canon = all_arm_cis(gathered, arms, args.boot, canon_seed)
        alt = all_arm_cis(gathered, arms, args.boot, other_seed)

        for arm in sorted(gathered):
            per_seed, arm_facts, d_by_subset = gathered[arm]
            merged, lo, hi = canon[arm]
            _, lo2, hi2 = alt[arm]
            mc_lo = abs(lo2 - lo)
            mc_hi = abs(hi2 - hi)
            per_subset = {s: float(v.mean()) for s, v in d_by_subset.items()}
            n_neg = sum(1 for v in per_subset.values() if v < 0)

            per_cell_delta = {f'{s}/{sd}': float(per_seed[s][sd].mean())
                              for s in per_seed for sd in per_seed[s]}
            seed_vals = []
            for sd in seeds:
                vals = [per_cell_delta[f'{s}/{sd}'] for s in SUBSETS
                        if f'{s}/{sd}' in per_cell_delta]
                if vals:
                    seed_vals.append(float(np.mean(vals)))
            seed_sd = float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else float('nan')
            no_fd001 = [per_subset[s] for s in per_subset if s != 'FD001']
            merged_no_fd001 = float(np.mean(no_fd001)) if no_fd001 else float('nan')

            for (subset, seed), (cell, f) in sorted(arm_facts.items()):
                base = base_index[(subset, seed, split)]
                row = {
                    'arm': arm, 'run_id': cell.parent.parent.name,
                    'subset': subset, 'seed': seed, 'split_seed': split,
                    'baseline_run_id': Path(
                        try_resolve(runs, 'E0', subset, seed, split)).parent.parent.name,
                    'clean_mean_roots': round(f['clean_mean_roots'], 6),
                    'baseline_clean_mean_roots': round(base['clean_mean_roots'], 6),
                    'delta_rmse': round(f['clean_mean_roots'] - base['clean_mean_roots'], 6),
                    'best_ep': f['best_ep'], 'elapsed_s': f['elapsed_s'],
                    'code_fingerprint': f['code_fingerprint'],
                }
                for b in BETAS:
                    k = f'noise_beta{b:g}_mean_roots'
                    row[f'noise_beta{b:g}_mean_roots'] = (
                        round(f[k], 6) if k in f else '')
                    row[f'noise_beta{b:g}_degradation'] = (
                        round(f[k] - f['clean_mean_roots'], 6) if k in f else '')
                row.update({
                    'arm_merged_delta': round(merged, 6),
                    'arm_ci_lo': round(lo, 6), 'arm_ci_hi': round(hi, 6),
                    'arm_ci_seed': canon_seed,
                    'arm_mc_endpoint_drift_max': round(max(mc_lo, mc_hi), 6),
                    'arm_ci_lo_alt_seed': round(lo2, 6),
                    'arm_ci_hi_alt_seed': round(hi2, 6),
                    'arm_direction_negative': f'{n_neg}/{len(per_subset)}',
                    'arm_seed_sd': round(seed_sd, 6) if seed_sd == seed_sd else '',
                    'arm_merged_delta_excl_FD001': round(merged_no_fd001, 6),
                    'arm_boot': args.boot,
                    'arm_n_engines': int(sum(len(v) for v in d_by_subset.values())),
                })
                for s in SUBSETS:
                    row[f'arm_delta_{s}'] = (round(per_subset[s], 6)
                                             if s in per_subset else '')
                aug_rows.append(row)

    fields = list(aug_rows[0])
    with open(out_dir / 'SENSOR_AUGMENTATION_RESULTS.csv', 'w', newline='',
              encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(aug_rows)

    print(f'[baseline]     {len(base_rows)} rows -> {out_dir}/SENSOR_BASELINE_RESULTS.csv')
    print(f'[augmentation] {len(aug_rows)} rows -> {out_dir}/SENSOR_AUGMENTATION_RESULTS.csv')
    print(f'\narm-level summary (clean, mean-of-roots, {args.boot} bootstraps):')
    seen = set()
    for r in aug_rows:
        key = (r['arm'], r['split_seed'])
        if key in seen:
            continue
        seen.add(key)
        c1 = r['arm_merged_delta'] <= -0.3
        c2 = r['arm_ci_hi'] < 0
        c3 = int(r['arm_direction_negative'].split('/')[0]) >= 3
        print(f"  {r['arm']:4s} split{r['split_seed']}  merged {r['arm_merged_delta']:+.4f} "
              f"[{r['arm_ci_lo']:+.4f}, {r['arm_ci_hi']:+.4f}]  "
              f"(MC drift {r['arm_mc_endpoint_drift_max']:.4f})  "
              f"dir {r['arm_direction_negative']}  n={r['arm_n_engines']}  "
              f"criteria {'PASS' if (c1 and c2 and c3) else 'FAIL'}"
              f" [le-0.3 {c1}, ci<0 {c2}, dir {c3}]")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
