#!/usr/bin/env python3
"""Export the two result tables reviewers need to recompute everything.

SENSOR_BASELINE_RESULTS.csv      one row per E0 cell (discovery + confirmation)
SENSOR_AUGMENTATION_RESULTS.csv  one row per augmented cell, with its paired E0
                                 baseline and the arm-level statistics repeated
                                 on every row of that arm (so a reviewer can filter
                                 by arm and read the CI without a second file)

Everything is recomputed from the per-engine predictions (`val_pred.csv` /
`val_noise_predictions.csv`), not copied from summary JSONs, except the fields that
only exist in metadata (`best_ep`, `elapsed_s`, `code_fingerprint`).

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
        # This file has a different schema from val_pred.csv: the prediction column
        # is y_pred_evaluated (the noise-stress protocol output), and the engine key
        # is engine_id. Same column audit_noise_pressure.py reads, so the numbers
        # here match the published noise-pressure table.
        rows = read_rows(nz)
        for b in BETAS:
            sel = [r for r in rows if abs(float(r['corruption_level']) - b) < 1e-9]
            if sel:
                pe = per_engine_roots(sel, 'y_pred_evaluated')
                out[f'noise_beta{b:g}_per_engine'] = pe
                out[f'noise_beta{b:g}_mean_roots'] = mean_of_roots(pe)
    return out


def bootstrap_merged(diffs_by_subset, n_boot=10000, seed=20261116):
    """diffs_by_subset: {subset: np.array over engines}. Merged = mean of subset means."""
    rng = np.random.default_rng(seed)
    keys = list(diffs_by_subset)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        vals = [diffs_by_subset[k][rng.integers(0, len(diffs_by_subset[k]),
                                                len(diffs_by_subset[k]))].mean()
                for k in keys]
        draws[i] = float(np.mean(vals))
    point = float(np.mean([diffs_by_subset[k].mean() for k in keys]))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


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
    base_index = {}          # (subset, seed, split_seed) -> facts
    for split, seeds in ((17, DISCOVERY_SEEDS), (29, CONFIRM_SEEDS)):
        for subset in SUBSETS:
            for seed in seeds:
                cell = try_resolve(runs, 'E0', subset, seed, split)
                if cell is None:
                    continue
                f = cell_facts(Path(cell))
                base_index[(subset, seed, split)] = f
                base_rows.append({
                    'run_id': Path(cell).parent.name,
                    'subset': subset, 'seed': seed, 'split_seed': split,
                    'val_engine_mean_roots_rmse': round(f['clean_mean_roots'], 6),
                    'val_engine_mean_roots_rmse_recorded': f['metrics_mean_roots'],
                    'best_ep': f['best_ep'], 'elapsed_s': f['elapsed_s'],
                    'code_fingerprint': f['code_fingerprint'],
                    'n_engines': len(f['clean_per_engine']),
                })
    with open(out_dir / 'SENSOR_BASELINE_RESULTS.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(base_rows[0]))
        w.writeheader()
        w.writerows(base_rows)

    # ------------------------------------------------------------ augmentation
    aug_rows = []
    for split, seeds, arms in ((17, DISCOVERY_SEEDS, AUG_ARMS),
                               (29, CONFIRM_SEEDS, CONFIRM_AUG_ARMS)):
        for arm in arms:
            diffs = defaultdict(list)
            arm_facts = {}
            for subset in SUBSETS:
                for seed in seeds:
                    cell = try_resolve(runs, arm, subset, seed, split)
                    base = base_index.get((subset, seed, split))
                    if cell is None or base is None:
                        continue
                    f = cell_facts(Path(cell))
                    arm_facts[(subset, seed)] = (Path(cell), f)
                    common = sorted(set(f['clean_per_engine']) & set(base['clean_per_engine']))
                    d = np.array([f['clean_per_engine'][u] - base['clean_per_engine'][u]
                                  for u in common])
                    diffs[subset].append(d)
            if not diffs:
                continue
            stacked = {s: np.concatenate(v) for s, v in diffs.items()}
            merged, lo, hi = bootstrap_merged(stacked, args.boot)
            per_subset = {s: float(v.mean()) for s, v in stacked.items()}
            n_neg = sum(1 for v in per_subset.values() if v < 0)
            seed_means = {}
            for subset in SUBSETS:
                for i, seed in enumerate(seeds):
                    if (subset, seed) in arm_facts:
                        seed_means.setdefault(seed, []).append(float(diffs[subset][i].mean()))
            seed_vals = [float(np.mean(v)) for v in seed_means.values()]
            seed_sd = float(np.std(seed_vals, ddof=1)) if len(seed_vals) > 1 else float('nan')
            no_fd001 = [per_subset[s] for s in per_subset if s != 'FD001']
            merged_no_fd001 = float(np.mean(no_fd001)) if no_fd001 else float('nan')

            for (subset, seed), (cell, f) in sorted(arm_facts.items()):
                base = base_index[(subset, seed, split)]
                row = {
                    'arm': arm, 'run_id': cell.parent.name,
                    'subset': subset, 'seed': seed, 'split_seed': split,
                    'baseline_run_id': Path(
                        try_resolve(runs, 'E0', subset, seed, split)).parent.name,
                    'clean_mean_roots': round(f['clean_mean_roots'], 6),
                    'baseline_clean_mean_roots': round(base['clean_mean_roots'], 6),
                    'delta_rmse': round(f['clean_mean_roots'] - base['clean_mean_roots'], 6),
                    'best_ep': f['best_ep'], 'elapsed_s': f['elapsed_s'],
                    'code_fingerprint': f['code_fingerprint'],
                }
                for b in BETAS:
                    k = f'noise_beta{b:g}_mean_roots'
                    kb = f'noise_beta{b:g}_mean_roots'
                    row[f'noise_beta{b:g}_mean_roots'] = (
                        round(f[k], 6) if k in f else '')
                    row[f'noise_beta{b:g}_degradation'] = (
                        round(f[k] - f['clean_mean_roots'], 6) if k in f else '')
                row.update({
                    'arm_merged_delta': round(merged, 6),
                    'arm_ci_lo': round(lo, 6), 'arm_ci_hi': round(hi, 6),
                    'arm_direction_negative': f'{n_neg}/{len(per_subset)}',
                    'arm_seed_sd': round(seed_sd, 6) if seed_sd == seed_sd else '',
                    'arm_merged_delta_excl_FD001': round(merged_no_fd001, 6),
                    'arm_boot': args.boot,
                    'arm_n_engines': int(sum(len(v) for v in stacked.values())),
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
    print('\narm-level summary (clean, mean-of-roots, engine bootstrap):')
    seen = set()
    for r in aug_rows:
        key = (r['arm'], r['split_seed'])
        if key in seen:
            continue
        seen.add(key)
        print(f"  {r['arm']:4s} split{r['split_seed']}  merged {r['arm_merged_delta']:+.4f} "
              f"[{r['arm_ci_lo']:+.4f}, {r['arm_ci_hi']:+.4f}]  "
              f"dir {r['arm_direction_negative']}  n={r['arm_n_engines']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
