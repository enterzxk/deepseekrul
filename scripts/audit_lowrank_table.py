#!/usr/bin/env python3
"""Audit LOW_RANK_SENSOR_RESULTS.csv against the frozen values.

This is a regression guard, not a recomputation of everything: the per-engine
recomputation lives in `export_lowrank_table.py` (it needs the checkpoints and the
data). What this script does is refuse to let the CSV drift silently away from:

  * the E0 cells it claims to reproduce (`identical_to_e0`),
  * the arm-level aggregates that `audit_e5_arms.py` produced and the judgement used,
  * the frozen parameter counts,
  * the branch-activity outcome.

Every assertion is single-match: a duplicate or missing key is a hard failure, never
`sorted()[0]`. That rule exists because a prefix glob once silently mixed split 17 and
split 29 E0 cells (7/12 "identical") and another check reported `mismatch=0` after an
exception had skipped every comparison.

Usage
-----
    python scripts/audit_lowrank_table.py \
        --lowrank docs/paper_b_sensor_generalization/LOW_RANK_SENSOR_RESULTS.csv \
        --baseline docs/paper_b_sensor_generalization/SENSOR_BASELINE_RESULTS.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

# Values produced by `audit_e5_arms.py` (per-interval RNG) and used by the stage-5
# judgement. Hardcoded on purpose: the checker's job is to fail when the CSV stops
# agreeing with what was judged, so it must not import the same code that wrote it.
FROZEN = {
    'E5lr4':  (0.671359, 0.240082, 1.112929),
    'E5lr14': (-0.035231, -0.402342, 0.340542),
    'E5mlp':  (0.183697, -0.182698, 0.553866),
}
FROZEN_PARAMS = {'E5none': 115651, 'E5lr4': 124028, 'E5lr14': 124168, 'E5mlp': 123972}
ARMS = ('E5none', 'E5lr4', 'E5lr14', 'E5mlp')
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = ('101', '202', '303')
TOL = 1e-6


def fail(msg: str) -> None:
    print('FAIL:', msg)
    raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lowrank', required=True)
    ap.add_argument('--baseline', required=True)
    a = ap.parse_args()

    low = list(csv.DictReader(open(a.lowrank, encoding='utf-8')))
    base = list(csv.DictReader(open(a.baseline, encoding='utf-8')))

    # ---- structure -------------------------------------------------------
    if len(low) != 48:
        fail(f'expected 48 rows, got {len(low)}')
    seen = {}
    for r in low:
        k = (r['arm'], r['subset'], r['seed'])
        if k in seen:
            fail(f'duplicate row {k}')
        seen[k] = r
    for arm in ARMS:
        for s in SUBSETS:
            for sd in SEEDS:
                if (arm, s, sd) not in seen:
                    fail(f'missing row {(arm, s, sd)}')
    print('structure: 48 unique rows, 4 arms x 4 subsets x 3 seeds')

    # ---- params ----------------------------------------------------------
    for arm, want in FROZEN_PARAMS.items():
        got = {int(seen[(arm, s, sd)]['params']) for s in SUBSETS for sd in SEEDS}
        if got != {want}:
            fail(f'{arm} params {got} != frozen {want}')
    print('params:', {k: v for k, v in FROZEN_PARAMS.items()})

    # ---- E5none reproduces E0 (split 17 only, single match) --------------
    e0 = [r for r in base if r['run_id'].startswith('E0-') and r['split_seed'] == '17']
    idx = {}
    for r in e0:
        k = (r['subset'], r['seed'])
        if k in idx:
            fail(f'duplicate E0 split-17 row for {k}; refusing to guess which one is meant')
        idx[k] = r
    if len(idx) != 12:
        fail(f'expected 12 E0 split-17 cells, got {len(idx)}')
    bad = 0
    for s in SUBSETS:
        for sd in SEEDS:
            r = seen[('E5none', s, sd)]
            if r['identical_to_e0'] != 'YES':
                fail(f'E5none {s}/{sd} not marked identical_to_e0=YES')
            want = round(float(idx[(s, sd)]['val_engine_mean_roots_rmse_recorded']), 6)
            got = round(float(r['val_engine_mean_roots_rmse']), 6)
            if got != want:
                print(f'  DIFF {s}/{sd}: {got} vs E0 {want}')
                bad += 1
    if bad:
        fail(f'{bad} E5none cells disagree with E0')
    print('identity: E5none == E0 (split 17) on the recorded metric, 12/12')

    # ---- arm aggregates vs what was judged -------------------------------
    for arm, (m, lo, hi) in FROZEN.items():
        vals = {(round(float(seen[(arm, s, sd)]['arm_merged_delta']), 6),
                 round(float(seen[(arm, s, sd)]['arm_ci95_lo']), 6),
                 round(float(seen[(arm, s, sd)]['arm_ci95_hi']), 6))
                for s in SUBSETS for sd in SEEDS}
        if len(vals) != 1:
            fail(f'{arm}: arm-level aggregate is not constant across its rows: {vals}')
        gm, glo, ghi = vals.pop()
        if abs(gm - m) > TOL or abs(glo - lo) > TOL or abs(ghi - hi) > TOL:
            fail(f'{arm}: CSV aggregate {(gm, glo, ghi)} != judged {(m, lo, hi)}')
    print('aggregates: match the values audit_e5_arms.py produced and the judgement used')

    # ---- branch activity -------------------------------------------------
    verdicts = {}
    for r in low:
        verdicts[r['branch_verdict']] = verdicts.get(r['branch_verdict'], 0) + 1
    if verdicts != {'AWAKE': 36, 'NO_BRANCH': 12}:
        fail(f'branch verdicts {verdicts} != {{AWAKE: 36, NO_BRANCH: 12}}')
    print('activity: AWAKE 36 / NO_BRANCH 12')

    print()
    print('LOW_RANK_SENSOR_RESULTS.csv PASSED all consistency checks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
