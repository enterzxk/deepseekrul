"""Post-hoc TEST analysis for Paper B, per TEST_EVAL_PLAN.md.

The criteria were fixed at plan v1.0 before any TEST cell ran; later versions only
corrected wording (v1.1 field mapping, v1.2/v1.3 status lines). This script
implements those criteria and must not be edited to change them after the fact.

Reads each cell's `test_predictions.csv` (clean rows only), pairs per engine
against the same `(subset, seed)` E0 cell, and reports:

  * merged Delta pooled RMSE with an engine-level paired bootstrap CI
  * merged Delta MAE (per-engine |error|) with the same kind of CI
  * bias = mean(y_pred - y_true)

Two things the plan fixes that this file must honour:

1. **Equal weight over the four subsets.** Engine counts are 100/259/100/248;
   pooling engines would let FD002/FD004 dominate the merged number.
2. **Every statistic gets its own rng.** The bootstrap stream must be a
   function of (data, seed, statistic) and nothing else. We already shipped one
   bug where a shared stream made an interval depend on call order.

Seeds are averaged inside a (subset, seed) cell first, then engines are
resampled -- never the other way round (that would inflate n and narrow the CI).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from _cells import resolve                                    # noqa: E402

ARMS = ('E1a', 'E1b', 'E2a')
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = {17: (101, 202, 303), 29: (42, 43, 101, 202, 303)}
BOOT_SEED = {17: 20261115, 29: 20261116}
N_BOOT = 10000


def load_clean(cell: Path):
    """{engine_id: (y_true, y_pred)} for the clean TEST rows of one cell."""
    out = {}
    with open(cell / 'test_predictions.csv', newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('corruption_type') not in ('none', '', None):
                continue
            if row.get('evaluation_protocol') not in ('clean_test', '', None):
                continue
            u = int(row['engine_id'])
            if u in out:
                raise SystemExit(f'{cell}: duplicate engine_id {u} in clean rows')
            out[u] = (float(row['y_true']), float(row['y_pred_evaluated']))
    if not out:
        raise SystemExit(f'{cell}: no clean rows')
    return out


def boot_ci(d_by_subset, n_boot, seed, stat_id):
    """Engine-level bootstrap for the merged Delta MAE (equal weight over subsets).

    `stat_id` only decorrelates this statistic's stream from any other statistic
    computed in the same run; the interval must not depend on what else ran.
    """
    rng = np.random.default_rng([seed, stat_id])
    keys = list(d_by_subset)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for k in keys:
            d = d_by_subset[k]['d']
            idx = rng.integers(0, len(d), len(d))
            vals.append(float(d[idx].mean()))
        draws[b] = float(np.mean(vals))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--boot', type=int, default=N_BOOT)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root = Path(a.root)

    report = {}
    for split in (17, 29):
        seeds = SEEDS[split]
        cells = {}
        for arm in ('E0',) + ARMS:
            for subset in SUBSETS:
                for seed in seeds:
                    cells[(arm, subset, seed)] = load_clean(
                        Path(resolve(root, arm, subset, seed, split)))

        arms_out = {}
        for arm in ARMS:
            d_mae, se_arm, se_e0, n_eng, bias = {}, {}, {}, {}, {}
            per_subset = {}
            for subset in SUBSETS:
                rows = []
                for seed in seeds:
                    e0 = cells[('E0', subset, seed)]
                    ea = cells[(arm, subset, seed)]
                    eng = sorted(set(e0) & set(ea))
                    if len(eng) != len(e0) or len(eng) != len(ea):
                        raise SystemExit(f'{arm} {subset} s{seed}: engine sets differ '
                                         f'(e0={len(e0)} arm={len(ea)} common={len(eng)})')
                    rows.append((eng, e0, ea))
                engs = [r[0] for r in rows]
                if any(e != engs[0] for e in engs):
                    raise SystemExit(f'{arm} {subset}: seeds disagree on engine set')
                eng = engs[0]
                # average per-engine statistics over seeds, THEN resample
                ae_a = np.mean([[abs(r[2][u][1] - r[2][u][0]) for u in eng] for r in rows], axis=0)
                ae_0 = np.mean([[abs(r[1][u][1] - r[1][u][0]) for u in eng] for r in rows], axis=0)
                sq_a = [np.array([(r[2][u][1] - r[2][u][0]) ** 2 for u in eng]) for r in rows]
                sq_0 = [np.array([(r[1][u][1] - r[1][u][0]) ** 2 for u in eng]) for r in rows]
                d_mae[subset] = {'d': ae_a - ae_0, 'n_engines': len(eng)}
                se_arm[subset] = sq_a
                se_e0[subset] = sq_0
                n_eng[subset] = len(eng)
                bias[subset] = float(np.mean([[r[2][u][1] - r[2][u][0] for u in eng] for r in rows]))
                # Delta pooled RMSE: per-seed RMSE difference, then average over seeds.
                # mean-of-per-seed-RMSE, not sqrt-of-seed-mean -- the two differ by ~1e-3.
                per_seed_d = [math.sqrt(x.mean()) - math.sqrt(y.mean()) for x, y in zip(sq_a, sq_0)]
                per_subset[subset] = dict(
                    delta_mae=float(np.mean(ae_a - ae_0)),
                    delta_pooled_rmse=float(np.mean(per_seed_d)),
                    arm_pooled=float(np.mean([math.sqrt(x.mean()) for x in sq_a])),
                    e0_pooled=float(np.mean([math.sqrt(y.mean()) for y in sq_0])),
                    n_engines=len(eng),
                )

            merged_mae = float(np.mean([v['delta_mae'] for v in per_subset.values()]))
            merged_rmse = float(np.mean([v['delta_pooled_rmse'] for v in per_subset.values()]))

            lo_m, hi_m = boot_ci(d_mae, a.boot, BOOT_SEED[split], 0)
            # Delta pooled RMSE is not a per-engine functional, so it needs its own
            # resampling loop over squared errors, with its own stream (stat id 1).
            rng = np.random.default_rng([BOOT_SEED[split], 1])
            keys = list(d_mae)
            draws = np.empty(a.boot)
            for b in range(a.boot):
                vals = []
                for k in keys:
                    n = d_mae[k]['n_engines']
                    idx = rng.integers(0, n, n)
                    va, v0 = se_arm[k], se_e0[k]
                    vals.append(float(np.mean([math.sqrt(x[idx].mean()) - math.sqrt(y[idx].mean())
                                               for x, y in zip(va, v0)])))
                draws[b] = float(np.mean(vals))
            lo_r, hi_r = np.percentile(draws, [2.5, 97.5])
            lo_r, hi_r = float(lo_r), float(hi_r)

            arms_out[arm] = dict(
                merged_delta_mae=merged_mae, mae_ci95=[lo_m, hi_m],
                merged_delta_pooled_rmse=merged_rmse, rmse_ci95=[lo_r, hi_r],
                n_negative=int(sum(1 for v in per_subset.values() if v['delta_mae'] < 0)),
                n_subsets=len(per_subset),
                per_subset=per_subset,
                bias_merged=float(np.mean(list(bias.values()))),
                bias_per_subset=bias,
            )
            r = arms_out[arm]
            print(f'split {split} {arm:4} dMAE {r["merged_delta_mae"]:+.4f} '
                  f'[{lo_m:+.4f},{hi_m:+.4f}]  dRMSE {r["merged_delta_pooled_rmse"]:+.4f} '
                  f'[{lo_r:+.4f},{hi_r:+.4f}]  neg {r["n_negative"]}/4  '
                  f'bias {r["bias_merged"]:+.4f}', flush=True)

        report[str(split)] = dict(
            arms=arms_out, n_boot=a.boot, boot_seed=BOOT_SEED[split],
            n_cells=len(cells), seeds=list(seeds),
        )

    txt = json.dumps(report, indent=1)
    if a.json:
        Path(a.json).write_text(txt, encoding='utf-8')
        print(f'wrote {a.json}')
    else:
        print(txt)


if __name__ == '__main__':
    raise SystemExit(main())
