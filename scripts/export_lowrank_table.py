#!/usr/bin/env python3
"""Export LOW_RANK_SENSOR_RESULTS.csv -- one row per E5 cell (48 rows).

Every number here is recomputed at export time, not copied from an earlier report:

  * `val_engine_mean_roots_rmse` comes from `val_pred.csv` (per-engine RMSE, then mean);
  * `delta_vs_none` is the paired engine-level difference against the SAME-seed
    `E5none` cell, then averaged over engines (never a difference of two means);
  * the arm-level 95% CI is produced by `audit_e5_arms.boot`, i.e. the exact function
    the frozen judgement used, so the CSV cannot drift away from the audited values;
  * `rel` / `n_batches` / `|lam|` / `||W_last||_F` come from the checkpoint through
    the same probe `audit_e5_activation.py` uses.

CI scope is deliberately explicit: the interval is the ARM's merged interval, not a
per-cell interval. A per-cell interval would be a new statistic that no frozen
criterion uses, and printing it next to a per-cell delta invites reading it as a
per-cell verdict. `ci_scope=arm_merged` says so in every row.

Usage
-----
    python scripts/export_lowrank_table.py --runs <runs> --data-dir <dir> \
        --out docs/paper_b_sensor_generalization/LOW_RANK_SENSOR_RESULTS.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_b.sensor_aug.evaluation import _predict  # noqa: E402
from paper_b.training.data import prepare_train  # noqa: E402
from paper_b.training.train import _build  # noqa: E402
from scripts._cells import resolve, try_resolve  # noqa: E402
from scripts.audit_e5_activation import BranchTap, last_g_weight  # noqa: E402
from scripts.audit_e5_arms import boot, roots  # noqa: E402

KEY = 'val_engine_mean_roots_rmse'
ARMS = ('E5none', 'E5lr4', 'E5lr14', 'E5mlp')
SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = (101, 202, 303)
DORMANT_BELOW = 1e-6
N_BOOT = 10000

COLS = [
    'arm', 'subset', 'seed', 'run_id', 'lr_mode', 'params',
    'val_engine_mean_roots_rmse',
    'paired_none_run_id', 'delta_vs_none',
    'arm_merged_delta', 'arm_ci95_lo', 'arm_ci95_hi', 'ci_scope',
    'identical_to_e0',
    'rel', 'n_batches', 'abs_lam', 'w_last_fro', 'branch_verdict',
]


def activation(cell: Path, cfg: dict, data_dir: Path) -> dict:
    """Same probe as audit_e5_activation.py, returning values instead of printing."""
    net = _build(cfg)
    sd = torch.load(cell / 'best.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    lam = abs(float(sd['lr.lam']))
    wkey, wn = last_g_weight(sd)
    tap = BranchTap()
    if not tap.attach(net):
        return dict(rel=None, n_batches=0, abs_lam=lam, w_last_fro=wn, verdict='NO_BRANCH')
    ds = prepare_train(cfg['subset'], data_dir, span=cfg['span'],
                       split_seed=cfg['split_seed'], fold=cfg['fold'],
                       val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                       legacy_rul=cfg['legacy_rul'], global_hi=cfg['global_hi'])
    _predict(net, ds['val'], 'cpu', cfg)
    tap.close()
    rel = tap.rel()
    if rel is None:
        return dict(rel=None, n_batches=tap.n_batches, abs_lam=lam, w_last_fro=wn,
                    verdict='NO_CAPTURE')
    return dict(rel=rel, n_batches=tap.n_batches, abs_lam=lam, w_last_fro=wn,
                verdict='DORMANT' if rel < DORMANT_BELOW else 'AWAKE')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=N_BOOT)
    a = ap.parse_args()

    runs = Path(a.runs)
    data_dir = Path(a.data_dir)

    # ---- per-cell roots, deltas, and the identity check -------------------
    cells, r = {}, {}
    for arm in ARMS:
        for subset in SUBSETS:
            for seed in SEEDS:
                cell = Path(resolve(runs, arm, subset, seed, 17))
                cfg = json.loads((cell / 'config.json').read_text())
                rr = roots(cell / 'val_pred.csv')
                cells[(arm, subset, seed)] = dict(cell=cell, cfg=cfg, roots=rr)
                r[(arm, subset, seed)] = rr

    per_cell_delta = {}
    for arm in ARMS[1:]:
        for subset in SUBSETS:
            for seed in SEEDS:
                ra = r[(arm, subset, seed)]
                r0 = r[('E5none', subset, seed)]
                eng = sorted(set(ra) & set(r0))
                if not eng:
                    raise RuntimeError(f'no shared engines: {arm} {subset} {seed}')
                per_cell_delta[(arm, subset, seed)] = (
                    float(np.mean([ra[u] - r0[u] for u in eng])), len(eng))
        # frozen estimator: average seeds within a subset, then resample engines
        dsub = {}
        for subset in SUBSETS:
            ds = [np.array([r[(arm, subset, sd)][u] - r[('E5none', subset, sd)][u]
                            for u in sorted(set(r[(arm, subset, sd)])
                                            & set(r[('E5none', subset, sd)]))])
                  for sd in SEEDS]
            dsub[subset] = np.mean(np.stack(ds, axis=0), axis=0)
        merged, lo, hi, nneg = boot(dsub, a.boot)
        r[(arm, 'ARM', 0)] = dict(merged=merged, lo=lo, hi=hi, nneg=nneg)

    rows = []
    for arm in ARMS:
        for subset in SUBSETS:
            for seed in SEEDS:
                d = cells[(arm, subset, seed)]
                cell, cfg, rr = d['cell'], d['cfg'], d['roots']
                metrics = json.loads((cell / 'metrics.json').read_text())
                val = metrics.get(KEY)
                # true parameter count from the model, not from the state_dict:
                # a state_dict also holds buffers (BN running stats) and would inflate it.
                params = sum(p.numel() for p in _build(cfg).parameters())

                none_id = cells[('E5none', subset, seed)]['cell'].name
                if arm == 'E5none':
                    delta, neng = 0.0, len(rr)
                    arm_merged, lo, hi = 0.0, 0.0, 0.0
                    e0 = try_resolve(runs, 'E0', subset, seed, 17)
                    same = 'NA'
                    if e0 is not None:
                        m0 = json.loads((Path(e0) / 'metrics.json').read_text())
                        same = 'YES' if m0.get(KEY) == val else 'NO'
                    act = dict(rel=None, n_batches=0, abs_lam=None, w_last_fro=None,
                               verdict='NO_BRANCH')
                else:
                    delta, neng = per_cell_delta[(arm, subset, seed)]
                    agg = r[(arm, 'ARM', 0)]
                    arm_merged, lo, hi = agg['merged'], agg['lo'], agg['hi']
                    same = ''
                    act = activation(cell, cfg, data_dir)

                def f(x, spec='%.6f'):
                    return '' if x is None else (spec % x)

                rows.append({
                    'arm': arm, 'subset': subset, 'seed': seed,
                    'run_id': cell.name, 'lr_mode': cfg.get('lr_mode'),
                    'params': params,
                    'val_engine_mean_roots_rmse': f(val),
                    'paired_none_run_id': none_id,
                    'delta_vs_none': f(delta),
                    'arm_merged_delta': f(arm_merged),
                    'arm_ci95_lo': f(lo), 'arm_ci95_hi': f(hi),
                    'ci_scope': 'arm_merged',
                    'identical_to_e0': same,
                    'rel': f(act['rel'], '%.6e'),
                    'n_batches': act['n_batches'],
                    'abs_lam': f(act['abs_lam'], '%.6e'),
                    'w_last_fro': f(act['w_last_fro'], '%.6e'),
                    'branch_verdict': act['verdict'],
                })
                print('%-7s %-6s %d  delta %+.4f  rel %s  %s'
                      % (arm, subset, seed, delta, f(act['rel'], '%.3e') or 'NA',
                         act['verdict']), flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print()
    print('wrote %s  rows=%d' % (out, len(rows)))
    print('arms:', {k: sum(1 for x in rows if x['arm'] == k) for k in ARMS})
    print('identity YES:', sum(1 for x in rows if x['identical_to_e0'] == 'YES'),
          '/', sum(1 for x in rows if x['identical_to_e0'] in ('YES', 'NO')))
    print('branch verdicts:', {v: sum(1 for x in rows if x['branch_verdict'] == v)
                               for v in ('AWAKE', 'DORMANT', 'NO_CAPTURE', 'NO_BRANCH')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
