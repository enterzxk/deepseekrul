"""Run the N0-N3 LOEO grid for DS02 and write one ledger row per cell.

Cell counts (schema §1.2): N0 = 6 folds x 1 (no seed axis) = 6; N1/N2/N3 =
6 folds x 3 seeds = 54; total 60 dev cells.

Per cell this writes, under ``experiments/ncmapss/<run_id>/``:

    pred.csv             unit, cycle, y_true_raw, y_true_cap125, y_pred
                         exactly one row per (unit, cycle); no bare `y_true`
                         column, so the criterion is always chosen explicitly
    unit_lifetime.json   unit -> max RUL over that unit's complete trajectory
                         (needed by the relative segment rule; never a feature)
    curve.csv            per-epoch train / holdout loss
    config.json          the exact config used
    metrics.json         provisional metrics (the authoritative table is
                         recomputed by evaluation.py from pred.csv)

and appends one row per cell to ``docs/paper_b_ncmapss/registry/runs.csv``.

A failing cell is recorded as ``status=failed`` with the reason in
``registry_note`` and the grid continues; nothing is retried until it succeeds,
because "which cells succeeded" must stay visible.

Note on N0. N0 is a small single-input MLP (4,353 parameters), not a
parameter-free rule. That is deliberate: a stronger clock baseline makes a DSPT
win meaningful rather than a straw man. Because it is trained, its seed is
fixed at 101 for every fold and recorded, so the cell is deterministic given
the data while still having a seed axis of exactly one. The schema line that
says N0 is parameter-free should be read as "no seed axis", not "no parameters".

    python -m paper_b.ncmapss.run_loeo --h5 data/NCMAPSS/N-CMAPSS_DS02-006.h5
    python -m paper_b.ncmapss.run_loeo --synthetic --folds 2   # plumbing check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import DEV_UNITS
from .features import SPAN
from .train import fit_rung, load_ds02, load_frozen

RUNGS = ('N0', 'N1', 'N2', 'N3')
SEEDS = (101, 202, 303)
N0_SEED = 101
CAP = 125.0
SEG_REL = (1 / 3, 2 / 3)
SEG_ABS = (75.0, 25.0)          # early >= 75, mid >= 25, late < 25 (RUL cycles)

LEDGER_COLS = """run_id phase rung model fold held_out_engine n_train_engines seed
git_sha code_md5 data_sha256 split_hash protocol_blob train_plan_sha256 feature_set feature_list
kmeans_fit_units kmeans_seed config_path config_sha256 freeze_source freeze_source_sha256 selection_rule epochs_run
best_epoch ckpt_path ckpt_sha256 preds_path pred_instant n_units n_flights rmse_raw_unit
mae_raw_unit bias_raw_unit nasa_raw_unit rmse_cap125_unit mae_cap125_unit bias_cap125_unit nasa_cap125_unit rmse_raw_flightpooled
rmse_cap125_flightpooled seg_early_relative_capped seg_early_relative_raw seg_early_absolute_capped seg_early_absolute_raw seg_mid_relative_capped seg_mid_relative_raw seg_mid_absolute_capped
seg_mid_absolute_raw seg_late_relative_capped seg_late_relative_raw seg_late_absolute_capped seg_late_absolute_raw seg_boundaries freeze_id freeze_ts_utc
eval_start_utc status audit_verdict independent_unit registry_note audited_by""".split()


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _nasa(e):
    a = np.where(e < 0, 1.0 / 13.0, 1.0 / 10.0)
    return float(np.exp(a * np.abs(e)).sum())


def _rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def metrics(y_true_raw, y_pred, lifetime):
    """Provisional metrics; evaluation.py recomputes the authoritative table."""
    y = np.asarray(y_true_raw, np.float64)
    p = np.asarray(y_pred, np.float64)
    e = p - y
    yc = np.minimum(y, CAP)
    ec = p - yc
    out = dict(
        n_flights=int(len(y)),
        rmse_raw_unit=_rmse(e), mae_raw_unit=float(np.abs(e).mean()),
        bias_raw_unit=float(e.mean()), nasa_raw_unit=_nasa(e),
        rmse_cap125_unit=_rmse(ec), mae_cap125_unit=float(np.abs(ec).mean()),
        bias_cap125_unit=float(ec.mean()), nasa_cap125_unit=_nasa(ec),
        rmse_raw_flightpooled=_rmse(e), rmse_cap125_flightpooled=_rmse(ec),
    )
    frac = y / np.maximum(lifetime, 1e-9)
    for name, mask in (('early', frac > SEG_REL[1]), ('mid', (frac > SEG_REL[0]) & (frac <= SEG_REL[1])),
                       ('late', frac <= SEG_REL[0])):
        for crit, err in (('capped', ec), ('raw', e)):
            out[f'seg_{name}_relative_{crit}'] = _rmse(err[mask]) if mask.any() else float('nan')
    for name, mask in (('early', y >= SEG_ABS[0]), ('mid', (y < SEG_ABS[0]) & (y >= SEG_ABS[1])),
                       ('late', y < SEG_ABS[1])):
        for crit, err in (('capped', ec), ('raw', e)):
            out[f'seg_{name}_absolute_{crit}'] = _rmse(err[mask]) if mask.any() else float('nan')
    return out


def _write_ledger(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow(row)


def _preflight(root):
    """Record the protocol anchor and the working-tree state before any cell runs."""
    import subprocess
    def run(*a):
        return subprocess.run(a, cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(git_sha=run('git', 'rev-parse', 'HEAD'),
                protocol_blob=run('git', 'rev-parse',
                                  'HEAD:docs/paper_b_ncmapss/DATASET_PROTOCOL.md'))


def run_fold(rung, fold_idx, held_out, cfg, data, out_root, ledger, pre, data_sha):
    seed = N0_SEED if rung == 'N0' else None
    seeds = (N0_SEED,) if rung == 'N0' else SEEDS
    rows = []
    for sd in seeds:
        run_id = f'{rung}_LOEO-{held_out}_s{sd}' if rung != 'N0' else f'{rung}_LOEO-{held_out}'
        d = Path(out_root) / run_id
        row = {c: '' for c in LEDGER_COLS}
        row.update(run_id=run_id, phase='dev', rung=rung,
                   model=dict(N0='age_only', N1='vanilla', N2='dspt_base',
                              N3='dspt_frozen')[rung],
                   fold=f'LOEO-{held_out}', held_out_engine=held_out,
                   n_train_engines=5, seed=(sd if rung != 'N0' else ''),
                   git_sha=pre['git_sha'], protocol_blob=pre['protocol_blob'],
                   train_plan_sha256=_sha256(cfg['_root'] / 'paper_b/configs/protocol.json'),
                   feature_set='WS14+W4+CYC',
                   feature_list=';'.join(['Xs_mean', 'Xs_std', 'Xs_slope', 'W_mean', 'W_std', 'age_scaled']),
                   kmeans_fit_units='' if rung in ('N0', 'N1') else
                   ';'.join(str(u) for u in DEV_UNITS if u != held_out),
                   kmeans_seed='' if rung in ('N0', 'N1') else cfg['seeds'][0],
                   pred_instant='flight_end', n_units=1,
                   seg_boundaries=json.dumps({'relative': list(SEG_REL), 'absolute': list(SEG_ABS)}),
                   independent_unit='engine', status='running')
        if rung == 'N3':
            row.update(freeze_source='paper_b/results/FREEZE_ABLATION_SEED101.json;'
                                     'paper_b/configs/protocol.json',
                       freeze_source_sha256=json.dumps(cfg['provenance']))
        try:
            dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            net, res = fit_rung(rung, data, cfg, seed=sd, device=dev)
            d.mkdir(parents=True, exist_ok=True)
            # predictions on the held-out unit
            ho = ~np.asarray(data['is_train'], bool)
            x = torch.from_numpy(data['x'][ho]).float()
            mask = torch.from_numpy(data['mask'][ho]).float()
            cond = torch.from_numpy(data['cond'][ho])
            if rung == 'N0':
                x = x[..., -1:]
            net.eval()
            with torch.no_grad():
                pred = net(x.to(dev), mask=mask.to(dev), cond=cond.to(dev))['prediction']
            pred = pred.reshape(-1).cpu().numpy().astype(np.float64)
            y_raw = np.asarray(data['y'][ho], np.float64)
            unit = np.asarray(data['unit'][ho])
            cyc = np.asarray(data['cycle'][ho])
            if len(pred) != len(y_raw):
                raise RuntimeError(f'prediction/label length mismatch {len(pred)} vs {len(y_raw)}')
            import pandas as pd
            # exactly five columns: no bare `y_true`, because that name would
            # silently carry the capped label and become a default criterion.
            pd.DataFrame(dict(unit=unit, cycle=cyc, y_true_raw=y_raw,
                              y_true_cap125=np.minimum(y_raw, CAP),
                              y_pred=pred)).to_csv(d / 'pred.csv', index=False)
            (d / 'unit_lifetime.json').write_text(json.dumps(data['unit_lifetime'], indent=1))
            pd.DataFrame(res['history']).to_csv(d / 'curve.csv', index=False)
            (d / 'config.json').write_text(json.dumps(
                {k: v for k, v in cfg.items() if not k.startswith('_')}, indent=1, default=str))
            m = metrics(y_raw, pred, data['unit_lifetime'][int(held_out)])
            (d / 'metrics.json').write_text(json.dumps(m, indent=1))
            row.update(m, status='ok', epochs_run=res['epochs'],
                       selection_rule=f"fixed_epochs:{res['epochs']}",
                       preds_path=str(d / 'pred.csv'), config_path=str(d / 'config.json'),
                       config_sha256=_sha256(d / 'config.json'),
                       registry_note='metrics provisional; recompute with evaluation.py')
        except Exception as exc:  # noqa: BLE001 - a failed cell must be recorded, not hidden
            row.update(status='failed',
                       registry_note=f'{type(exc).__name__}: {exc}',
                       epochs_run='')
            (Path(out_root) / run_id).mkdir(parents=True, exist_ok=True)
            (Path(out_root) / run_id / 'error.txt').write_text(traceback.format_exc())
        _write_ledger(ledger, row)
        rows.append(row)
        print(f'{run_id}: {row["status"]}'
              + (f'  rmse_raw={row.get("rmse_raw_unit"):.3f}' if row['status'] == 'ok' else ''))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--folds', type=int, default=0, help='limit folds for a plumbing check')
    ap.add_argument('--out-root', default='experiments/ncmapss')
    ap.add_argument('--ledger', default='docs/paper_b_ncmapss/registry/runs.csv')
    ap.add_argument('--data-sha256', default='')
    a = ap.parse_args()
    cfg = load_frozen()
    cfg['_root'] = Path(__file__).resolve().parents[2]
    pre = _preflight(cfg['_root'])
    print('preflight:', pre)

    if a.synthetic:
        import tempfile
        from .test_contract import _write_fake
        p = Path(tempfile.mkdtemp()) / 'fake_ds02.h5'
        _write_fake(p, units=tuple(DEV_UNITS) + (11, 14, 15), cycles=80, rows_per_cycle=12)
    else:
        if not a.h5:
            raise SystemExit('--h5 is required unless --synthetic')
        p = Path(a.h5)

    folds = list(DEV_UNITS)
    if a.folds:
        folds = folds[:a.folds]
    all_rows = []
    for k, held in enumerate(folds, 1):
        train_units = [u for u in DEV_UNITS if u != held]
        data = load_ds02(p, train_units, cfg)
        print(f'fold {k}/{len(folds)}: holdout {held}, train {train_units}, '
              f'{data["x"].shape[0]} flights, drops {data["drop_info"]["n_dropped"]}')
        for rung in RUNGS:
            all_rows += run_fold(rung, k, held, cfg, data, a.out_root, a.ledger, pre,
                                 a.data_sha256 or ('SYNTHETIC' if a.synthetic else _sha256(p)))
    ok = sum(r['status'] == 'ok' for r in all_rows)
    print(f'done: {ok}/{len(all_rows)} ok, ledger {a.ledger}')


if __name__ == '__main__':
    main()
