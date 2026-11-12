"""Run the N0-N3 LOEO grid for one N-CMAPSS subset and write one ledger row per
evaluation point.

The subset is not fixed here: the official dev/test split is read from the HDF5
(``data.official_split``) and the folds come from ``evaluation.loeo_folds``, so
the same entry point runs DS02, DS05, DS06 and DS07 without a unit list stored
in code. Cell counts per subset: N0 = 6 folds x 1 (no seed axis) = 6; N1/N2/N3 =
6 folds x 3 seeds = 54; total 60 dev cells = 120 ledger rows.

Per cell this writes, under ``experiments/ncmapss/<subset>/<run_id>_iv/`` and
``..._f200/``:

    pred.csv             unit, cycle, y_true_raw, y_true_cap125, y_pred
                         exactly one row per (unit, cycle); no bare `y_true`
                         column, so the criterion is always chosen explicitly
    ckpt_iv.pt /         the weights that produced this row, so the number can be
    ckpt_f200.pt         verified rather than retrained (one per evaluation point:
                         the two rows come from two different snapshots)
    unit_lifetime.json   unit -> max RUL over that unit's complete trajectory
                         (needed by the relative segment rule; never a feature)
    curve.csv            per-epoch train / holdout loss (the full training run)
    config.json          the exact config used, plus the epoch evaluated
    metrics.json         provisional metrics (the authoritative table is
                         recomputed by evaluation.py from pred.csv)

**Two evaluation points per cell, one training run.** The model is trained once
to ``max(epoch*, 200)`` passes and the weights are snapshotted at ``epoch*`` (the
inner-selection stopping point) and at 200 passes (the fixed endpoint). Each
snapshot is scored separately, so every cell yields two ledger rows:

    <run_id>_iv     selection_rule=inner_val          (epoch*)
    <run_id>_f200   selection_rule=fixed_passes:200   (200 passes)

The ``iv``/``f200`` stop tokens are the ones the frozen template specifies. An
earlier batch used ``_ev<epoch>``, which collapses the two rows into one
directory whenever ``epoch* == 200``; the four-subset batch does not repeat it.

Neither point is preferred. Measured on DS02 fold 1 (2026-11-12), ``epoch*``
lands in 20-94 while the outer curve's own minimum sits at 157-198, a systematic
~2x disagreement; reporting both makes that disagreement visible instead of
hiding it behind one number. The outer holdout curve is diagnostic only and
never selects anything.

and appends one row per evaluation point to
``experiments/ncmapss/<subset>/runs_dev.csv`` (or ``runs_diag.csv`` for a
diagnostic arm). The sealed per-subset tables under ``docs/paper_b_ncmapss/``
are written by the registry owner from these run outputs, not by this script.

**Diagnostic arms are routed elsewhere.** ``DIAG_RUNGS`` (currently ``N1c``) are
not pre-registered and therefore never enter that frozen-grid table: their rows
go to ``experiments/ncmapss/diag/runs_diag.csv`` with ``phase=diag`` and their
artifacts under ``experiments/ncmapss/diag/``. The routing is applied in
``main()`` from the rung name, not from a flag, so no invocation can put a
diagnostic arm into the main ledger by accident.

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

from .data import official_split
from .evaluation import loeo_folds
from .features import SPAN
from .train import build, fit_rung, load_frozen, load_split, load_subset

RUNGS = ('N0', 'N1', 'N2', 'N3')
#: diagnostic arm, never part of the 60-cell grid: vanilla + the operating
#: condition embedding DSPT gets, used to split the N1-vs-N2 gap into
#: "tokenisation" and "condition id" (manager request, 2026-11-12)
DIAG_RUNGS = ('N1c',)
MODEL_NAMES = {'N0': 'age_only', 'N1': 'vanilla', 'N1c': 'vanilla_cond',
               'N2': 'dspt_base', 'N3': 'dspt_frozen'}
SEEDS = (101, 202, 303)
N0_SEED = 101
CAP = 125.0
SEG_REL = (1 / 3, 2 / 3)
SEG_ABS = (75.0, 25.0)          # early >= 75, mid >= 25, late < 25 (RUL cycles)
# DS02 budget (protocol 4.2): 200 passes with batch=min(64,N) gives 6 updates per
# pass, i.e. ~1200 parameter updates, the same order as C-MAPSS's ~1560. 40 passes
# (240 updates) leaves every rung under-trained; 800 starts memorising (N2 13.99).
PASSES = 200
BATCH = 64
#: the fixed endpoint reported next to the inner-selection epoch*. Both points
#: come from the same training run (one fit, two snapshots), so a difference
#: between the two rows is a difference of stopping point, not of initialisation.
FIXED_EVAL_PASSES = PASSES
#: frozen protocol anchor; checked by content hash so it works on a host that
#: cannot reach GitHub (3090 currently cannot).
PROTOCOL_PATH = 'docs/paper_b_ncmapss/DATASET_PROTOCOL.md'
FREEZE_TEMPLATE = 'docs/paper_b_ncmapss/registry/FREEZE_TEMPLATE.json'


def expected_protocol_blob(root):
    """Read the frozen anchor from the ledger template.

    Hard-coding it here meant every anchor move silently broke the preflight
    with a stale constant; the template is already the single place the anchor
    is recorded, so read it there.
    """
    import json
    tpl = json.loads((root / FREEZE_TEMPLATE).read_text(encoding='utf-8'))
    candidates = [tpl, tpl.get('repo') or {}, tpl.get('provenance') or {},
                  tpl.get('freeze_source') or {}]
    for block in candidates:
        for key in ('known_protocol_blob_at_freeze', 'protocol_blob'):
            val = block.get(key) if isinstance(block, dict) else None
            if isinstance(val, str) and len(val) == 40:
                return val
    raise SystemExit('freeze template has no protocol blob; refusing to run')

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
    """Provisional metrics; evaluation.py recomputes the authoritative table.

    ``lifetime`` is the relative-segment denominator: a scalar for a single-engine
    fold, or one value per row for a split with several engines (the test split).
    """
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
    life = np.broadcast_to(np.asarray(lifetime, np.float64), y.shape)
    frac = y / np.maximum(life, 1e-9)
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
    """Record the working-tree state and verify the frozen protocol anchor.

    The anchor is checked with ``git hash-object`` on the file itself, not with
    ``git rev-parse HEAD:<path>``: the compute host may be unable to fetch, and
    a content hash answers the question that matters (is this the frozen text?)
    without depending on the checkout's commit.
    """
    import subprocess
    def run(*a):
        return subprocess.run(a, cwd=root, capture_output=True, text=True).stdout.strip()
    blob = run('git', 'hash-object', PROTOCOL_PATH)
    expected = expected_protocol_blob(root)
    if blob != expected:
        raise SystemExit(
            f'protocol anchor mismatch: {PROTOCOL_PATH} has blob {blob}, '
            f'frozen anchor is {expected}. Refusing to run.')
    return dict(git_sha=run('git', 'rev-parse', 'HEAD'), protocol_blob=blob)


def _subset(data, units):
    """Restrict a fold's windows to a set of units (windows never cross units)."""
    m = np.isin(data['unit'], list(units))
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(m):
            out[k] = v[m]
        else:
            out[k] = v
    return out


def select_epoch_star(rung, data, cfg, seed, dev, batch):
    """Inner-fold epoch selection; the outer held-out unit is never involved.

    For each of the outer fold's five training engines: hold it out, train on the
    other four, record the full validation curve. ``epoch*`` is the argmin of the
    curve **averaged over the five inner folds**, floored at 20 passes.

    Averaging rather than taking the median of the five per-fold minima: with
    four training engines the per-fold validation loss bottoms out at epoch 1-12
    (the model memorises the four and predicts the fifth worse), so the median of
    the minima collapses to a near-untrained model (measured: N2 5.29 -> 18.09).
    The per-fold minima are kept for audit.
    """
    outer_train = sorted({int(u) for u in data['unit'][np.asarray(data['is_train'], bool)]})
    curves, per_fold_best, per_fold_min = [], [], []
    for u in outer_train:
        inner = [x for x in outer_train if x != u]
        sub = _subset(data, inner + [u])
        sub['is_train'] = np.isin(sub['unit'], inner)
        _, rep = fit_rung(rung, sub, cfg, seed=seed, device=dev, batch=batch,
                          early_stop=True, max_epochs=400, min_epochs=20, patience=12)
        c = [h['holdout_loss'] for h in rep['history']]
        curves.append(c)
        per_fold_best.append(int(rep['best_epoch']))
        per_fold_min.append(float(min(c)))
    L = max(len(c) for c in curves)
    avg = np.array([np.mean([c[e] if e < len(c) else c[-1] for c in curves])
                    for e in range(L)])
    raw_argmin = int(np.argmin(avg) + 1)
    epoch_star = max(raw_argmin, 20)
    return dict(epoch_star=epoch_star, raw_argmin=raw_argmin,
                floor_applied=bool(raw_argmin < 20),
                per_fold_best_epoch=per_fold_best,
                per_fold_min_loss=per_fold_min,
                avg_curve=[float(x) for x in avg])


def _predict_held_out(rung, cfg, data, dev, state, target_mu, target_sd):
    """Load one snapshot and predict the held-out unit's flight-end RUL."""
    net = build(rung, cfg).to(dev)
    net.load_state_dict(state)
    net.eval()
    ho = ~np.asarray(data['is_train'], bool)
    x = torch.from_numpy(data['x'][ho]).float()
    mask = torch.from_numpy(data['mask'][ho]).float()
    cond = torch.from_numpy(data['cond'][ho])
    if rung == 'N0':
        x = x[..., -1:]
    with torch.no_grad():
        pred = net(x.to(dev), mask=mask.to(dev), cond=cond.to(dev))['prediction']
    pred = pred.reshape(-1).cpu().numpy().astype(np.float64)
    # invert the training-fold target standardisation before anything is written
    return pred * target_sd + target_mu


def _subset_name(path):
    """``N-CMAPSS_DS05.h5`` -> ``DS05``; falls back to the file stem."""
    import re
    m = re.search(r'(DS\d+)', Path(path).stem, re.I)
    return m.group(1).upper() if m else Path(path).stem


def _write_subset_audit(path, subset, out_root, dev_units, test_units):
    """Per-subset data audit, written before any training happens.

    The dataset table in the paper needs a machine-readable source produced by
    the same run that used the numbers: the official split, flights, rows, rows
    per flight and the label range, all read from the file rather than retyped.
    ``capped_active`` is derived, not assumed: it says whether any unit's raw RUL
    exceeds the cap, which is what decides if the capped criterion is a second
    criterion or the same one under another name.
    """
    from .data import load_aux, load_labels
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rep = {'subset': subset, 'h5': str(path), 'sha256': _sha256(path),
           'dev_units': list(dev_units), 'test_units': list(test_units),
           'splits': {}, 'capped_active': False}
    for split, units in (('dev', dev_units), ('test', test_units)):
        a = load_aux(path, split)
        y = load_labels(path, split)
        per_unit, n_rows, n_flights, max_rul = {}, 0, 0, 0.0
        for u in units:
            m = a['unit'] == u
            rows = int(m.sum())
            flights = int(len(np.unique(a['cycle'][m])))
            unit_max = float(np.max(y[m])) if rows else 0.0
            per_unit[str(int(u))] = {
                'rows': rows, 'flights': flights,
                'rows_per_flight': round(rows / max(flights, 1), 1),
                'max_rul': unit_max,
            }
            n_rows += rows
            n_flights += flights
            max_rul = max(max_rul, unit_max)
        rep['splits'][split] = {'n_units': len(units), 'n_flights': n_flights,
                                'n_rows': n_rows, 'max_rul': max_rul,
                                'per_unit': per_unit}
        rep['capped_active'] = bool(rep['capped_active'] or max_rul > CAP)
    (out_root / 'audit.json').write_text(json.dumps(rep, indent=1))
    print(f'audit: {out_root / "audit.json"}  capped_active={rep["capped_active"]}  '
          f'dev_flights={rep["splits"]["dev"]["n_flights"]}  '
          f'test_flights={rep["splits"]["test"]["n_flights"]}')
    return rep


def run_fold(rung, fold_idx, held_out, cfg, data, out_root, ledger, pre, data_sha,
             phase='dev', subset='DS02', dev_units=()):
    seed = N0_SEED if rung == 'N0' else None
    seeds = (N0_SEED,) if rung == 'N0' else SEEDS
    rows = []
    for sd in seeds:
        base_id = (f'{rung}_LOEO-{held_out}_s{sd}' if rung != 'N0'
                   else f'{rung}_LOEO-{held_out}')
        # the subset is part of the id because four subsets share one schema and
        # the same fold/seed names; without it a merged table has duplicate ids
        run_id = f'{subset}_{base_id}'
        row_template = {c: '' for c in LEDGER_COLS}
        row_template.update(phase=phase, rung=rung,
                            model=MODEL_NAMES[rung],
                            fold=f'LOEO-{held_out}', held_out_engine=held_out,
                            n_train_engines=5, seed=(sd if rung != 'N0' else ''),
                            git_sha=pre['git_sha'], protocol_blob=pre['protocol_blob'],
                            train_plan_sha256=_sha256(cfg['_root'] / 'paper_b/configs/protocol.json'),
                            feature_set='WS14+W4+CYC',
                            feature_list=';'.join(['Xs_mean', 'Xs_std', 'Xs_slope',
                                                   'W_mean', 'W_std', 'age_scaled']),
                            kmeans_fit_units='' if rung in ('N0', 'N1') else
                            ';'.join(str(u) for u in dev_units if u != held_out),
                            kmeans_seed='' if rung in ('N0', 'N1') else cfg['seeds'][0],
                            pred_instant='flight_end', n_units=1,
                            seg_boundaries=json.dumps({'relative': list(SEG_REL),
                                                       'absolute': list(SEG_ABS)}),
                            independent_unit='engine', status='running')
        if rung == 'N3':
            row_template.update(freeze_source='paper_b/results/FREEZE_ABLATION_SEED101.json;'
                                             'paper_b/configs/protocol.json',
                                freeze_source_sha256=json.dumps(cfg['provenance']))
        try:
            dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            n_train = int(np.asarray(data['is_train'], bool).sum())
            batch = min(BATCH, n_train)
            # epoch* from the five inner folds of the training engines only
            sel = select_epoch_star(rung, data, cfg, sd, dev, batch)
            epoch_star = int(sel['epoch_star'])
            # one fit, two stopping points (manager ruling 2026-11-12): neither is
            # preferred, both are reported, and both come from the same weights.
            # The tag is the frozen template's stop token, not the epoch number:
            # `_ev<epoch>` makes both rows land in one directory when epoch* == 200.
            points = [('inner_val', epoch_star, 'iv'),
                      (f'fixed_passes:{FIXED_EVAL_PASSES}', FIXED_EVAL_PASSES, 'f200')]
            net, res = fit_rung(rung, data, cfg, seed=sd, device=dev,
                                epochs=max(ep for _, ep, _ in points), batch=batch,
                                snapshot_at=sorted({ep for _, ep, _ in points}))
            snaps = res['snapshots']
            ho = ~np.asarray(data['is_train'], bool)
            y_raw = np.asarray(data['y'][ho], np.float64)
            unit = np.asarray(data['unit'][ho])
            cyc = np.asarray(data['cycle'][ho])
            hist = res['history']
            # diagnostic only: the outer curve minimum shows where epoch* sits
            # relative to the outer optimum; it never selects anything
            best = min(hist, key=lambda h: h['holdout_loss'])
            import pandas as pd
            for rule, ep, tag in points:
                if ep not in snaps:
                    raise RuntimeError(f'no snapshot at epoch {ep}')
                d = Path(out_root) / f'{run_id}_{tag}'
                d.mkdir(parents=True, exist_ok=True)
                pred = _predict_held_out(rung, cfg, data, dev, snaps[ep],
                                         res['target_mu'], res['target_sd'])
                if len(pred) != len(y_raw):
                    raise RuntimeError(f'prediction/label length mismatch '
                                       f'{len(pred)} vs {len(y_raw)}')
                # the weights that produced this row, one file per evaluation point
                ckpt = d / f'ckpt_{tag}.pt'
                torch.save(snaps[ep], ckpt)
                # exactly five columns: no bare `y_true`, because that name would
                # silently carry the capped label and become a default criterion.
                pd.DataFrame(dict(unit=unit, cycle=cyc, y_true_raw=y_raw,
                                  y_true_cap125=np.minimum(y_raw, CAP),
                                  y_pred=pred)).to_csv(d / 'pred.csv', index=False)
                (d / 'unit_lifetime.json').write_text(json.dumps(data['unit_lifetime'], indent=1))
                pd.DataFrame(hist).to_csv(d / 'curve.csv', index=False)
                (d / 'config.json').write_text(json.dumps(
                    {**{k: v for k, v in cfg.items() if not k.startswith('_')},
                     'target_norm': 'fold_local', 'target_mu': res['target_mu'],
                     'target_sd': res['target_sd'],
                     'passes_trained': int(res['epochs']),
                     'epoch_evaluated': int(ep), 'selection_rule': rule,
                     'batch_effective': res['batch']}, indent=1, default=str))
                m = metrics(y_raw, pred, data['unit_lifetime'][int(held_out)])
                m['epoch_evaluated'] = int(ep)
                m['selection_rule'] = rule
                m['passes_trained'] = int(res['epochs'])
                m['batch_effective'] = int(res['batch'])
                m['epoch_star'] = epoch_star
                m['epoch_star_raw_argmin'] = int(sel['raw_argmin'])
                m['epoch_floor_applied'] = bool(sel['floor_applied'])
                m['inner_per_fold_best_epoch'] = list(sel['per_fold_best_epoch'])
                m['inner_per_fold_min_loss'] = list(sel['per_fold_min_loss'])
                m['inner_avg_curve'] = list(sel['avg_curve'])
                m['outer_curve_min'] = float(best['holdout_loss'])
                m['outer_curve_argmin'] = int(best['epoch'])
                m['outer_loss_at_evaluated_epoch'] = float(hist[ep - 1]['holdout_loss'])
                (d / 'metrics.json').write_text(json.dumps(m, indent=1))
                row = dict(row_template)
                row.update(m, status='ok',
                           run_id=f'{run_id}_{tag}',
                           epochs_run=int(ep), best_epoch=int(ep),
                           selection_rule=rule,
                           ckpt_path=str(ckpt), ckpt_sha256=_sha256(ckpt),
                           preds_path=str(d / 'pred.csv'),
                           config_path=str(d / 'config.json'),
                           config_sha256=_sha256(d / 'config.json'),
                           registry_note=f'rule={rule}; epoch_star={epoch_star} '
                                         f'raw_argmin={sel["raw_argmin"]} '
                                         f'floor_applied={sel["floor_applied"]}; '
                                         f'trained_to={res["epochs"]}; '
                                         f'inner_best_epochs={sel["per_fold_best_epoch"]}; '
                                         f'inner_min_losses={[round(x, 4) for x in sel["per_fold_min_loss"]]}; '
                                         f'batch_effective={res["batch"]}; '
                                         f'target_norm=fold_local mu={res["target_mu"]:.3f} '
                                         f'sd={res["target_sd"]:.3f}; '
                                         'outer_early_stop=false; outer curve diagnostic only '
                                         '(never used for selection); '
                                         'metrics provisional, recompute with evaluation.py')
                _write_ledger(ledger, row)
                rows.append(row)
                print(f'{run_id}_{tag}: ok  rmse_raw={m["rmse_raw_unit"]:.3f}  '
                      f'rule={rule}')
        except Exception as exc:  # noqa: BLE001 - a failed cell must be recorded, not hidden
            row = dict(row_template)
            row.update(run_id=run_id, status='failed',
                       registry_note=f'{type(exc).__name__}: {exc}',
                       epochs_run='')
            (Path(out_root) / run_id).mkdir(parents=True, exist_ok=True)
            (Path(out_root) / run_id / 'error.txt').write_text(traceback.format_exc())
            _write_ledger(ledger, row)
            rows.append(row)
            print(f'{run_id}: failed  {type(exc).__name__}: {exc}')
    return rows


def _dev_epoch_star(dev_ledger, rung, seed):
    """Stopping point for a full-dev fit, taken from the dev LOEO grid.

    When all six dev engines train the model there is no inner validation engine,
    so ``epoch*`` does not exist at test time. It is taken instead from the dev
    grid: the mean of that (rung, seed)'s six per-fold ``epoch*`` values, rounded
    half-up. Pure dev data, mechanically recomputable, and the test engines play
    no part in it.

    The number is not tight -- DS05 N3 seed 303 spans 20..110 across folds -- so
    the fixed 200-pass column is the primary test evidence and this column is
    reported with that caveat attached.
    """
    want = '' if rung == 'N0' else str(seed)
    rows = [r for r in csv.DictReader(open(dev_ledger, encoding='utf-8'))
            if r['rung'] == rung and r['selection_rule'] == 'inner_val'
            and r['seed'] == want and r['status'] == 'ok']
    vals = [int(r['epochs_run']) for r in rows if r['epochs_run']]
    if len(vals) != 6:
        raise RuntimeError(f'{rung} seed {want!r}: expected 6 dev folds in '
                           f'{dev_ledger}, got {len(vals)}')
    mean = sum(vals) / len(vals)
    return int(np.floor(mean + 0.5)), vals


def run_test_cell(rung, cfg, path, dev_units, test_units, out_root, ledger, pre,
                  data_sha, subset, dev_ledger):
    """Train on all six dev engines and score the sealed test engines.

    No test row influences anything: the stopping points come from the dev grid,
    the feature transforms are fitted on dev only, and the test split is never
    used as a validation signal (``no_holdout=True`` records a NaN curve).
    """
    rows = []
    seeds = (N0_SEED,) if rung == 'N0' else SEEDS
    dev_data, fitted = load_split(path, dev_units, cfg, split='dev', subset=subset)
    test_data, _ = load_split(path, dev_units, cfg, split='test', subset=subset,
                              fitted=fitted)
    n_train = int(np.asarray(dev_data['is_train'], bool).sum())
    batch = min(BATCH, n_train)
    test_unit = np.asarray(test_data['unit'])
    life_per_row = np.array([test_data['unit_lifetime'][int(u)] for u in test_unit],
                            np.float64)
    dev_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for sd in seeds:
        ep_iv, fold_epochs = _dev_epoch_star(dev_ledger, rung, sd)
        base_id = f'{rung}_TEST' + ('' if rung == 'N0' else f'_s{sd}')
        run_id = f'{subset}_{base_id}'
        row_template = {c: '' for c in LEDGER_COLS}
        row_template.update(phase='test', rung=rung, model=MODEL_NAMES[rung],
                            fold='TEST', held_out_engine='',
                            n_train_engines=len(dev_units),
                            seed=(sd if rung != 'N0' else ''),
                            git_sha=pre['git_sha'], protocol_blob=pre['protocol_blob'],
                            train_plan_sha256=_sha256(cfg['_root'] / 'paper_b/configs/protocol.json'),
                            feature_set='WS14+W4+CYC',
                            feature_list=';'.join(['Xs_mean', 'Xs_std', 'Xs_slope',
                                                   'W_mean', 'W_std', 'age_scaled']),
                            kmeans_fit_units='' if rung in ('N0', 'N1') else
                            ';'.join(str(u) for u in dev_units),
                            kmeans_seed='' if rung in ('N0', 'N1') else cfg['seeds'][0],
                            pred_instant='flight_end', n_units=len(test_units),
                            seg_boundaries=json.dumps({'relative': list(SEG_REL),
                                                       'absolute': list(SEG_ABS)}),
                            independent_unit='engine', status='running')
        if rung == 'N3':
            row_template.update(freeze_source='paper_b/results/FREEZE_ABLATION_SEED101.json;'
                                             'paper_b/configs/protocol.json',
                                freeze_source_sha256=json.dumps(cfg['provenance']))
        try:
            points = [('inner_val', ep_iv, 'iv'),
                      (f'fixed_passes:{FIXED_EVAL_PASSES}', FIXED_EVAL_PASSES, 'f200')]
            net, res = fit_rung(rung, dev_data, cfg, seed=sd, device=dev_,
                                epochs=max(ep for _, ep, _ in points), batch=batch,
                                no_holdout=True,
                                snapshot_at=sorted({ep for _, ep, _ in points}))
            snaps = res['snapshots']
            y_raw = np.asarray(test_data['y'], np.float64)
            cyc = np.asarray(test_data['cycle'])
            hist = res['history']
            import pandas as pd
            for rule, ep, tag in points:
                if ep not in snaps:
                    raise RuntimeError(f'no snapshot at epoch {ep}')
                d = Path(out_root) / f'{run_id}_{tag}'
                d.mkdir(parents=True, exist_ok=True)
                pred = _predict_held_out(rung, cfg, test_data, dev_, snaps[ep],
                                         res['target_mu'], res['target_sd'])
                if len(pred) != len(y_raw):
                    raise RuntimeError(f'prediction/label length mismatch '
                                       f'{len(pred)} vs {len(y_raw)}')
                ckpt = d / f'ckpt_{tag}.pt'
                torch.save(snaps[ep], ckpt)
                pd.DataFrame(dict(unit=test_unit, cycle=cyc, y_true_raw=y_raw,
                                  y_true_cap125=np.minimum(y_raw, CAP),
                                  y_pred=pred)).to_csv(d / 'pred.csv', index=False)
                (d / 'unit_lifetime.json').write_text(json.dumps(test_data['unit_lifetime'], indent=1))
                pd.DataFrame(hist).to_csv(d / 'curve.csv', index=False)
                (d / 'config.json').write_text(json.dumps(
                    {**{k: v for k, v in cfg.items() if not k.startswith('_')},
                     'phase': 'test', 'target_norm': 'fold_local',
                     'target_mu': res['target_mu'], 'target_sd': res['target_sd'],
                     'passes_trained': int(res['epochs']),
                     'epoch_evaluated': int(ep), 'selection_rule': rule,
                     'batch_effective': res['batch'],
                     'dev_units': list(dev_units), 'test_units': list(test_units)},
                    indent=1, default=str))
                m = metrics(y_raw, pred, life_per_row)
                m['epoch_evaluated'] = int(ep)
                m['selection_rule'] = rule
                m['passes_trained'] = int(res['epochs'])
                m['batch_effective'] = int(res['batch'])
                m['epoch_star'] = int(ep_iv)
                m['epoch_star_rule'] = 'mean_of_six_dev_loeo_epochs'
                m['epoch_star_fold_values'] = list(fold_epochs)
                (d / 'metrics.json').write_text(json.dumps(m, indent=1))
                row = dict(row_template)
                row.update(m, status='ok', run_id=f'{run_id}_{tag}',
                           epochs_run=int(ep), best_epoch=int(ep),
                           selection_rule=rule,
                           ckpt_path=str(ckpt), ckpt_sha256=_sha256(ckpt),
                           preds_path=str(d / 'pred.csv'),
                           config_path=str(d / 'config.json'),
                           config_sha256=_sha256(d / 'config.json'),
                           registry_note=f'phase=test; rule={rule}; '
                                         f'epoch_star={ep_iv} '
                                         f'(mean of dev LOEO epochs {fold_epochs}); '
                                         f'trained_on=all {len(dev_units)} dev engines; '
                                         f'no_holdout=true (test never used as validation); '
                                         f'batch_effective={res["batch"]}; '
                                         f'target_norm=fold_local mu={res["target_mu"]:.3f} '
                                         f'sd={res["target_sd"]:.3f}; '
                                         'fixed_passes:200 is the primary test column')
                _write_ledger(ledger, row)
                rows.append(row)
                print(f'{run_id}_{tag}: ok  rmse_raw={m["rmse_raw_unit"]:.3f}  rule={rule}')
        except Exception as exc:  # noqa: BLE001 - a failed cell must be recorded, not hidden
            row = dict(row_template)
            row.update(run_id=run_id, status='failed',
                       registry_note=f'{type(exc).__name__}: {exc}', epochs_run='')
            (Path(out_root) / run_id).mkdir(parents=True, exist_ok=True)
            (Path(out_root) / run_id / 'error.txt').write_text(traceback.format_exc())
            _write_ledger(ledger, row)
            rows.append(row)
            print(f'{run_id}: failed  {type(exc).__name__}: {exc}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', default='')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--folds', type=int, default=0, help='limit folds for a plumbing check')
    ap.add_argument('--out-root', default='experiments/ncmapss')
    ap.add_argument('--ledger', default='',
                    help='default: <out-root>/<subset>/runs_dev.csv')
    ap.add_argument('--subset', default='',
                    help='subset label, e.g. DS05 (default: parsed from the h5 name)')
    ap.add_argument('--data-sha256', default='')
    ap.add_argument('--rungs', default=','.join(RUNGS),
                    help=f'comma-separated rungs (main grid: {",".join(RUNGS)}; '
                         f'diagnostics: {",".join(DIAG_RUNGS)})')
    ap.add_argument('--phase', choices=('dev', 'test'), default='dev',
                    help='dev = leave-one-engine-out on the dev engines; '
                         'test = train on all dev engines, score the sealed test engines')
    ap.add_argument('--dev-ledger', default='',
                    help='dev ledger used for epoch* in the test phase '
                         '(default: <out-root>/<subset>/runs_dev.csv)')
    a = ap.parse_args()
    cfg = load_frozen()
    cfg['_root'] = Path(__file__).resolve().parents[2]
    pre = _preflight(cfg['_root'])
    print('preflight:', pre)

    if a.synthetic:
        import tempfile
        from .test_contract import _write_fake
        p = Path(tempfile.mkdtemp()) / 'fake_ds02.h5'
        _write_fake(p, dev_units=(1, 2, 3, 4, 5, 6), test_units=(7, 8),
                    cycles=80, rows_per_cycle=12)
    else:
        if not a.h5:
            raise SystemExit('--h5 is required unless --synthetic')
        p = Path(a.h5)

    # the split is read from the file, never from a list in this package
    dev_units, test_units = official_split(p)
    subset = a.subset or _subset_name(p)
    out_root = Path(a.out_root) / subset
    ledger = a.ledger or str(out_root / ('runs_test.csv' if a.phase == 'test'
                                         else 'runs_dev.csv'))
    print(f'subset {subset}: dev {list(dev_units)}  test {list(test_units)}  '
          f'phase {a.phase}  out {out_root}')
    if not a.synthetic:
        _write_subset_audit(p, subset, out_root, dev_units, test_units)
    data_sha = a.data_sha256 or ('SYNTHETIC' if a.synthetic else _sha256(p))

    if a.phase == 'test':
        dev_ledger = a.dev_ledger or str(out_root / 'runs_dev.csv')
        print(f'test phase: ledger {ledger}  epoch* from {dev_ledger}')
        all_rows = []
        for rung in [r.strip() for r in a.rungs.split(',') if r.strip()]:
            if rung in DIAG_RUNGS:
                raise SystemExit(f'{rung} is a diagnostic arm; the test phase is '
                                 'only for the pre-registered rungs')
            all_rows += run_test_cell(rung, cfg, p, dev_units, test_units,
                                      str(out_root), ledger, pre, data_sha,
                                      subset, dev_ledger)
        ok = sum(r['status'] == 'ok' for r in all_rows)
        print(f'done: {ok}/{len(all_rows)} ok, ledger {ledger}')
        return

    # one fold-construction authority: evaluation.loeo_folds, fed with the two
    # lists read from the file. The runner keeps no unit list of its own.
    folds = loeo_folds(dev_units, test_units)
    if a.folds:
        folds = folds[:a.folds]
    all_rows = []
    for k, fold in enumerate(folds, 1):
        held = int(fold.val_unit)
        train_units = [int(u) for u in fold.train_units]
        data = load_subset(p, train_units, cfg, subset=subset)
        print(f'fold {k}/{len(folds)}: holdout {held}, train {train_units}, '
              f'{data["x"].shape[0]} flights, drops {data["drop_info"]["n_dropped"]}')
        for rung in [r.strip() for r in a.rungs.split(',') if r.strip()]:
            # A diagnostic arm never enters the frozen-grid evidence table: it was
            # not pre-registered, so it goes to <subset>/diag/ with phase=diag and
            # its own ledger. This is enforced here rather than left to the caller,
            # so no invocation can pollute runs_dev.csv by accident.
            if rung in DIAG_RUNGS:
                out_dir = str(out_root / 'diag')
                led = str(out_root / 'diag' / 'runs_diag.csv')
                phase = 'diag'
            else:
                out_dir, led, phase = str(out_root), ledger, 'dev'
            all_rows += run_fold(rung, k, held, cfg, data, out_dir, led, pre,
                                 a.data_sha256 or ('SYNTHETIC' if a.synthetic else _sha256(p)),
                                 phase=phase, subset=subset, dev_units=dev_units)
    ok = sum(r['status'] == 'ok' for r in all_rows)
    print(f'done: {ok}/{len(all_rows)} ok, ledger {ledger}')


if __name__ == '__main__':
    main()
