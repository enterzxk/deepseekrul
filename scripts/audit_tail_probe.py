#!/usr/bin/env python3
"""Two zero-training checks before any tail module is designed.

P0 -- does the TEST phenomenon also exist on the VALIDATION partition?
    The diagnostic was computed on official TEST. Designing a module for a
    pattern that only exists there would be fitting the test set. So first
    reproduce it on validation: top-20% share of squared error, the RUL 75-100
    bucket bias, and seed stability of the tail. If it does not reproduce, stop.

P1 -- how much of the tail is recoverable by OUTPUT CALIBRATION alone?
    This sets the bar any module has to beat. Calibration is fitted with
    cross-fitting BY ENGINE (never on the same engines it is evaluated on),
    because a same-fold fit would report the fit, not the generalisation.

Convention: per (engine, cut point) sample, average the signed error over seeds
first, then aggregate -- same as HEADROOM_DIAGNOSTIC.md section 0.

Usage
-----
    python scripts/audit_tail_probe.py --runs <runs> --split 29 --arm E0
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

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')
SEEDS = {17: (101, 202, 303), 29: (42, 43, 101, 202, 303)}
TAIL = (75.0, 100.0)
TOP_FRAC = 0.20
FOLDS = 5


def load_val(cell: Path) -> dict[tuple[int, float], tuple[float, float]]:
    """{(unit, cutoff): (y_true, y_pred)} for one cell."""
    out = {}
    with open(cell / 'val_pred.csv', newline='') as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        pcol = next((c for c in ('y_pred', 'y_pred_evaluated', 'y_pred_raw') if c in cols), None)
        icol = next((c for c in ('unit', 'engine_id') if c in cols), None)
        if pcol is None or icol is None or 'y_true' not in cols:
            raise SystemExit(f'{cell}: unexpected columns {cols}')
        for r in rd:
            cut = r.get('cutoff_cycle')
            key = (int(r[icol]), float(cut) if cut not in (None, '') else float(len(out)))
            if key in out:
                raise SystemExit(f'{cell}: duplicate key {key}')
            out[key] = (float(r['y_true']), float(r[pcol]))
    if not out:
        raise SystemExit(f'{cell}: no rows')
    return out


def p90(x):
    return float(np.percentile(np.abs(x), 90))


def fold_of(engines, k):
    """Deterministic engine-grouped fold assignment (no rng: reproducible)."""
    uniq = sorted(engines)
    return {u: i % k for i, u in enumerate(uniq)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--split', type=int, default=29)
    ap.add_argument('--arm', default='E0')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    runs, split = Path(a.runs), a.split
    seeds = SEEDS[split]

    report = {}
    print(f'arm={a.arm} split={split} seeds={seeds}  (validation partition)')
    print()
    print('== P0: does the TEST structure reproduce on validation? ==')
    print(f'{"subset":7} {"n_samp":>6} {"n_eng":>5} {"RMSE":>8} {"top20%":>8} '
          f'{"RUL75-100 n":>12} {"bias":>8} {"late":>6}')
    per_subset_samples = {}
    for subset in SUBSETS:
        acc = {}
        for seed in seeds:
            cell = Path(resolve(runs, a.arm, subset, seed, split))
            for key, (yt, yp) in load_val(cell).items():
                d = acc.setdefault(key, dict(y_true=yt, errs=[]))
                if d['y_true'] != yt:
                    raise SystemExit(f'{subset} {key}: y_true differs across seeds')
                d['errs'].append(yp - yt)
        if {len(d['errs']) for d in acc.values()} != {len(seeds)}:
            raise SystemExit(f'{subset}: uneven seed coverage')

        keys = sorted(acc)
        err = np.array([np.mean(acc[k]['errs']) for k in keys])
        yt = np.array([acc[k]['y_true'] for k in keys])
        eng = np.array([k[0] for k in keys])

        rmse = float(np.sqrt((err ** 2).mean()))
        sq = err ** 2
        ktop = max(1, int(round(TOP_FRAC * len(keys))))
        top = np.argsort(-sq)[:ktop]
        share = float(sq[top].sum() / sq.sum())

        in_tail = (yt >= TAIL[0]) & (yt < TAIL[1])
        t_bias = float(err[in_tail].mean()) if in_tail.any() else float('nan')
        t_late = float((err[in_tail] > 0).mean()) if in_tail.any() else float('nan')

        worst = np.argsort(-np.abs(err))[:3]
        worst_tab = [(int(keys[i][0]), float(err[i]),
                      float(np.std(acc[keys[i]]['errs'], ddof=1))) for i in worst]

        per_subset_samples[subset] = dict(keys=keys, err=err, yt=yt, eng=eng, acc=acc)
        report[subset] = dict(n_samples=len(keys), n_engines=len(set(eng.tolist())),
                              rmse=rmse, top_share=share,
                              tail_n=int(in_tail.sum()), tail_bias=t_bias,
                              tail_late=t_late, worst3=worst_tab)
        print(f'{subset:7} {len(keys):6d} {len(set(eng.tolist())):5d} {rmse:8.4f} '
              f'{share:8.4f} {int(in_tail.sum()):12d} {t_bias:+8.2f} {t_late*100:5.0f}%')
        print('        worst3 ' + ', '.join(f'e{u} {e:+.1f} (sd {s:.1f})'
                                            for u, e, s in worst_tab))

    print()
    print('== P0 gate: top20% >= 0.60, tail bias >= +2.0 ==')
    ok = all(v['top_share'] >= 0.60 and (not np.isnan(v['tail_bias'])) and v['tail_bias'] >= 2.0
             for v in report.values())
    for s, v in report.items():
        flag = 'PASS' if (v['top_share'] >= 0.60 and v['tail_bias'] >= 2.0) else 'FAIL'
        print(f'  {s}: top20% {v["top_share"]:.3f} / tail bias {v["tail_bias"]:+.2f}  {flag}')
    print(f'  P0 overall: {"REPRODUCED" if ok else "NOT REPRODUCED"}')
    report['_p0_pass'] = ok

    # ---- P0b: the TEST-matched analogue, one sample per engine (its LAST cut) ----
    # The block above buckets CUT POINTS; official TEST buckets ENGINES (last window).
    # Those are different populations -- an engine with 120 cycles contributes several
    # cut points to the 75-100 bucket above but exactly one to the TEST bucket. So P0b
    # is the comparison that actually answers "does the TEST structure transfer".
    print()
    print('== P0b: TEST-matched (one sample per engine: its last cut) ==')
    print(f'{"subset":7} {"n_eng":>5} {"RMSE":>8} {"top20%":>8} {"tail n":>7} '
          f'{"bias":>8} {"late":>6}')
    p0b = {}
    for subset in SUBSETS:
        d = per_subset_samples[subset]
        last = {}
        for k, e, yt in zip(d['keys'], d['err'], d['yt']):
            u, cut = k
            if u not in last or cut > last[u][0]:
                last[u] = (cut, e, yt)
        err2 = np.array([v[1] for v in last.values()])
        yt2 = np.array([v[2] for v in last.values()])
        rmse2 = float(np.sqrt((err2 ** 2).mean()))
        sq2 = err2 ** 2
        k2 = max(1, int(round(TOP_FRAC * len(err2))))
        top2 = np.argsort(-sq2)[:k2]
        share2 = float(sq2[top2].sum() / sq2.sum())
        m2 = (yt2 >= TAIL[0]) & (yt2 < TAIL[1])
        b2 = float(err2[m2].mean()) if m2.any() else float('nan')
        l2 = float((err2[m2] > 0).mean()) if m2.any() else float('nan')
        p0b[subset] = dict(n_engines=len(err2), rmse=rmse2, top_share=share2,
                           tail_n=int(m2.sum()), tail_bias=b2, tail_late=l2)
        print(f'{subset:7} {len(err2):5d} {rmse2:8.4f} {share2:8.4f} '
              f'{int(m2.sum()):7d} {b2:+8.2f} {l2*100:5.0f}%')
    ok_b = all(v['top_share'] >= 0.60 and (not np.isnan(v['tail_bias'])) and v['tail_bias'] >= 2.0
               for v in p0b.values())
    for s, v in p0b.items():
        flag = 'PASS' if (v['top_share'] >= 0.60 and v['tail_bias'] >= 2.0) else 'FAIL'
        print(f'  {s}: top20% {v["top_share"]:.3f} / tail bias {v["tail_bias"]:+.2f}  {flag}')
    print(f'  P0b overall: {"REPRODUCED" if ok_b else "NOT REPRODUCED"}')
    report['_p0b_lastcut'] = p0b
    report['_p0b_pass'] = ok_b

    # ---------------- P1: calibration probe, cross-fitted by engine ----------
    print()
    print('== P1: how much does output calibration recover? (cross-fitted by engine) ==')
    print(f'{"subset":7} {"base RMSE":>10} {"bin RMSE":>9} {"dRMSE":>8} '
          f'{"base P90":>9} {"bin P90":>8} {"dP90":>8}')
    cal_report = {}
    for subset in SUBSETS:
        d = per_subset_samples[subset]
        err, eng = d['err'], d['eng']
        # yhat per sample = y_true + err  (seed-averaged prediction)
        yhat = d['yt'] + err
        # engine-grouped folds via unique/inverse: no dict lookups, no key-dtype traps
        uniq, inv = np.unique(eng, return_inverse=True)
        fid = inv % FOLDS
        if len(uniq) < FOLDS:
            raise SystemExit(f'{subset}: only {len(uniq)} engines for {FOLDS} folds')

        out_err = np.empty_like(err)
        for f in range(FOLDS):
            tr, te = fid != f, fid == f
            if te.sum() == 0:
                continue
            # 5-parameter piecewise offset on yhat quantiles of the TRAIN fold
            qs = np.quantile(yhat[tr], [0.2, 0.4, 0.6, 0.8])
            bin_tr = np.digitize(yhat[tr], qs)
            bin_te = np.digitize(yhat[te], qs)
            off = np.array([np.mean(err[tr][bin_tr == b]) if (bin_tr == b).any() else 0.0
                            for b in range(len(qs) + 1)])
            out_err[te] = err[te] - off[bin_te]

        base_rmse = float(np.sqrt((err ** 2).mean()))
        bin_rmse = float(np.sqrt((out_err ** 2).mean()))
        base_p90, bin_p90 = p90(err), p90(out_err)
        cal_report[subset] = dict(base_rmse=base_rmse, bin_rmse=bin_rmse,
                                  d_rmse=bin_rmse - base_rmse,
                                  base_p90=base_p90, bin_p90=bin_p90,
                                  d_p90=bin_p90 - base_p90,
                                  tail_bias_after=float(out_err[(d['yt'] >= TAIL[0])
                                                                & (d['yt'] < TAIL[1])].mean()))
        print(f'{subset:7} {base_rmse:10.4f} {bin_rmse:9.4f} '
              f'{bin_rmse - base_rmse:+8.4f} {base_p90:9.4f} {bin_p90:8.4f} '
              f'{bin_p90 - base_p90:+8.4f}')
    report['_calibration'] = cal_report

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')
        print(f'wrote {a.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
