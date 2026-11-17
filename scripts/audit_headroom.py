#!/usr/bin/env python3
"""Where is the baseline's error concentrated? (HEADROOM_DIAGNOSTIC.md)

Convention (the doc's section 0, and the whole point of this file):
**average each engine's error over seeds FIRST, then compute RMSE over engines.**
That is NOT the same quantity as `pooled_rmse` (pooled per seed, then averaged over
seeds). The two differ by ~0.5 cycles on FD001 (10.000 vs 10.5276) precisely because
the first removes seed noise; the second keeps it. Never swap one for the other.

Row filter is identical to `audit_test.py`: `evaluation_protocol == 'clean_test'`
and `corruption_type == 'none'`, one row per engine, duplicate engine is fatal.

Usage
-----
    python scripts/audit_headroom.py --runs <runs> --split 29 --arm E0
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
BUCKETS = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 127))
TOP_FRAC = 0.20


def nasa(d: float) -> float:
    return math.exp(-d / 13.0) - 1.0 if d < 0 else math.exp(d / 10.0) - 1.0


def load_clean(cell: Path) -> dict[int, tuple[float, float]]:
    out = {}
    with open(cell / 'test_predictions.csv', newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('corruption_type') not in ('none', '', None):
                continue
            if row.get('evaluation_protocol') not in ('clean_test', '', None):
                continue
            u = int(row['engine_id'])
            if u in out:
                raise SystemExit(f'{cell}: duplicate engine_id {u}')
            out[u] = (float(row['y_true']), float(row['y_pred_evaluated']))
    if not out:
        raise SystemExit(f'{cell}: no clean rows')
    return out


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
    print(f'arm={a.arm} split={split} seeds={seeds}')
    print(f'{"subset":7} {"n":>4} {"engineRMSE":>11} {"top20% share":>13} '
          f'{"halve->":>8} {"bias-only":>10}')
    for subset in SUBSETS:
        # per engine: y_true, and the per-seed signed errors
        per_engine = {}
        for seed in seeds:
            cell = Path(resolve(runs, a.arm, subset, seed, split))
            rows = load_clean(cell)
            for u, (yt, yp) in rows.items():
                d = per_engine.setdefault(u, dict(y_true=yt, errs=[]))
                if d['y_true'] != yt:
                    raise SystemExit(f'{subset} engine {u}: y_true differs across seeds '
                                     f'({d["y_true"]} vs {yt}) -- seeds must be paired')
                d['errs'].append(yp - yt)
        n_seed = {len(d['errs']) for d in per_engine.values()}
        if n_seed != {len(seeds)}:
            raise SystemExit(f'{subset}: engines have differing seed counts {n_seed}')

        eng = sorted(per_engine)
        # CONVENTION (doc section 0, literally): average the *error* over seeds per
        # engine, THEN take the RMSE of those per-engine averages. Averaging the
        # squared error instead would be algebraically identical to the pooled RMSE
        # (all engines have the same seed count) and would erase the very distinction
        # section 0 exists to make: 10.000 vs 10.5276.
        signed_u = np.array([np.mean(per_engine[u]['errs']) for u in eng])
        mse_u = signed_u ** 2
        rmse = float(np.sqrt(mse_u.mean()))

        k = max(1, int(round(TOP_FRAC * len(eng))))
        top = np.argsort(-mse_u)[:k]
        share = float(mse_u[top].sum() / mse_u.sum())

        # "halve the error of the worst 20%" means the ERROR is halved, so its
        # squared contribution becomes 1/4. Halving the squared term instead (which
        # is what a naive reading gives) yields 7.90 on FD001 rather than 6.60.
        halved = mse_u.copy()
        halved[top] *= 0.25
        rmse_halved = float(np.sqrt(halved.mean()))

        resid = signed_u - signed_u.mean()          # remove only the global bias
        rmse_bias = float(np.sqrt((resid ** 2).mean()))

        worst = sorted(range(len(eng)), key=lambda i: -abs(signed_u[i]))[:3]
        worst_tab = [(int(eng[i]), float(signed_u[i]),
                      float(np.std(per_engine[eng[i]]['errs'], ddof=1)))
                     for i in worst]

        buckets = {}
        for lo, hi in BUCKETS:
            idx = [i for i, u in enumerate(eng) if lo <= per_engine[u]['y_true'] < hi]
            if not idx:
                continue
            e = signed_u[idx]
            buckets[f'{lo}-{hi}'] = dict(
                n=len(idx), mae=float(np.mean(np.abs(e))), bias=float(e.mean()),
                late_frac=float(np.mean([1.0 if x > 0 else 0.0 for x in e])))

        scores = np.array([sum(nasa(x) for x in per_engine[u]['errs']) / len(seeds)
                           for u in eng])
        k10 = max(1, int(round(0.10 * len(eng))))
        top10 = np.argsort(-scores)[:k10]
        nasa_share = float(scores[top10].sum() / scores.sum())

        report[subset] = dict(
            n_engines=len(eng), engine_rmse=rmse, top_share=share, k_top=k,
            halved_rmse=rmse_halved, bias_only_rmse=rmse_bias,
            worst3=worst_tab, buckets=buckets, nasa_top10_share=nasa_share,
            global_bias=float(signed_u.mean()))

        print(f'{subset:7} {len(eng):4d} {rmse:11.4f} {share:13.4f} '
              f'{rmse_halved:8.4f} {rmse_bias:10.4f}')
        print(f'        worst3 ' + ', '.join(f'e{u} {e:+.1f} (sd {s:.1f})'
                                             for u, e, s in worst_tab))
        for b, v in buckets.items():
            print(f'        RUL {b:8} n={v["n"]:3d} MAE {v["mae"]:6.2f} '
                  f'bias {v["bias"]:+6.2f} late {v["late_frac"]*100:4.0f}%')
        print(f'        NASA top10% share {nasa_share:.4f}')

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')
        print(f'wrote {a.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
