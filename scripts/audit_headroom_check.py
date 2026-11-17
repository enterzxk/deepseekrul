"""Independent check of HEADROOM_DIAGNOSTIC.md.

The diagnostic's script is not in the repo, so this is a second implementation
written from the document's stated caliber, not from its code:

  * data: split_seed=29, E0 official TEST, 5 seeds (42/43/101/202/303)
  * caliber: average the per-engine error over seeds FIRST, then compute
    engine-level RMSE. This is NOT `pooled_rmse` (which averages per-seed pooled
    RMSE); the document says so explicitly and the two must not be mixed.

Reported quantities: RMSE / MAE / bias / late-fraction / top-20% share of the
squared error, the three counterfactuals (halve top-20%, halve top-10%, remove
the global bias), the worst three engines with their cross-seed sd, and the
RUL-bucket table.
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
SEEDS = (42, 43, 101, 202, 303)
BUCKETS = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 126))


def load(cell: Path):
    out = {}
    with open(cell / 'test_predictions.csv', newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('corruption_type') not in ('none', '', None):
                continue
            out[int(row['engine_id'])] = (float(row['y_true']), float(row['y_pred_evaluated']))
    if not out:
        raise SystemExit(f'{cell}: no clean rows')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root = Path(a.root)

    per_subset, pooled_by_seed, worst, bucket_rows = {}, {}, {}, []
    for subset in SUBSETS:
        cells = [load(Path(resolve(root, 'E0', subset, sd, 29))) for sd in SEEDS]
        eng = sorted(set.intersection(*[set(c) for c in cells]))
        if any(len(c) != len(eng) for c in cells):
            raise SystemExit(f'{subset}: seeds disagree on the engine set')
        err = np.array([[c[u][1] - c[u][0] for u in eng] for c in cells])   # (seed, engine)
        ytrue = np.array([cells[0][u][0] for u in eng])
        m = err.mean(axis=0)                       # seed-averaged per-engine error
        sd = err.std(axis=0, ddof=1)               # cross-seed sd of that error
        rmse = float(np.sqrt(np.mean(m ** 2)))
        # pooled_rmse caliber, for the caliber-distinction cross-check only
        pooled_by_seed[subset] = float(np.mean([math.sqrt(np.mean(e ** 2)) for e in err]))

        order = np.argsort(-(m ** 2))
        n = len(m)
        shares = {}
        scen = {}
        for pct in (0.2, 0.1):
            k = max(1, int(round(pct * n)))
            shares[pct] = float(np.sum(m[order[:k]] ** 2) / np.sum(m ** 2))
            half = m.copy()
            half[order[:k]] = half[order[:k]] / 2.0
            scen[pct] = float(np.sqrt(np.mean(half ** 2)))
        debias = m - m.mean()
        per_subset[subset] = dict(
            n=n, rmse=rmse, mae=float(np.mean(np.abs(m))), bias=float(m.mean()),
            late=float(np.mean(m > 0)), nasa_sum=float(np.sum(
                [math.exp(-d / 13) - 1 if d < 0 else math.exp(d / 10) - 1 for d in m])),
            top20_share=shares[0.2], halve_top20=scen[0.2], halve_top10=scen[0.1],
            debias_rmse=float(np.sqrt(np.mean(debias ** 2))),
            pooled_rmse_seed_mean=pooled_by_seed[subset],
        )
        worst[subset] = [
            dict(engine=int(eng[i]), err=float(m[i]), seed_sd=float(sd[i]))
            for i in order[:3]
        ]
        for u, y, e, s in zip(eng, ytrue, m, sd):
            bucket_rows.append(dict(subset=subset, engine=int(u), y_true=float(y),
                                    err=float(e), seed_sd=float(s)))

    print(f'{"subset":7} {"n":>4} {"RMSE":>7} {"MAE":>7} {"bias":>7} {"late":>6} '
          f'{"NASA":>6} {"top20":>7} {"half20":>7} {"half10":>7} {"debias":>7}')
    for s in SUBSETS:
        v = per_subset[s]
        print(f'{s:7} {v["n"]:>4} {v["rmse"]:>7.3f} {v["mae"]:>7.3f} {v["bias"]:>7.3f} '
              f'{v["late"]:>6.2%} {v["nasa_sum"]:>6.0f} {v["top20_share"]:>7.1%} '
              f'{v["halve_top20"]:>7.3f} {v["halve_top10"]:>7.3f} {v["debias_rmse"]:>7.3f}')

    print('\ncaliber cross-check (seed-mean of pooled RMSE, must differ from RMSE above):')
    for s in SUBSETS:
        print(f'  {s}: engine-level {per_subset[s]["rmse"]:.4f}  vs  '
              f'seed-mean pooled {per_subset[s]["pooled_rmse_seed_mean"]:.4f}')

    print('\nworst three per subset:')
    for s in SUBSETS:
        print('  ' + s + '  ' + '  '.join(
            f'e{w["engine"]} {w["err"]:+.1f}/{w["seed_sd"]:.1f}' for w in worst[s]))

    print('\nRUL buckets (all subsets pooled):')
    print(f'{"bucket":>10} {"n":>5} {"MAE":>7} {"bias":>7} {"late":>6}')
    buckets = {}
    for lo, hi in BUCKETS:
        sel = [r for r in bucket_rows if lo <= r['y_true'] < hi]
        if not sel:
            continue
        e = np.array([r['err'] for r in sel])
        buckets[f'{lo}-{hi}'] = dict(n=len(sel), mae=float(np.mean(np.abs(e))),
                                     bias=float(e.mean()), late=float(np.mean(e > 0)))
        print(f'{lo:>4}-{hi:<5} {len(sel):>5} {buckets[f"{lo}-{hi}"]["mae"]:>7.2f} '
              f'{buckets[f"{lo}-{hi}"]["bias"]:>+7.2f} {buckets[f"{lo}-{hi}"]["late"]:>6.0%}')

    if a.json:
        Path(a.json).write_text(json.dumps(dict(per_subset=per_subset, worst=worst,
                                                buckets=buckets), indent=1), encoding='utf-8')
        print(f'\nwrote {a.json}')


if __name__ == '__main__':
    raise SystemExit(main())
