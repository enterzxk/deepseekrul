"""Independent check of the three artifact exclusions in HEADROOM_DIAGNOSTIC.md §7.

The buckets must be cut on the **raw** RUL from `RUL_FD00x.txt`, not on the
`y_true` stored in `test_predictions.csv` -- that column is the 125-capped
label, and the whole point of this check is whether the mid-life hump survives
when the cap is removed.

Caliber is the same as the diagnostic: average the per-engine signed error over
seeds first, then aggregate. Data: split_seed=29, E0, seeds 42/43/101/202/303.
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
BUCKETS = ((0, 25), (25, 50), (50, 75), (75, 100), (100, 126), (126, 10 ** 6))


def read_rul(data_dir: Path, subset: str):
    """Raw (uncapped) RUL label per engine id, 1-based, from RUL_<subset>.txt."""
    for name in (f'RUL_{subset}.txt', f'RUL_{subset}.TXT', f'rul_{subset}.txt'):
        p = data_dir / name
        if p.exists():
            vals = [float(x) for x in p.read_text().split()]
            return {i + 1: v for i, v in enumerate(vals)}
    raise SystemExit(f'{subset}: no RUL_{subset}.txt under {data_dir}')


def load_preds(cell: Path):
    out = {}
    with open(cell / 'test_predictions.csv', newline='') as fh:
        for row in csv.DictReader(fh):
            if row.get('corruption_type') not in ('none', '', None):
                continue
            out[int(row['engine_id'])] = float(row['y_pred_evaluated'])
    if not out:
        raise SystemExit(f'{cell}: no clean rows')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    root, data_dir = Path(a.root), Path(a.data_dir)

    rows, slopes = [], {}
    for subset in SUBSETS:
        rul = read_rul(data_dir, subset)
        preds = [load_preds(Path(resolve(root, 'E0', subset, sd, 29))) for sd in SEEDS]
        eng = sorted(set.intersection(*[set(p) for p in preds]))
        yhat = np.array([np.mean([p[u] for p in preds]) for u in eng])
        raw = np.array([rul[u] for u in eng])
        capped = np.array([min(125.0, rul[u]) for u in eng])
        for u, r, c, h in zip(eng, raw, capped, yhat):
            rows.append(dict(subset=subset, engine=int(u), raw=float(r), capped=float(c),
                             yhat=float(h), err_raw=float(h - r), err_capped=float(h - c)))
        slopes[subset] = dict(
            slope_raw=float(np.polyfit(raw, yhat, 1)[0]),
            slope_capped=float(np.polyfit(capped, yhat, 1)[0]),
            sd_pred=float(np.std(yhat)), sd_raw=float(np.std(raw)),
            sd_capped=float(np.std(capped)),
        )

    print('== raw-RUL buckets (all subsets pooled)')
    print(f'{"bucket":>12} {"n":>5} {"MAE":>7} {"bias":>7} {"late":>6}')
    for lo, hi in BUCKETS:
        sel = [r for r in rows if lo <= r['raw'] < hi]
        if not sel:
            continue
        e = np.array([r['err_raw'] for r in sel])
        lab = f'{lo}-{hi}' if hi < 10 ** 6 else f'{lo}+'
        print(f'{lab:>12} {len(sel):>5} {np.mean(np.abs(e)):>7.2f} {e.mean():>+7.2f} '
              f'{np.mean(e > 0):>6.0%}')

    print('\n== raw 75-100 bucket, per subset')
    print(f'{"subset":>7} {"n":>4} {"MAE":>7} {"bias":>7} {"late":>6}')
    for subset in SUBSETS:
        sel = [r for r in rows if r['subset'] == subset and 75 <= r['raw'] < 100]
        if not sel:
            print(f'{subset:>7} {0:>4}')
            continue
        e = np.array([r['err_raw'] for r in sel])
        print(f'{subset:>7} {len(sel):>4} {np.mean(np.abs(e)):>7.2f} {e.mean():>+7.2f} '
              f'{np.mean(e > 0):>6.0%}')

    print('\n== regression dilution check (slope of yhat on truth)')
    print(f'{"subset":>7} {"slope_raw":>10} {"slope_capped":>13} {"sd_pred":>8} {"sd_raw":>7}')
    for subset in SUBSETS:
        s = slopes[subset]
        print(f'{subset:>7} {s["slope_raw"]:>10.3f} {s["slope_capped"]:>13.3f} '
              f'{s["sd_pred"]:>8.1f} {s["sd_raw"]:>7.1f}')

    if a.json:
        Path(a.json).write_text(json.dumps(dict(slopes=slopes), indent=1), encoding='utf-8')
        print(f'\nwrote {a.json}')


if __name__ == '__main__':
    raise SystemExit(main())
