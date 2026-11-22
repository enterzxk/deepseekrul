"""Gate 1: is CADF plug-and-play across the five reproduced SOTA backbones?

=============================================================================
SIGN CONVENTION -- read this first.
=============================================================================
The pre-registration states two thresholds that use OPPOSITE conventions:

    "not degraded"   :  delta <= +0.05     (reads as module - baseline)
    "positive gain"  :  delta  >  0        (reads as baseline - module)

If both are taken literally, the same delta has to be both negative and
positive. This script pins ONE convention and translates the other:

    delta := baseline - module            (POSITIVE = the module is better)

    not degraded   :  delta_EOL_MAE >= -0.05     (i.e. module - baseline <= 0.05)
    positive gain  :  delta_RMSE    >  0   AND  paired win rate > 0.50

Everything printed below is in this convention. Do not re-derive the guardrail
from the "+0.05" in the plan without translating it first.

=============================================================================
Input contract (both files: a JSON list of rows)
=============================================================================
    {"method": str,   # aidgn | two_phase | dual_mixer | ffve | fcstgnn
     "subset": str,   # FD001 | FD004
     "seed":   int,   # 101 | 202 | 303
     "engine": int,   # engine id
     "cut":    float, # truncation fraction of that engine's own life, 0..1
     "y":      float, # true RUL at that cut
     "pred":   float} # predicted RUL at that cut

Per-engine, per-cut predictions are required because EOL(0.95) is defined at a
cut, and the paired win rate is defined per engine. A pre-aggregated file
cannot produce either quantity.

Gate: a backbone passes only if BOTH subsets pass (FD001 and FD004), because
B_min already fails FD004 -- so "4 of 6 backbones" is really "4 of the 5 SOTA
backbones pass on both subsets".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ('aidgn', 'two_phase', 'dual_mixer', 'ffve', 'fcstgnn')
SUBSETS = ('FD001', 'FD004')
SEEDS = (101, 202, 303)
EOL_CUT = 0.95
NOT_DEGRADED = -0.05      # delta >= this
MIN_PASS_METHODS = 4
BOOT = 10000
BOOT_SEED = 20261122


def load(path: Path):
    rows = json.loads(path.read_text())
    need = {'method', 'subset', 'seed', 'engine', 'cut', 'y', 'pred'}
    seen = set()
    for i, r in enumerate(rows):
        missing = need - set(r)
        if missing:
            raise SystemExit(f'{path}: row {i} missing {sorted(missing)}')
        key = (r['method'], r['subset'], int(r['seed']), int(r['engine']), round(float(r['cut']), 6))
        if key in seen:
            raise SystemExit(f'{path}: duplicate row {key}')
        seen.add(key)
    return rows


def by_cell(rows):
    """{(method, subset, seed): {engine: [(cut, y, pred), ...]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[(r['method'], r['subset'], int(r['seed']))][int(r['engine'])].append(
            (float(r['cut']), float(r['y']), float(r['pred'])))
    return out


def cell_metrics(cell):
    """Per-engine RMSE over all cuts, and the |error| at the cut nearest 0.95."""
    rmse, eol = {}, {}
    for eng, pts in cell.items():
        pts = sorted(pts)
        rmse[eng] = float(np.sqrt(np.mean([(p - y) ** 2 for _, y, p in pts])))
        near = min(pts, key=lambda t: abs(t[0] - EOL_CUT))
        eol[eng] = float(abs(near[2] - near[1]))
    return rmse, eol


def boot_ci(d, seed):
    rng = np.random.default_rng(seed)
    draws = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--module', required=True)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    base = by_cell(load(Path(a.baseline)))
    mod = by_cell(load(Path(a.module)))

    print(f'gate1.py sha256 = {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}')
    print('convention: delta = baseline - module (POSITIVE = module better)')
    print(f'not degraded: delta_EOL_MAE >= {NOT_DEGRADED:+.2f}   '
          f'positive gain: delta_RMSE > 0 and win rate > 0.50\n')

    cells, per_method = {}, defaultdict(dict)
    for method in METHODS:
        for subset in SUBSETS:
            d_rmse, d_eol, wins, total = [], [], 0, 0
            for seed in SEEDS:
                key = (method, subset, seed)
                if key not in base or key not in mod:
                    raise SystemExit(f'missing cell {key} '
                                     f'(baseline={key in base}, module={key in mod})')
                rb, eb = cell_metrics(base[key])
                rm_, em = cell_metrics(mod[key])
                eng = sorted(set(rb) & set(rm_))
                if len(eng) != len(rb) or len(eng) != len(rm_):
                    raise SystemExit(f'{key}: engine sets differ '
                                     f'(base={len(rb)} mod={len(rm_)} common={len(eng)})')
                d_rmse.append(np.array([rb[u] - rm_[u] for u in eng]))
                d_eol.append(np.array([eb[u] - em[u] for u in eng]))
                wins += sum(1 for u in eng if rm_[u] < rb[u])
                total += len(eng)
            dr = np.mean(np.stack(d_rmse, axis=0), axis=0)
            de = np.mean(np.stack(d_eol, axis=0), axis=0)
            lo_r, hi_r = boot_ci(dr, BOOT_SEED)
            lo_e, hi_e = boot_ci(de, BOOT_SEED + 1)
            win = wins / total
            ok_eol = float(de.mean()) >= NOT_DEGRADED
            ok_gain = (float(dr.mean()) > 0) and (win > 0.50)
            cells[(method, subset)] = dict(
                delta_rmse=float(dr.mean()), rmse_ci=[lo_r, hi_r],
                delta_eol_mae=float(de.mean()), eol_ci=[lo_e, hi_e],
                win_rate=float(win), not_degraded=bool(ok_eol), positive_gain=bool(ok_gain),
                passed=bool(ok_eol and ok_gain), n_engines=int(len(eng)))
            per_method[method][subset] = cells[(method, subset)]
            c = cells[(method, subset)]
            print(f'{method:11} {subset}  dRMSE {c["delta_rmse"]:+.3f} '
                  f'[{lo_r:+.3f},{hi_r:+.3f}]  dEOL {c["delta_eol_mae"]:+.3f} '
                  f'[{lo_e:+.3f},{hi_e:+.3f}]  win {c["win_rate"]:.0%}  '
                  f'{"PASS" if c["passed"] else "FAIL"}'
                  f'{"  (eol)" if not c["not_degraded"] else ""}'
                  f'{"  (gain)" if not c["positive_gain"] else ""}')

    n_pass = sum(1 for m in METHODS
                 if all(per_method[m].get(s, {}).get('passed') for s in SUBSETS))
    print(f'\nbackbones passing on BOTH subsets: {n_pass}/{len(METHODS)} '
          f'(need >= {MIN_PASS_METHODS})')
    verdict = 'PASS' if n_pass >= MIN_PASS_METHODS else 'FAIL'
    print(f'GATE 1: {verdict}')

    if a.json:
        Path(a.json).write_text(json.dumps(dict(
            convention='delta = baseline - module (positive = module better)',
            thresholds=dict(not_degraded_ge=NOT_DEGRADED, win_rate_gt=0.50,
                            min_pass_methods=MIN_PASS_METHODS),
            cells={f'{m}/{s}': v for (m, s), v in cells.items()},
            n_pass=n_pass, verdict=verdict,
        ), indent=1), encoding='utf-8')
        print(f'wrote {a.json}')
    return 0 if verdict == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
