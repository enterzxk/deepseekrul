#!/usr/bin/env python3
"""Is the E5 `none` arm a true identity baseline under the new h8?

`lr_mode='none'` builds the same graph as E0 (no extra modules, same RNG
consumption), and the pipeline was measured to be deterministic given a seed
(an earlier retrain reproduced 7.6728326636558535 exactly). So E5's `none` arm
should reproduce the stage-1 E0 cells bit for bit.

That matters because it decides how to read E5:

  * identical  -> `none` really is the same model at the new h8, so the E5
                  within-pairing is clean and the new h8 did not change the
                  baseline's behaviour;
  * different  -> the comparison inside E5 is still valid (all four arms share
                  the new h8), but we may NOT describe `none` as "the E0 model",
                  and the size of the shift has to be reported.

Read-only. Resolves cells through `_cells.py` (split_seed filter), never by
name prefix alone.

Usage
-----
    python scripts/audit_e5_none.py --runs <runs> --seeds 101,202,303
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._cells import try_resolve  # noqa: E402

KEY = 'val_engine_mean_roots_rmse'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--seeds', default='101,202,303')
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    args = ap.parse_args()

    runs = Path(args.runs)
    seeds = [int(s) for s in args.seeds.split(',')]
    subsets = args.subsets.split(',')

    rows, identical, compared = [], 0, 0
    for seed in seeds:
        for subset in subsets:
            e0 = try_resolve(runs, 'E0', subset, seed, 17)
            none = try_resolve(runs, 'E5none', subset, seed, 17)
            if e0 is None or none is None:
                rows.append((subset, seed, None, None, 'not ready'))
                continue
            m0 = json.loads((Path(e0) / 'metrics.json').read_text())
            mn = json.loads((Path(none) / 'metrics.json').read_text())
            a, b = m0.get(KEY), mn.get(KEY)
            same = (a == b)
            compared += 1
            identical += int(same)
            rows.append((subset, seed, a, b, 'IDENTICAL' if same else f'diff {b - a:+.6f}'))

    print(f'{"subset":8} {"seed":>5} {"E0":>12} {"E5none":>12}  verdict')
    for subset, seed, a, b, verdict in rows:
        fa = '-' if a is None else f'{a:.6f}'
        fb = '-' if b is None else f'{b:.6f}'
        print(f'{subset:8} {seed:>5} {fa:>12} {fb:>12}  {verdict}')
    print()
    print(f'compared {compared}, identical {identical}')
    if compared == 0:
        print('VERDICT: nothing to compare yet')
    elif identical == compared:
        print('VERDICT: E5 none arm reproduces stage-1 E0 exactly -> it is a true '
              'identity baseline under the new h8')
    else:
        print('VERDICT: E5 none arm does NOT reproduce E0; report the shift and do '
              'not describe none as "the E0 model"')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
