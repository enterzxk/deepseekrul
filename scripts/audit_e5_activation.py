#!/usr/bin/env python3
"""Is the E5 branch actually awake in a trained checkpoint?

The last layer of `g` is zero-initialised (E5_PREREG.md §1), so the branch starts
as an exact identity. If it is still near zero after training, the branch never
learned anything -- and then H1's 5..14 spectrum clause reports "indistinguishable"
for a reason that has nothing to do with the rank constraint being useless.

This prints, per cell, the quantities §9.6 requires:

    |lam|         residual scale coefficient
    ||W_last||_F  norm of g's zero-initialised final layer (0 at init)

Read-only, CPU, no forward pass.

Usage
-----
    python scripts/audit_e5_activation.py --runs <runs> --seeds 101,202,303
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts._cells import try_resolve  # noqa: E402

ARMS = ('E5lr4', 'E5lr14', 'E5mlp')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--seeds', default='101,202,303')
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    args = ap.parse_args()

    runs = Path(args.runs)
    seeds = [int(s) for s in args.seeds.split(',')]
    subsets = args.subsets.split(',')

    print(f'{"arm":7} {"subset":7} {"seed":>5} {"|lam|":>10} {"||W_last||_F":>13}  verdict')
    awake = dormant = 0
    for arm in ARMS:
        for subset in subsets:
            for seed in seeds:
                cell = try_resolve(runs, arm, subset, seed, 17)
                if cell is None:
                    continue
                sd = torch.load(Path(cell) / 'best.pt', map_location='cpu',
                                weights_only=True)
                lam = abs(float(sd['lr.lam']))
                keys = [k for k in sd if k.startswith('lr.g.') and k.endswith('.weight')]
                w = max(float(sd[k].norm()) for k in keys) if keys else float('nan')
                # At init the final layer is exactly 0; anything at float noise
                # level means it never moved.
                is_awake = w > 1e-6
                awake += int(is_awake)
                dormant += int(not is_awake)
                print(f'{arm:7} {subset:7} {seed:>5} {lam:>10.3e} {w:>13.6e}  '
                      f'{"AWAKE" if is_awake else "DORMANT"}')
    print()
    print(f'awake {awake}, dormant {dormant}')
    if dormant:
        print('WARNING: a dormant cell must be reported as "branch did not learn" '
              '(E5_PREREG.md §9.6); its H1 spectrum clause is void, not evidence '
              'that the rank constraint is useless.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
