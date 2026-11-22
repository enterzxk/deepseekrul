"""Paper C Phase 1: M2 CMST screening (full / gate-only / pool-only)."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.m2 import run_fold_m2

ARMS = ['m2', 'm2_gate_only', 'm2_pool_only']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p1_m2.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints_p1_m2')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['subset'], r['model'], r['fold']) for r in rows}
    for subset in ['FD001', 'FD004']:
        for arm in ARMS:
            for fold in range(5):
                key = (subset, arm, fold)
                if key in done:
                    print('SKIP', key); continue
                print('RUN', key, flush=True)
                r = run_fold_m2(subset, fold, epochs=a.epochs, arm=arm,
                                ckpt_dir=a.ckpt_dir)
                rows.append(r); out.write_text(json.dumps(rows, indent=1))
    print('\nSUMMARY M2')
    for subset in ['FD001', 'FD004']:
        for arm in ARMS:
            vals = [r['engine_mean_rmse'] for r in rows
                    if r['subset'] == subset and r['model'] == arm]
            if vals:
                print(f'{subset} {arm}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
