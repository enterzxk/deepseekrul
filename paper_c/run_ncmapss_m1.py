"""N-CMAPSS Phase 2.1: M1 v1.2 screening."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.ncmapss_m1 import run_fold_nc_m1

SUBSETS = ['DS02', 'DS05', 'DS06', 'DS07']
ARMS = ['m1_v12_win', 'm1_v12_eng', 'm1_v12_static']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p2_nc_m1.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints_nc_m1')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['subset'], r['model'], r['fold']) for r in rows}
    for subset in SUBSETS:
        for arm in ARMS:
            for fold in range(5):
                key = (subset, arm, fold)
                if key in done:
                    print('SKIP', key); continue
                print('RUN', key, flush=True)
                r = run_fold_nc_m1(subset, arm, 50, fold, epochs=a.epochs,
                                   ckpt_dir=a.ckpt_dir)
                rows.append(r); out.write_text(json.dumps(rows, indent=1))
    print('\nSUMMARY N-CMAPSS M1')
    for subset in SUBSETS:
        for arm in ARMS:
            vals = [r['engine_mean_rmse'] for r in rows
                    if r['subset'] == subset and r['model'] == arm]
            if vals:
                print(f'{subset} {arm}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
