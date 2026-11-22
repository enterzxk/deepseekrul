"""Paper C Phase 1: M3 CD-RCL screening (beta vs vanilla NLL)."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.m3 import run_fold_m3

ARMS = ['beta', 'vanilla']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p1_m3.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints_p1_m3')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['subset'], r['model'], r['fold']) for r in rows}
    for subset in ['FD001', 'FD004']:
        for arm in ARMS:
            for fold in range(5):
                key = (subset, f'm3_{arm}', fold)
                if key in done:
                    print('SKIP', key); continue
                print('RUN', key, flush=True)
                r = run_fold_m3(subset, fold, epochs=a.epochs, loss_name=arm,
                                ckpt_dir=a.ckpt_dir)
                rows.append(r); out.write_text(json.dumps(rows, indent=1))
    print('\nSUMMARY M3')
    for subset in ['FD001', 'FD004']:
        for arm in ARMS:
            vals = [r['engine_mean_rmse'] for r in rows
                    if r['subset'] == subset and r['model'] == f'm3_{arm}']
            if vals:
                print(f'{subset} m3_{arm}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
