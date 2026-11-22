"""Paper C Phase 1: M1 CADF screening on FD001/FD004, 5-fold CV."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.train import run_fold

ARMS = ['gru', 'gru_cap', 'm1', 'm1_static']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p1_m1.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints_p1')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['subset'], r['model'], r['fold']) for r in rows}
    for subset in ['FD001', 'FD004']:
        for arm in ARMS:
            for fold in range(5):
                key = (subset, arm, fold)
                if key in done:
                    print('SKIP', key)
                    continue
                print('RUN', key, flush=True)
                r = run_fold(subset, arm, 50, fold, epochs=a.epochs,
                             ckpt_dir=a.ckpt_dir)
                rows.append(r)
                out.write_text(json.dumps(rows, indent=1))
    print('\nSUMMARY M1')
    for subset in ['FD001', 'FD004']:
        for arm in ARMS:
            vals = [r['engine_mean_rmse'] for r in rows
                    if r['subset'] == subset and r['model'] == arm]
            if vals:
                print(f'{subset} {arm}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
