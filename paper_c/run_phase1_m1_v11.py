"""Paper C Phase 1 v1.1: M1 CADF variants on FD004 (one revision)."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.train import run_fold

ARMS = ['m1_k5', 'm1_k10', 'm1_a8']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p1_m1_v11.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints_p1_v11')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['subset'], r['model'], r['fold']) for r in rows}
    for arm in ARMS:
        for fold in range(5):
            key = ('FD004', arm, fold)
            if key in done:
                print('SKIP', key); continue
            print('RUN', key, flush=True)
            r = run_fold('FD004', arm, 50, fold, epochs=a.epochs,
                         ckpt_dir=a.ckpt_dir)
            rows.append(r); out.write_text(json.dumps(rows, indent=1))
    print('\nSUMMARY M1 v1.1 FD004')
    for arm in ARMS:
        vals = [r['engine_mean_rmse'] for r in rows if r['model'] == arm]
        if vals:
            print(f'{arm}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
