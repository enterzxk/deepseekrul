"""N-CMAPSS Phase 2.1: M2-Lite screening."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.ncmapss_m2lite import run_fold_nc_m2lite

SUBSETS = ['DS02', 'DS05', 'DS06', 'DS07']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p2_nc_m2lite.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints_nc_m2lite')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['subset'], r['model'], r['fold']) for r in rows}
    for subset in SUBSETS:
        for fold in range(5):
            key = (subset, 'm2_lite', fold)
            if key in done:
                print('SKIP', key); continue
            print('RUN', key, flush=True)
            r = run_fold_nc_m2lite(subset, 50, fold, epochs=a.epochs,
                                   ckpt_dir=a.ckpt_dir)
            rows.append(r); out.write_text(json.dumps(rows, indent=1))
    print('\nSUMMARY N-CMAPSS M2-Lite')
    for subset in SUBSETS:
        vals = [r['engine_mean_rmse'] for r in rows if r['subset'] == subset]
        if vals:
            print(f'{subset}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
