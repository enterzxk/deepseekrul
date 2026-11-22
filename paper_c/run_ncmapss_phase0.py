"""N-CMAPSS Phase 2.0: B_min selection across DS02/DS05/DS06/DS07."""
import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.ncmapss_train import run_fold_nc

SUBSETS = ['DS02', 'DS05', 'DS06', 'DS07']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p2_nc_selection.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints_nc')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r['subset'], r['model'], r['L'], r['fold']) for r in rows}
    for subset in SUBSETS:
        for model in ['gru', 'tcn']:
            for L in [30, 50]:
                for fold in range(5):
                    key = (subset, model, L, fold)
                    if key in done:
                        print('SKIP', key); continue
                    print('RUN', key, flush=True)
                    r = run_fold_nc(subset, model, L, fold, epochs=a.epochs,
                                    ckpt_dir=a.ckpt_dir)
                    rows.append(r); out.write_text(json.dumps(rows, indent=1))
    print('\nSUMMARY N-CMAPSS B_min')
    for subset in SUBSETS:
        for model in ['gru', 'tcn']:
            for L in [30, 50]:
                vals = [r['engine_mean_rmse'] for r in rows
                        if r['subset'] == subset and r['model'] == model and r['L'] == L]
                if vals:
                    print(f'{subset} {model} L{L}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
