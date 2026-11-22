"""Paper C Phase 0: backbone selection on FD001/FD004, 5-fold CV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper_c.train import run_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/paper_c/p0_selection.json')
    ap.add_argument('--ckpt-dir', default='paper_c/checkpoints')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out.exists():
        rows = json.loads(out.read_text())
    done = {(r['subset'], r['model'], r['L'], r['fold']) for r in rows}
    for subset in ['FD001', 'FD004']:
        for model in ['gru', 'tcn']:
            for L in [30, 50]:
                for fold in range(a.folds):
                    key = (subset, model, L, fold)
                    if key in done:
                        print('SKIP', key)
                        continue
                    print('RUN', key, flush=True)
                    r = run_fold(subset, model, L, fold, epochs=a.epochs,
                                 ckpt_dir=a.ckpt_dir)
                    rows.append(r)
                    out.write_text(json.dumps(rows, indent=1))
    # summary
    print('\nSUMMARY (engine_mean mean±sd over folds)')
    for subset in ['FD001', 'FD004']:
        for model in ['gru', 'tcn']:
            for L in [30, 50]:
                vals = [r['engine_mean_rmse'] for r in rows
                        if r['subset'] == subset and r['model'] == model and r['L'] == L]
                if vals:
                    print(f'{subset} {model} L{L}: {np.mean(vals):.3f} ± {np.std(vals):.3f}')


if __name__ == '__main__':
    main()
