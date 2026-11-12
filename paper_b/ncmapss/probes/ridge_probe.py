# Provenance (header added 2026-11-13 by zxk; the code below is unchanged apart from the
# ledger path, which moved to DS02_pilot/ when the pilot batch was archived).
#   produced: ridge baseline: six-fold RMSE 5.19 / 14.51 / 7.73 / 13.81 / 4.97 / 9.33 (mean ~9.3), the number that stopped the first 60-cell run
#   output:   stdout only; cited in REPRODUCIBILITY_AUDIT.md (linear ridge baseline)
#   run:      python -m paper_b.ncmapss.probes.ridge_probe   (from the repo root)
"""Decisive diagnostic: can a plain ridge on the same 51 features beat the clock?

If ridge (linear, same features, same fold) lands far below N1/N2/N3, the neural
rungs are undertrained or buggy and the 60-cell run must not start. If ridge is
no better than N0, the features simply do not carry the signal on 6 units and
the result is a finding, not a bug.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from paper_b.ncmapss.train import load_ds02, load_frozen

H5 = 'data/NCMAPSS/extracted/N-CMAPSS_DS02-006.h5'
DEV = (2, 5, 10, 16, 18, 20)


def ridge_fit(X, y, lam=1.0):
    X1 = np.concatenate([X, np.ones((len(X), 1), np.float64)], axis=1)
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    w = np.linalg.solve(A, X1.T @ y)
    return w


def ridge_pred(w, X):
    X1 = np.concatenate([X, np.ones((len(X), 1), np.float64)], axis=1)
    return X1 @ w


def main():
    cfg = load_frozen()
    out = []
    for held in DEV:
        train_units = [u for u in DEV if u != held]
        data = load_ds02(H5, train_units, cfg)
        x = np.asarray(data['x'], np.float64)      # (n, span, 51)
        y = np.asarray(data['y'], np.float64)      # (n,)
        tr = np.asarray(data['is_train'], bool)
        # last timestep = the current flight's statistics
        X = x[:, -1, :]
        w = ridge_fit(X[tr], y[tr])
        pred = ridge_pred(w, X[~tr])
        err = pred - y[~tr]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        # clock-only reference: fit RUL ~ a + b*cycle on the same fold
        # (cycle is the last column of the 51-block, already scaled)
        cyc = X[:, -1]
        c1 = np.stack([cyc[tr], np.ones(tr.sum())], axis=1)
        cw = np.linalg.lstsq(c1, y[tr], rcond=None)[0]
        cpred = cyc[~tr] * cw[0] + cw[1]
        crmse = float(np.sqrt(np.mean((cpred - y[~tr]) ** 2)))
        out.append(dict(held=held, n_tr=int(tr.sum()), n_te=int((~tr).sum()),
                        ridge_rmse=rmse, clock_rmse=crmse,
                        y_mean_tr=float(y[tr].mean())))
        print(f'holdout {held}: ridge={rmse:.3f}  clock={crmse:.3f}  '
              f'n_tr={int(tr.sum())} n_te={int((~tr).sum())}', flush=True)
    json.dump(out, open('experiments/ncmapss_ridge_probe.json', 'w'), indent=2)
    print('wrote experiments/ncmapss_ridge_probe.json')


if __name__ == '__main__':
    sys.exit(main())
