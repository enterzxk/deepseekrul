# Provenance (header added 2026-11-13 by zxk; the code below is unchanged apart from the
# ledger path, which moved to DS02_pilot/ when the pilot batch was archived).
#   produced: N0 and N1 at 40 / 200 / 800 passes: 15.23/2.48/2.68 and 21.71/13.04/13.23
#   output:   stdout only; the counter-evidence that more passes do help, before standardisation
#   run:      python -m paper_b.ncmapss.probes.budget_probe   (from the repo root)
"""Is the neural rung undertrained, or broken?

Load one fold once, then train the same rung with increasing budgets and report
holdout RMSE. If RMSE keeps falling towards the linear-ridge reference (~9.3
mean, 5.2 on this fold) the budget was the problem. If it plateaus far above,
the implementation is wrong.
"""
from __future__ import annotations

import numpy as np
import torch

from paper_b.ncmapss.train import fit_rung, load_ds02, load_frozen

H5 = 'data/NCMAPSS/extracted/N-CMAPSS_DS02-006.h5'
DEV = (2, 5, 10, 16, 18, 20)
HELD = 2
BUDGETS = (40, 200, 800)


def main():
    cfg = load_frozen()
    train_units = [u for u in DEV if u != HELD]
    data = load_ds02(H5, train_units, cfg)
    y = np.asarray(data['y'])
    ho = ~np.asarray(data['is_train'], bool)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'fold holdout={HELD} n_train={int((~ho).sum())} n_hold={int(ho.sum())} dev={dev}', flush=True)
    for rung in ('N0', 'N1'):
        for ep in BUDGETS:
            net, res = fit_rung(rung, data, cfg, seed=101, device=dev, epochs=ep, batch=64)
            net.eval()
            with torch.no_grad():
                out = net(torch.from_numpy(data['x']).to(dev),
                          mask=torch.from_numpy(data['mask']).to(dev),
                          cond=torch.from_numpy(data['cond']).to(dev))
            p = out['prediction'].float().cpu().numpy().reshape(-1)
            err = p[ho] - y[ho]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            tr_err = p[~ho] - y[~ho]
            tr_rmse = float(np.sqrt(np.mean(tr_err ** 2)))
            print(f'{rung} passes={ep:4d}  holdout_rmse={rmse:7.3f}  train_rmse={tr_rmse:7.3f}  '
                  f'bias={float(np.mean(err)):6.2f}', flush=True)


if __name__ == '__main__':
    main()
