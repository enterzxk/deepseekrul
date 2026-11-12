# Provenance (header added 2026-11-13 by zxk; the code below is unchanged apart from the
# ledger path, which moved to DS02_pilot/ when the pilot batch was archived).
#   produced: N1/N2/N3 at 40 / 200 / 800 passes: 21.46/17.42/15.42, 8.36/6.81/13.99, 23.40/6.93/6.50
#   output:   stdout only; the 200-pass budget choice (N2 peaks at 200, memorises by 800)
#   run:      python -m paper_b.ncmapss.probes.norm_probe2   (from the repo root)
"""N1 with standardised target at several budgets: is it budget-bound at all?"""
from __future__ import annotations

import numpy as np
import torch

from paper_b.ncmapss.train import fit_rung, load_ds02, load_frozen

H5 = 'data/NCMAPSS/extracted/N-CMAPSS_DS02-006.h5'
DEV = (2, 5, 10, 16, 18, 20)
HELD = 2


def main():
    cfg = load_frozen()
    train_units = [u for u in DEV if u != HELD]
    data = load_ds02(H5, train_units, cfg)
    y = np.asarray(data['y'], np.float64)
    tr = np.asarray(data['is_train'], bool)
    ho = ~tr
    mu, sd = float(y[tr].mean()), float(y[tr].std())
    data_n = dict(data)
    data_n['y'] = ((y - mu) / sd).astype(np.float32)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'fold holdout={HELD} mu={mu:.2f} sd={sd:.2f}', flush=True)
    for rung in ('N1', 'N2', 'N3'):
        for ep in (40, 200, 800):
            net, res = fit_rung(rung, data_n, cfg, seed=101, device=dev, epochs=ep, batch=64)
            net.eval()
            with torch.no_grad():
                out = net(torch.from_numpy(data_n['x']).to(dev),
                          mask=torch.from_numpy(data_n['mask']).to(dev),
                          cond=torch.from_numpy(data_n['cond']).to(dev))
            p = out['prediction'].float().cpu().numpy().reshape(-1) * sd + mu
            err = p[ho] - y[ho]
            perr = p[~ho] - y[~ho]
            print(f'{rung} ep={ep:4d}  holdout={float(np.sqrt(np.mean(err**2))):7.3f}  '
                  f'train={float(np.sqrt(np.mean(perr**2))):7.3f}  bias={float(np.mean(err)):6.2f}',
                  flush=True)


if __name__ == '__main__':
    main()
