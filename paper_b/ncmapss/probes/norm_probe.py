# Provenance (header added 2026-11-13 by zxk; the code below is unchanged apart from the
# ledger path, which moved to DS02_pilot/ when the pilot batch was archived).
#   produced: fold-local target standardisation: N0 15.23 -> 2.45 at the 40-pass budget
#   output:   stdout only; the evidence that the budget needed no change, cited in the budget decision
#   run:      python -m paper_b.ncmapss.probes.norm_probe   (from the repo root)
"""Does fold-local target standardisation let a 40-pass budget converge?

Same fold, same 51 features, same 40 passes x batch 64. Only the target is
standardised on the training units and the prediction is mapped back. If N0
reaches ~2.5 and N1 approaches the ridge reference (~5.2), the budget needs no
change at all and the frozen anchor stays.
"""
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
    print(f'fold holdout={HELD} n_train={int(tr.sum())} mu={mu:.3f} sd={sd:.3f}', flush=True)

    data_n = dict(data)
    data_n['y'] = ((y - mu) / sd).astype(np.float32)

    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for rung in ('N0', 'N1'):
        net, res = fit_rung(rung, data_n, cfg, seed=101, device=dev, epochs=40, batch=64)
        net.eval()
        with torch.no_grad():
            out = net(torch.from_numpy(data_n['x']).to(dev),
                      mask=torch.from_numpy(data_n['mask']).to(dev),
                      cond=torch.from_numpy(data_n['cond']).to(dev))
        p = out['prediction'].float().cpu().numpy().reshape(-1) * sd + mu
        err = p[ho] - y[ho]
        print(f'{rung} 40 passes + standardised y  holdout_rmse={float(np.sqrt(np.mean(err**2))):7.3f}  '
              f'bias={float(np.mean(err)):6.2f}', flush=True)


if __name__ == '__main__':
    main()
