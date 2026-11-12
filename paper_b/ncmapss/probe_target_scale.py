"""Decisive probe: is the DS02 under-training an optimisation-geometry problem?

Hypothesis. The nets are not failing because 40 passes is too few *per se*, but
because the output head starts near 0 and Adam moves a parameter by roughly the
learning rate per step: at lr=1e-3, 240 updates can shift the output by ~0.24
while the target lives around 40 cycles. A linear/ridge fit has no such problem,
which is exactly the observed pattern (ridge ~9.3, nets ~21.7, N0 MLP ~15.2).

Test. On fold 1, train the same models twice under the same 40-pass budget:
  (a) raw target, exactly as run_loeo does;
  (b) target standardised on the training fold, inverted at prediction.
If (b) approaches the ridge level while (a) stays at the mean-prediction level,
the cause is the optimisation geometry, not missing information and not a bug in
the features.

    python -m paper_b.ncmapss.probe_target_scale --h5 data/NCMAPSS/extracted/N-CMAPSS_DS02-006.h5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train import build, fit_rung, load_ds02, load_frozen

DEV = [2, 5, 10, 16, 18, 20]


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def train_and_score(rung, data, cfg, seed, dev, scale_y):
    d = dict(data)
    y = np.asarray(data['y'], np.float64)
    tr = np.asarray(data['is_train'], bool)
    mu, sd = (float(y[tr].mean()), float(y[tr].std())) if scale_y else (0.0, 1.0)
    d['y'] = ((y - mu) / sd).astype(np.float32)
    net, rep = fit_rung(rung, d, cfg, seed=seed, device=dev, epochs=40, batch=64)
    ho = ~tr
    x = torch.from_numpy(data['x'][ho]).float()
    mask = torch.from_numpy(data['mask'][ho]).float()
    cond = torch.from_numpy(data['cond'][ho])
    if rung == 'N0':
        x = x[..., -1:]
    net.eval()
    with torch.no_grad():
        p = net(x.to(dev), mask=mask.to(dev), cond=cond.to(dev))['prediction']
    p = p.reshape(-1).cpu().numpy().astype(np.float64) * sd + mu
    return _rmse(p, y[ho]), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--holdout', type=int, default=2)
    ap.add_argument('--out', default='experiments/ncmapss/probe_target_scale.json')
    a = ap.parse_args()
    cfg = load_frozen()
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_units = [u for u in DEV if u != a.holdout]
    data = load_ds02(Path(a.h5), train_units, cfg)
    y = np.asarray(data['y'], np.float64)
    tr = np.asarray(data['is_train'], bool)
    print(f'fold holdout={a.holdout}  y_train mean={y[tr].mean():.2f} sd={y[tr].std():.2f} '
          f'range=[{y.min():.0f},{y.max():.0f}]')
    rows = []
    for rung in ('N0', 'N1'):
        for scale in (False, True):
            r, rep = train_and_score(rung, data, cfg, 101, dev, scale)
            rows.append(dict(rung=rung, target_scaled=scale, rmse=r,
                             holdout_min=rep['final_holdout_loss'],
                             holdout_min_epoch=rep['history'][-1]['epoch']))
            print(f'{rung} scaled={scale}: holdout RMSE {r:.3f}')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
