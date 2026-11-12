"""G4 third pass: is dRUL/dc > 0 given the sensors actually correct?

Pass 1 showed the registered +10 probe returns a POSITIVE response for every
cycle arm (A2 +1.14, A3a +1.21, A5 +1.34, A3p +1.43 per +10) while the age-only
control A0 returns -3.96. Pass 2 showed every arm's prediction still falls
monotonically with age WITHIN an engine (Spearman -0.97..-0.98). Both are true
at once only if the model uses the sensors for the age->RUL trend and the
explicit cycle input carries a separate, positive conditional effect.

That effect is not necessarily a bug. Conditioning on the sensors means holding
the health signature fixed; at a fixed signature, the engine that has reached a
larger age is the one that degrades more slowly, so it has MORE remaining life.
The registered G4 criterion (frac_negative >= 0.5 at fixed sensors) therefore
assumes a negative sign where the conditional sign is plausibly positive.

This pass tests that directly. Take the sensor-only arm A1 as the sensor
summary, then fit, by OLS on the training windows (the distribution the models
actually learned from) and on the val windows:

    y ~ 1 + p_A1 + c

and report the coefficient on c. If it is positive and close to the observed
probe slope, the cycle arms learned the correct conditional effect and G4's
null is mis-specified.

Usage: python experiments/hc/tmp/g4_probe3.py
"""
import json
import re
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import data_hc
from paper_b.hc.model_hc import build_hc
from paper_b.hc.train_hc import predict_hc

OUT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc')
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
PAT = re.compile(r'^HC-C-A1-(FD00[1-4])-s101-W1(r\d+)?$')


def pick(subset):
    best = None
    for run in sorted(OUT.glob('HC-C-A1-*')):
        m = PAT.match(run.name)
        if not m or m.group(1) != subset:
            continue
        f = run / 'close.json'
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        if c.get('status') != 'ok':
            continue
        t = c.get('closed_utc')
        if best is None or t < best[0]:
            best = (t, run)
    return best[1] if best else None


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ((y - pred) ** 2).sum() / ss if ss else float('nan')
    return beta, r2


def main():
    dev = torch.device('cuda:0')
    print('%-6s %8s %10s %10s %8s %8s'
          % ('subset', 'n_train', 'b_cycle', 'b_per10', 'r2', 'n_val'))
    for s in SUBSETS:
        run = pick(s)
        if run is None:
            print(f'{s:6s} MISSING')
            continue
        cfg = json.loads((run / 'config.json').read_text())
        net = build_hc(cfg)
        net.load_state_dict(torch.load(run / 'ckpt' / 'best.pt',
                                       map_location='cpu', weights_only=True))
        net.to(dev).eval()
        ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                           cfg['val_cuts'], cycle_source=cfg['cycle_source'])

        tr, va = ds['train'], ds['val']
        ptr = predict_hc(net, tr, dev)
        pva = predict_hc(net, va, dev)
        ctr = np.asarray(tr['cycle_raw'], np.float32)
        cva = np.asarray(va['cycle_raw'], np.float32)
        ytr = np.asarray(tr['y'], np.float32)
        yva = np.asarray(va['y'], np.float32)

        btr, r2tr = ols(np.column_stack([ptr, ctr]), ytr)
        bva, r2va = ols(np.column_stack([pva, cva]), yva)

        # Local conditional slope: inside a narrow band of the sensor-only
        # prediction, regress y on c. This is the quantity the +10 probe
        # measures, whereas the global OLS above averages over all health
        # levels. Bins with too few windows are dropped rather than reported.
        edges = np.quantile(ptr, np.linspace(0.0, 1.0, 11))
        num = den = 0.0
        for i in range(10):
            sel = (ptr >= edges[i]) & (ptr <= edges[i + 1])
            if sel.sum() < 200:
                continue
            b, _ = ols(ctr[sel, None], ytr[sel])
            num += float(b[1]) * float(sel.sum())
            den += float(sel.sum())
        loc = num / den if den else float('nan')

        print('%-6s %8d %10.4f %10.3f %8.3f %8d'
              % (s, len(ytr), btr[2], btr[2] * 10.0, r2tr, len(yva)))
        print('%-6s %8s %10.4f %10.3f %8.3f   local/10=%+.3f'
              % ('  val', '', bva[2], bva[2] * 10.0, r2va, loc * 10.0))


if __name__ == '__main__':
    main()
