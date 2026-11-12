"""G4 second pass: is the positive cycle slope the model, or the val sampling?

`make_val_windows` takes a few CUT FRACTIONS of each training engine, so the
window set is not a sample of the training distribution. If cut fractions are
fixed per engine, then for a fixed fraction c = f*n and y = (1-f)*n both grow
with engine life n -- a positive c-y confound across engines that no model can
remove without knowing n. The training set, by contrast, walks every cycle of
every engine, so within-engine y falls as c rises.

This pass separates the two by computing, per (arm, subset):

  pooled      Spearman(c, y) and Spearman(c, pred) over all val windows
  within      the same inside each engine (windows grouped by unit), averaged

If `pooled` is positive while `within` is strongly negative, the probe is
confounded by the sampling design and a positive +10 response is expected from
a correct model. If `within` is also positive, the model really did learn the
wrong sign.

Usage: python experiments/hc/tmp/g4_probe2.py
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
PAT = re.compile(r'^HC-C-(A\d\w?)-((?:FD00[1-4]))-s(\d+)-W1(r\d+)?$')
VARIANTS = ['A0', 'A1', 'A2', 'A3a', 'A4', 'A5', 'A3p']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']


def closed_utc(run):
    f = run / 'close.json'
    if not f.exists():
        return None
    try:
        c = json.loads(f.read_text())
    except Exception:
        return None
    return c.get('closed_utc') if c.get('status') == 'ok' else None


def evidence_runs():
    best = {}
    for run in sorted(OUT.glob('HC-C-*')):
        m = PAT.match(run.name)
        if not m:
            continue
        t = closed_utc(run)
        if t is None:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        if key not in best or t < best[key][0]:
            best[key] = (t, run)
    return {(v, s): best[(v, s, 101)][1] for v in VARIANTS for s in SUBSETS
            if (v, s, 101) in best}


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den else float('nan')


def within(groups, c, v):
    out = []
    for sl in groups:
        if sl.stop - sl.start >= 3:
            out.append(spear(c[sl], v[sl]))
    return float(np.mean(out)) if out else float('nan'), len(out)


def main():
    dev = torch.device('cuda:0')
    runs = evidence_runs()
    print('%-4s %-6s %5s %6s %6s %7s %7s %7s %7s'
          % ('arm', 'subset', 'nwin', 'nunit', 'y_mean', 'pool_c_y',
             'within_c_y', 'pool_c_p', 'within_c_p'))
    agg = {}
    for v in VARIANTS:
        for s in SUBSETS:
            run = runs.get((v, s))
            if run is None:
                continue
            cfg = json.loads((run / 'config.json').read_text())
            src = cfg['cycle_source']
            net = build_hc(cfg)
            net.load_state_dict(torch.load(run / 'ckpt' / 'best.pt',
                                           map_location='cpu', weights_only=True))
            net.to(dev).eval()
            ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                               cfg['val_cuts'], cycle_source=src)
            va = ds['val']
            raw = np.asarray(va['cycle_raw'], np.float32)
            y = np.asarray(va['y'], np.float32)
            p = predict_hc(net, va, dev)
            k = len(cfg['val_cuts'])
            groups = [slice(i, min(i + k, len(raw))) for i in range(0, len(raw), k)]
            wy, nunit = within(groups, raw, y)
            wp, _ = within(groups, raw, p)
            print('%-4s %-6s %5d %6d %6.1f %7.2f %7.2f %7.2f %7.2f'
                  % (v, s, len(raw), nunit, y.mean(), spear(raw, y),
                     wy, spear(raw, p), wp))
            agg.setdefault(v, []).append((wy, wp))

    print('\n== per-arm mean of within-engine Spearman ==')
    print('%-4s %12s %12s' % ('arm', 'within_c_y', 'within_c_p'))
    for v in VARIANTS:
        if v not in agg:
            continue
        a = np.array(agg[v], np.float64)
        print('%-4s %12.2f %12.2f' % (v, np.nanmean(a[:, 0]), np.nanmean(a[:, 1])))


if __name__ == '__main__':
    main()
