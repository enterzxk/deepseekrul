"""G4 diagnostic: is the positive cycle->RUL response real, or a probe artifact?

The pre-registered G4 probe adds an ABSOLUTE +10 to the raw cycle and measures
d(prediction). Every post-norm arm came back positive (RUL rises with age),
which is the wrong sign; A0, the age-only control, came back correctly
negative. That means either the models learned the sign wrong, or the probe
leaves the regime the models were trained on.

This script separates those two. For every evidence run it reports:

  corr        Spearman(cycle_raw, y_true)          -- the direction in the data
  sweep       d in {-50,-25,-10,-5,+5,+10,+25,+50} -- monotonicity / saturation
  abs10       the registered probe, unchanged
  abs10_in    +10 then clamp to the OBSERVED val range (no extrapolation)
  rel10       +10% of the current cycle (scale free)
  by level    abs10 split at cycle < 250 / >= 250  -- where the sign lives
  zero        p(cycle) - p(0)                      -- total cycle contribution

Nothing is hardcoded: the run list is selected from the run directories with the
ledger rule (status=ok, run_id matches -W1(r\\d+)?$, earliest closed_utc), and
every config value comes from that run's own config.json.

Usage:
  python experiments/hc/tmp/g4_diag.py --keys
  python experiments/hc/tmp/g4_diag.py
"""
import argparse
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
SWEEP = [-50.0, -25.0, -10.0, -5.0, 5.0, 10.0, 25.0, 50.0]


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
    """Ledger rule: ok + W1(rN)? + earliest closed_utc, one per cell."""
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
    return {(v, s): best[(v, s, 101)][1] for (v, s) in
            [(v, s) for v in VARIANTS for s in SUBSETS] if (v, s, 101) in best}


def response(net, va, dev, raw, source, pert):
    d2 = dict(va)
    d2['cycle_hc'] = data_hc.cycle_vector(pert, source)
    p0 = predict_hc(net, va, dev)
    p1 = predict_hc(net, d2, dev)
    diff = p1 - p0
    return dict(mean=float(diff.mean()), abs=float(np.abs(diff).mean()),
                neg=float((diff < 0).mean()))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    runs = evidence_runs()
    print(f'evidence runs: {len(runs)}')
    if a.keys:
        cfg = json.loads((runs[('A2', 'FD001')] / 'config.json').read_text())
        print(sorted(cfg))
        return

    rows = []
    for v in VARIANTS:
        for s in SUBSETS:
            run = runs.get((v, s))
            if run is None:
                print(f'{v:4s} {s:6s} MISSING')
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

            rec = dict(variant=v, subset=s, run=run.name, source=src,
                       n=len(y), raw_min=float(raw.min()), raw_max=float(raw.max()),
                       y_mean=float(y.mean()),
                       corr=spearman(raw, y),
                       rmse=float(np.sqrt(((p - y) ** 2).mean())))
            for d in SWEEP:
                key = 'd%+g' % d
                r = response(net, va, dev, raw, src, raw + d)
                rec[key] = r
            rec['abs10'] = response(net, va, dev, raw, src, raw + 10.0)
            lo, hi = float(raw.min()), float(raw.max())
            rec['abs10_in'] = response(net, va, dev, raw, src,
                                       np.clip(raw + 10.0, lo, hi))
            rec['rel10'] = response(net, va, dev, raw, src, raw * 1.10)
            rec['zero'] = response(net, va, dev, raw, src, np.zeros_like(raw))

            # Split by cycle level without slicing the dict: push the +10 only on
            # the selected windows (others get delta 0, so their diff is 0) and
            # average over the selection. Identical to subsetting, no shape risk.
            for tag, sel in (('lo', raw < 250.0), ('hi', raw >= 250.0)):
                if sel.sum() < 5:
                    continue
                d2 = dict(va)
                d2['cycle_hc'] = data_hc.cycle_vector(
                    raw + 10.0 * sel.astype(np.float32), src)
                p1 = predict_hc(net, d2, dev)
                diff = (p1 - p)[sel]
                rec['split_' + tag] = dict(
                    n=int(sel.sum()), mean=float(diff.mean()),
                    abs=float(np.abs(diff).mean()),
                    neg=float((diff < 0).mean()))
            rows.append(rec)
            print('done %-4s %-6s %s' % (v, s, run.name), flush=True)

    out = Path('/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/tmp')
    (out / 'g4_diag.json').write_text(json.dumps(rows, indent=1))

    print('\n== registered probe vs in-range vs relative ==')
    print('%-4s %-6s %6s %5s %5s %5s | %7s %6s | %7s %6s | %7s %6s'
          % ('arm', 'subset', 'rmse', 'corr', 'rawmax', 'n',
             'abs10', 'neg', 'abs10in', 'neg', 'rel10', 'neg'))
    for r in rows:
        print('%-4s %-6s %6.3f %5.2f %5.0f %5d | %7.3f %6.2f | %7.3f %6.2f | %7.3f %6.2f'
              % (r['variant'], r['subset'], r['rmse'], r['corr'], r['raw_max'], r['n'],
                 r['abs10']['mean'], r['abs10']['neg'],
                 r['abs10_in']['mean'], r['abs10_in']['neg'],
                 r['rel10']['mean'], r['rel10']['neg']))

    print('\n== abs10 split by cycle level (raw<250 vs >=250) ==')
    for r in rows:
        lo = r.get('split_lo')
        hi = r.get('split_hi')
        f = lambda x: ('n=%d mean=%+.3f neg=%.2f' % (x['n'], x['mean'], x['neg'])) \
            if x else 'n<5'
        print('%-4s %-6s lo: %-32s hi: %s' % (r['variant'], r['subset'], f(lo), f(hi)))

    print('\n== sweep of mean d(pred) ==')
    hdr = ' '.join('%+g' % d for d in SWEEP)
    print('%-4s %-6s %s' % ('arm', 'subset', hdr))
    for r in rows:
        print('%-4s %-6s %s' % (r['variant'], r['subset'],
                                ' '.join('%+.2f' % r['d%+g' % d]['mean'] for d in SWEEP)))

    print('\n== per-arm mean over subsets ==')
    print('%-4s %8s %8s %8s %8s' % ('arm', 'abs10', 'abs10in', 'rel10', 'zero'))
    for v in VARIANTS:
        sub = [r for r in rows if r['variant'] == v]
        if not sub:
            continue
        m = lambda k: np.mean([r[k]['mean'] for r in sub])
        print('%-4s %+8.3f %+8.3f %+8.3f %+8.3f' % (v, m('abs10'), m('abs10_in'),
                                                     m('rel10'), m('zero')))


if __name__ == '__main__':
    main()
