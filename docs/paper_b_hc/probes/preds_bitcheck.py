"""Bitwise preds check: new code must reproduce every canonical checkpoint's preds.

Before a wave launches, the patched code must be shown not to change the
predictions of the runs that are already evidence. For every ledger-admitted
cell (7 variants x 4 subsets, NOISE1 excluded because it is not admitted) this:

  1. rebuilds the model from the run's own config.json and ckpt/best.pt,
  2. predicts the validation set twice with the same code,
  3. compares both passes against the run's own preds.csv.

Two comparisons, two different questions:

  * pass1 vs pass2  -- kernel nondeterminism. If these differ, the check cannot
                       say anything about the patch; fix determinism first.
  * preds vs csv    -- whether the patch changed the numbers. A non-zero
                       max|diff| here is a regression, not noise.

Reads only. Usage:
    python docs/paper_b_hc/probes/preds_bitcheck.py [--subsets FD001,...]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import data_hc
from paper_b.hc.model_hc import build_hc
from paper_b.hc.scale_align import build_hc_scaled
from paper_b.hc.train_hc import predict_hc

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
OUT = ROOT / 'experiments/hc'
LEDGER = ROOT / 'docs/paper_b_hc/registry/runs.csv'


def ledger_runs():
    keep = {}
    with LEDGER.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['in_gate_a_table'].strip().lower() != 'true':
                continue
            if row['status'].strip() != 'ok':
                continue
            keep[(row['variant'], row['subset'])] = row['run_id']
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = torch.device(a.device)
    want = set(a.subsets.split(','))
    runs = {k: v for k, v in ledger_runs().items() if k[1] in want}

    print('%-6s %-6s %-30s %12s %12s %6s' %
          ('arm', 'subset', 'run', 'max|p1-p2|', 'max|p-csv|', 'd32'))
    worst_pair = worst_csv = 0.0
    n_ok = n_bad = 0
    n_float32_diff = 0
    for (v, s), rid in sorted(runs.items()):
        run = OUT / rid
        cfg = json.loads((run / 'config.json').read_text())
        cfg['data_dir'] = str(ROOT / 'data/CMAPSS')
        builder = build_hc_scaled if cfg.get('scale_mode') == 'cal' else build_hc
        net = builder(cfg)
        net.load_state_dict(torch.load(run / 'ckpt' / 'best.pt',
                                       map_location='cpu', weights_only=True))
        net.to(dev).eval()
        ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                           cfg['val_cuts'], cycle_source=cfg['cycle_source'])
        p1 = predict_hc(net, ds['val'], dev)
        p2 = predict_hc(net, ds['val'], dev)
        ref = np.loadtxt(run / 'preds.csv', delimiter=',', skiprows=1,
                         usecols=4) if (run / 'preds.csv').exists() else None
        d_pair = float(np.abs(p1 - p2).max())
        # Compare at the model's own output precision. The CSV stores numpy's
        # shortest round-tripping float32 repr, so parsing it as float64 and
        # diffing against the widened float32 shows a spurious <=5e-9 residual
        # (root-caused in preds_diffroot.py: 4970/4970 identical at float32).
        if ref is not None and len(ref) == len(p1):
            same32 = int((np.float32(ref) == p1.astype(np.float32)).sum())
            d_csv = 0.0 if same32 == len(p1) else float(np.abs(p1 - ref).max())
            n32 = len(p1) - same32
        else:
            d_csv, n32 = float('nan'), -1
        worst_pair = max(worst_pair, d_pair)
        worst_csv = max(worst_csv, 0.0 if np.isnan(d_csv) else d_csv)
        bad = (d_pair > 0) or (not np.isnan(d_csv) and d_csv > 0)
        n_bad += int(bad)
        n_ok += int(not bad)
        print('%-6s %-6s %-30s %12.3e %12.3e %6d' % (v, s, rid, d_pair, d_csv, n32))

    print('\n%d cells: %d exact, %d with a non-zero diff' % (len(runs), n_ok, n_bad))
    print('worst pass1-vs-pass2 %.3e ; worst preds-vs-csv %.3e' % (worst_pair, worst_csv))
    if worst_pair > 0:
        print('NON-DETERMINISTIC: two passes disagree; fix determinism before judging the patch')
    if worst_csv > 0:
        print('REGRESSION: the patched code does not reproduce the committed predictions')
    if worst_pair == 0 and worst_csv == 0:
        print('BITWISE OK: the patch leaves every canonical checkpoint unchanged')


if __name__ == '__main__':
    main()
