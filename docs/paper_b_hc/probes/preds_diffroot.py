"""Root-cause the 5e-9 gap between stored preds and recomputed preds.

The bitwise check reported 24/28 cells exact and 4/28 differing by <=5e-9, with
the two recomputed passes identical (so it is not kernel nondeterminism). This
script decides between the two remaining mechanisms:

  (a) SERIALISATION -- the CSV stores fewer digits than the value needs, so the
      parsed number differs from the recomputed one by the rounding of the
      stored string.
  (b) MODEL -- the stored value is exact but the recomputed value differs, i.e.
      the predictions really changed.

It does so by printing, for the worst row of each affected cell, the raw CSV
string next to the parsed and the recomputed value, and by checking whether the
difference equals the decimal rounding of the stored string.

Reads only. Usage: python docs/paper_b_hc/probes/preds_diffroot.py
"""
import csv
import json
from decimal import Decimal
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


def decimals(s):
    return len(s.split('.')[1]) if '.' in s else 0


def main():
    dev = torch.device('cuda:0')
    runs = ledger_runs()
    worst_rows = []
    print('%-4s %-6s %6s %8s %10s %-22s %-22s'
          % ('arm', 'subset', 'ndec', 'argmax', 'diff', 'csv_string', 'recomputed'))
    for (v, s), rid in sorted(runs.items()):
        run = OUT / rid
        with (run / 'preds.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        raw = [r['y_pred'] for r in rows]
        parsed = np.array([float(x) for x in raw], np.float64)
        cfg = json.loads((run / 'config.json').read_text())
        cfg['data_dir'] = str(ROOT / 'data/CMAPSS')
        builder = build_hc_scaled if cfg.get('scale_mode') == 'cal' else build_hc
        net = builder(cfg)
        net.load_state_dict(torch.load(run / 'ckpt' / 'best.pt',
                                       map_location='cpu', weights_only=True))
        net.to(dev).eval()
        ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                           cfg['val_cuts'], cycle_source=cfg['cycle_source'])
        p = predict_hc(net, ds['val'], dev).astype(np.float64)
        d = np.abs(p - parsed)
        j = int(np.argmax(d))
        if d.max() > 0:
            worst_rows.append(dict(arm=v, subset=s, run=rid, i=j,
                                   csv=raw[j], parsed=float(parsed[j]),
                                   recomputed=float(p[j]), diff=float(d[j]),
                                   ndec=decimals(raw[j])))
            print('%-4s %-6s %6d %8d %10.3e %-22s %-22s'
                  % (v, s, decimals(raw[j]), j, d[j], raw[j], repr(float(p[j]))))

    print('\n%d cells with a non-zero diff' % len(worst_rows))
    for r in worst_rows:
        # rounding radius of the stored string: half of the last stored digit
        radius = 0.5 * 10.0 ** (-r['ndec'])
        verdict = 'SERIALISATION' if r['diff'] <= radius * 1.01 else 'MODEL'
        exact = Decimal(r['csv']) == Decimal(repr(r['recomputed']))
        print('  %s %s: diff=%.3e  last-digit radius=%.3e  -> %s  (string==repr: %s)'
              % (r['arm'], r['subset'], r['diff'], radius, verdict, exact))

    # If serialisation: every stored string must round to the recomputed float32
    # at the stored precision, i.e. the residual is bounded by half a unit in the
    # last printed digit. A single MODEL case would break that bound.
    bad = [r for r in worst_rows
           if r['diff'] > 0.5 * 10.0 ** (-r['ndec']) * 1.01]
    print('\nVERDICT: %s' % ('serialisation rounding of the stored decimal'
                             if not bad else 'at least one cell is a model difference'))

    # Decisive test: the model's own output precision is float32 (bf16 widened).
    # If parsing the stored string and casting back to float32 reproduces the
    # recomputed value exactly, the CSV is lossless at that precision and the
    # float64 gap is an artefact of comparing at higher precision.
    print('\nfloat32 round-trip over all 28 cells:')
    n_rt = n_tot = 0
    for (v, s), rid in sorted(runs.items()):
        run = OUT / rid
        with (run / 'preds.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        parsed = np.array([float(r['y_pred']) for r in rows], np.float64)
        cfg = json.loads((run / 'config.json').read_text())
        cfg['data_dir'] = str(ROOT / 'data/CMAPSS')
        builder = build_hc_scaled if cfg.get('scale_mode') == 'cal' else build_hc
        net = builder(cfg)
        net.load_state_dict(torch.load(run / 'ckpt' / 'best.pt',
                                       map_location='cpu', weights_only=True))
        net.to(dev).eval()
        ds = data_hc.build(cfg['subset'], cfg['data_dir'], cfg['span'],
                           cfg['val_cuts'], cycle_source=cfg['cycle_source'])
        p = predict_hc(net, ds['val'], dev)
        same = int((np.float32(parsed) == p.astype(np.float32)).sum())
        n_rt += same
        n_tot += len(p)
        if same != len(p):
            print('  %s %s: %d/%d identical at float32' % (v, s, same, len(p)))
    print('  %d/%d predictions identical at float32' % (n_rt, n_tot))


if __name__ == '__main__':
    main()
