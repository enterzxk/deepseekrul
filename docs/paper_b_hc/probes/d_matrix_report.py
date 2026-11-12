"""Stage D four-arm comparison from the ledger and the per-run metrics.json.

Reads experiments/hc/d/ledger.csv (written by run_d_matrix.sh) and each run's
metrics.json, and prints the comparison the paper needs:

  * per (subset, arm): mean +/- sd over the seeds of the capped main metric and
    the three D4.6 readings that do not depend on the censoring boundary;
  * paired per-seed differences B1-B0, B2-B1, B3-B2 (the mechanism chain), with
    the per-subset sd so a difference smaller than the seed spread is visible as
    such rather than being read as an effect;
  * a coverage table naming every cell that is missing or not a closed formal run.

Design notes:

  * The main metric is the CAPPED one (min(r,125) label, clipped prediction).
    rmse_uncensored and rmse_uncapped are reported apart and never mixed in.
  * A cell whose metrics.json lacks frozen_sha256, or carries budget_override,
    is reported as NOT FORMAL and excluded from the summary -- a smoke run must
    not be averaged in with real cells.
  * No p-values on 3 seeds. The seed spread is printed and that is all it
    supports; the G2 line already established that a 3-seed sd is not a test.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from paper_b.hc.train_d import run_id

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/d'
ARMS = ['B0', 'B1', 'B2', 'B3']
SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
SEEDS = [101, 202, 303]
CHAIN = [('B1-B0', 'B1', 'B0'), ('B2-B1', 'B2', 'B1'), ('B3-B2', 'B3', 'B2')]
KEYS = ['val_engine_mean_rmse', 'rmse_uncensored', 'rmse_uncapped',
        'boundary_hit', 'frac_capped']


def load_cell(root, arm, subset, seed):
    p = os.path.join(root, run_id(arm, subset, seed), 'metrics.json')
    if not os.path.exists(p):
        return None, 'MISSING'
    with open(p, encoding='utf-8') as f:
        m = json.load(f)
    if m.get('budget_override') is not None:
        return m, 'SMOKE'
    if not m.get('frozen_sha256'):
        return m, 'NO_FROZEN_HASH'
    if 'frozen' not in str(m.get('lam_source')):
        return m, 'NOT_FROZEN_LAMBDA'
    if int(m.get('K', -1)) != 600:
        return m, 'WRONG_K'
    return m, 'FORMAL'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()

    cells, status = {}, {}
    for s in SUBSETS:
        for arm in ARMS:
            for sd in SEEDS:
                m, st = load_cell(a.root, arm, s, sd)
                cells[(arm, s, sd)] = m
                status[(arm, s, sd)] = st

    bad = {k: v for k, v in status.items() if v != 'FORMAL'}
    print('coverage: %d/%d formal' % (len(status) - len(bad), len(status)))
    if bad:
        print('not formal:')
        for (arm, s, sd), st in sorted(bad.items()):
            print('  %-4s %-6s s%-4d %s' % (arm, s, sd, st))
    print()

    hdr = '%-6s %-4s %6s %8s %8s %8s %8s %6s %6s' % (
        'subset', 'arm', 'n', 'main', 'sd', 'uncens', 'uncap', 'bnd', 'ep')
    print(hdr)
    print('-' * len(hdr))
    agg = {}
    for s in SUBSETS:
        for arm in ARMS:
            rows = [cells[(arm, s, sd)] for sd in SEEDS
                    if status[(arm, s, sd)] == 'FORMAL']
            if not rows:
                print('%-6s %-4s %6d %8s %8s %8s %8s %6s %6s'
                      % (s, arm, 0, '-', '-', '-', '-', '-', '-'))
                continue
            def col(k):
                v = [r.get(k) for r in rows if r.get(k) is not None]
                return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
            main_m, main_s = col('val_engine_mean_rmse')
            unc_m, _ = col('rmse_uncensored')
            uncap_m, _ = col('rmse_uncapped')
            bnd_m, _ = col('boundary_hit')
            ep = float(np.mean([r['best_ep'] for r in rows]))
            agg[(arm, s)] = dict(main=main_m, main_sd=main_s, n=len(rows),
                                 per_seed={sd: cells[(arm, s, sd)]['val_engine_mean_rmse']
                                           for sd in SEEDS
                                           if status[(arm, s, sd)] == 'FORMAL'})
            print('%-6s %-4s %6d %8.3f %8.3f %8.3f %8.3f %6.3f %6.1f'
                  % (s, arm, len(rows), main_m, main_s, unc_m, uncap_m, bnd_m, ep))
    print()

    print('paired chain (per seed, negative = the right-hand arm is better)')
    hdr2 = '%-6s %-7s %8s %8s %8s' % ('subset', 'pair', 'mean', 'sd', 'per-seed')
    print(hdr2)
    print('-' * len(hdr2))
    for s in SUBSETS:
        for name, hi, lo in CHAIN:
            d = []
            for sd in SEEDS:
                A, B = cells[(hi, s, sd)], cells[(lo, s, sd)]
                if (status[(hi, s, sd)] == 'FORMAL'
                        and status[(lo, s, sd)] == 'FORMAL'):
                    d.append(A['val_engine_mean_rmse'] - B['val_engine_mean_rmse'])
            if not d:
                print('%-6s %-7s %8s %8s %8s' % (s, name, '-', '-', '-'))
                continue
            print('%-6s %-7s %8.3f %8.3f %s'
                  % (s, name, float(np.mean(d)),
                     float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                     ' '.join('%.2f' % x for x in d)))
    print('\nreminder: 3 seeds give a spread, not a test; a difference smaller than '
          'the per-subset sd is not evidence of an effect.')

    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump(dict(coverage={k: v for k, v in status.items()},
                           aggregate={('%s|%s' % k): v for k, v in agg.items()}),
                      f, indent=1)
        print('record -> %s' % a.json_out)


if __name__ == '__main__':
    main()
