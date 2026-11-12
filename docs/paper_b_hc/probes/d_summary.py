"""Compact table over the d_acceptance JSON records, plus the B2/B3 fairness check.

Reads experiments/hc/tmp/d_acceptance/*.json and prints one row per record. The
one assertion it makes is the ablation-fairness one: B2 and B3 must carry a
byte-identical head init (same fingerprint), otherwise the fixed-vs-variable
rate difference is confounded with a different random init.
"""
import glob
import json
import os
import sys

D = 'experiments/hc/tmp/d_acceptance'

rows = []
for p in sorted(glob.glob(os.path.join(D, '*.json'))):
    with open(p, encoding='utf-8') as f:
        r = json.load(f)
    tag = os.path.basename(p)[:-5]
    rows.append((tag, r))

hdr = ('%-26s %-6s %4s %-10s %-17s %9s %7s %7s %7s %6s %7s %8s %8s %10s %10s %5s %5s'
       % ('record', 'subset', 'K', 'mode', 'fingerprint', 'd0_med', 'Dend_med',
          'alive>=1', 'miss', 'nu*K', 'nuhat', 'T_med', 'S0', 'gD', 'gR',
          'GP', 'ALL'))
print(hdr)
print('-' * len(hdr))
for tag, r in rows:
    h = r['head']
    print('%-26s %-6s %4d %-10s %-17s %9.6f %7.3f %7.4f %7.4f %6.3f %7.6f %8.1f %8.1e %10.2e %10.2e %5s %5s'
          % (tag, r['subset'], r['K'], 'B2' if r['fixed_rate'] else 'B3',
             h['fingerprint'], r['first_cycle_increment']['median'],
             r['endpoint_damage']['median'], r['frac_alive_with_D_ge1'],
             r['frac_uncapped_not_hit'], r['rate_times_K'],
             r['future_rate']['median_across_engines'],
             r['arrival_uncensored']['median'], r['s0']['raw'],
             r['grads']['damage_head'], r['grads']['rate_head'],
             r['verdict']['gradient_path'], r['verdict']['overall']))

print()
# fairness: same head init across rate modes, per subset
by = {}
for tag, r in rows:
    if 'nob0' in tag or 'K200' in tag:
        continue
    by.setdefault(r['subset'], {})['B2' if r['fixed_rate'] else 'B3'] = r
bad = 0
for subset, d in sorted(by.items()):
    if 'B2' in d and 'B3' in d:
        a, b = d['B2']['head']['fingerprint'], d['B3']['head']['fingerprint']
        ok = a == b
        bad += 0 if ok else 1
        print('head-init fairness %s: B2 %s vs B3 %s -> %s'
              % (subset, a, b, 'IDENTICAL' if ok else 'DIFFER'))
print('\nchecks per record')
for tag, r in rows:
    print('%-26s %-6s %s' % (tag, r['subset'],
                             json.dumps({k: v for k, v in r['checks'].items()
                                         if v is not None})))
sys.exit(1 if bad else 0)
