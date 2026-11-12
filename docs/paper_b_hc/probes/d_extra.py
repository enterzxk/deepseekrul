"""Extra fields for the acceptance record: prior ratio, caps, S_0, gradient tensors."""
import glob
import json
import os

D = 'experiments/hc/tmp/d_acceptance'
keep = ['b0fix2_K600_b3_FD001', 'b0fix2_K600_b3_FD002',
        'b0fix2_K600_b3_FD003', 'b0fix2_K600_b3_FD004',
        'b0fix2_K600_b2_FD001', 'nob0v2_K600_b3_nob0_FD001']
for tag in keep:
    p = os.path.join(D, tag + '.json')
    with open(p, encoding='utf-8') as f:
        r = json.load(f)
    print('== %s' % tag)
    b0 = r['head']['b0']
    print('  b0=%s  Lmean=%.4f  train_n=%d  softplus(b0)=%s'
          % ('none' if b0 is None else '%.6f' % b0, r['train_lifetime']['mean'],
             r['train_lifetime']['n'],
             'n/a' if b0 is None else '%.6f' % r['head']['softplus_b0']))
    print('  damage_over_age_prior=%s  endpoint_frac_ge1=%.4f  arrival_frac_at_cap=%s'
          % (r['damage_over_age_prior'], r['endpoint_damage']['frac_ge1'],
             r['arrival_uncensored']['frac_at_cap']))
    print('  future_rate ratio_to_observed=%.4f  T: %s'
          % (r['future_rate']['ratio_to_observed'],
             json.dumps(r['arrival_uncensored'])))
    print('  S0 raw=%.4e used=%.4e floor_hit=%s  n_terms=%d'
          % (r['s0']['raw'], r['s0']['used'], r['s0']['floor_hit'], r['s0']['n_terms']))
    print('  loss_train_form=%.4e  rul_diag=%s'
          % (r['loss_train_form'], json.dumps(r['rul_diag_uncensored'])))
    print('  grads=%s' % json.dumps(r['grads']))
    print('  engines_used=%d/%d' % (r['engines_used'], r['engines_total']))
