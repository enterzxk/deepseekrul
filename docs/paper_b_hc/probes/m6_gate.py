"""M6 (EFR) gate: build, step-0 identity with B3, and the correction is live.

  1. build_d('M6') works and shares B3's weights bit for bit
  2. step-0 prediction equals B3 exactly (amplitude head is zero-init)
  3. perturbing the amplitude head moves the prediction (not dead code)
  4. the rate multiplier stays inside [exp(-a), exp(a)]
"""
import json

import torch

from paper_b.hc import train_d as T

SH = '/home/user/sprint24/experiments/hc/d/FD001_shared.json'
INIT = '/home/user/sprint24/experiments/hc/tmp/wave1_init_A1_s101.pt'
DATA = '/home/user/rulb/data/CMAPSS'

sh = json.load(open(SH))
dev = torch.device('cpu')
kw = dict(subset='FD001', K=int(sh['K']), l_mean=float(sh['l_mean']),
          head_seed=int(sh['head_seed']), dev=dev, init_from=INIT, seed=101)

b3, _ = T.build_d('B3', **kw)
m6, _ = T.build_d('M6', **kw)

sb, sm = b3.state_dict(), m6.state_dict()
common = [k for k in sb if k in sm]
extra = [k for k in sm if k not in sb]
dmax = max(float((sb[k].float() - sm[k].float()).abs().max()) for k in common)
print('GATE1 common=%d extra=%s max|dB3-dM6|=%.3e' % (len(common), extra[:4], dmax))
assert dmax == 0.0, 'M6 must not change any B3 weight at init'
w = m6.efr.amp[2].weight.detach()
print('GATE1b amp.out zero-init max|w|=%.3e  efr params=%d'
      % (float(w.abs().max()), sum(p.numel() for p in m6.efr.parameters())))
assert float(w.abs().max()) == 0.0

ebt = T.engine_batch('FD001', DATA, T.C.SPAN, T.C.VAL_CUTS, split='train')
n = int(ebt['lengths'][0])
with torch.no_grad():
    b3.eval(); m6.eval()
    f3 = T.forward_engine(b3, 'B3', ebt, 0, n, dev, int(sh['K']))
    f6 = T.forward_engine(m6, 'M6', ebt, 0, n, dev, int(sh['K']))
    pd = float((f3['pred'] - f6['pred']).abs().max())
print('GATE2 step-0 max|pred_B3 - pred_M6| = %.3e' % pd)
assert pd == 0.0

# 3. make the correction non-zero and check it moves the prediction
with torch.no_grad():
    m6.efr.amp[2].bias.fill_(0.05)
    f6b = T.forward_engine(m6, 'M6', ebt, 0, n, dev, int(sh['K']))
    moved = float((f6b['pred'] - f3['pred']).abs().max())
    q = f6b['q']
    nu_age = m6.head.future_rates(f6b['D'], f6b['nu'], f6b['z'], int(sh['K']))
    mult = m6.efr.rates(nu_age, f6b['D'], f6b['nu'], f6b['z'], q, int(sh['K'])) / nu_age
print('GATE3 moved=%.4f  multiplier range [%.4f, %.4f]  (clip a=1 -> [0.368, 2.718])'
      % (moved, float(mult.min()), float(mult.max())))
assert moved > 0.0
assert float(mult.min()) >= 0.3679 - 1e-6 and float(mult.max()) <= 2.7183 + 1e-6

# 4. argparse accepts the arm
import argparse
print('GATE4 arm in ALL_ARMS:', 'M6' in T.ALL_ARMS, '| in HI/ARRIVAL:',
      'M6' in T.HI_ARMS, 'M6' in T.ARRIVAL_ARMS)
print('M6 GATE OK')
