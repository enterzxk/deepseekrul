"""M7 == B3 at initialisation (CPU only, no GPU, no training).

Two claims to check:
  1. build_d('M7') constructs the same network as build_d('B3') -- same keys,
     bit-identical tensors. M7 must not change initialisation.
  2. The only loss difference is the alive term, which is absent for B3 and
     present for M7.
"""
import json
import torch

from paper_b.hc import train_d as T

SH = '/home/user/sprint24/experiments/hc/d/FD001_shared.json'
INIT = '/home/user/sprint24/experiments/hc/tmp/wave1_init_A1_s101.pt'
sh = json.load(open(SH))

kw = dict(subset='FD001', K=int(sh['K']), l_mean=float(sh['l_mean']),
          head_seed=int(sh['head_seed']), dev=torch.device('cpu'),
          init_from=INIT, seed=101)
n3, k3 = T.build_d('B3', **kw)
n7, k7 = T.build_d('M7', **kw)

sd3, sd7 = n3.state_dict(), n7.state_dict()
assert set(sd3) == set(sd7), (sorted(set(sd3) ^ set(sd7))[:5])
maxdiff = max(float((sd3[k].float() - sd7[k].float()).abs().max()) for k in sd3)
nparam = sum(p.numel() for p in n3.parameters())
print('keys=%d params=%d new_keys_same=%s max|dB3-dM7|=%.3e'
      % (len(sd3), nparam, k3 == k7, maxdiff))
assert maxdiff == 0.0, 'M7 initialisation differs from B3'

fo = dict(pred=torch.tensor([10., 20., 30.]),
          r=torch.tensor([5., 3., 0.]),
          delta=torch.tensor([0.5, 0.6, 0.7]),
          D=torch.tensor([0.5, 1.1, 1.8]))
L3 = T.aggregate(T.term_sums(None, 'B3', fo, 125.0, scr=('B3' in T.SCR_ARMS)), 1.0)
L7 = T.aggregate(T.term_sums(None, 'M7', fo, 125.0, scr=('M7' in T.SCR_ARMS)), 1.0)
print("B3 terms:", sorted(L3), " M7 terms:", sorted(L7))
assert 'alive' not in L3 and 'alive' in L7
base = L3['rul'] + 125.0 ** 2 * L3['terminal'] + 125.0 ** 2 * L3['smooth']
full = L7['rul'] + 125.0 ** 2 * L7['terminal'] + 125.0 ** 2 * L7['smooth'] + 125.0 ** 2 * L7['alive']
print('alive=%.6f  loss delta=%.6f (expect 78.125)' % (float(L7['alive']), float(full - base)))
print('M7 ARM OK')
