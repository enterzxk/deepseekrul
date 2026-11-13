"""M5 gate: build, step-0 identity with B0, and a non-zero DPC loss.

Three things must hold before M5 takes a GPU slot:
  1. build_d('M5') returns a net whose shared weights match B0 bit for bit
  2. step-0 RUL prediction equals B0 exactly (the head never touches the path)
  3. the DPC term is finite and non-zero, and the target is aligned at t+h
"""
import json

import torch

from paper_b.hc import train_d as T
from paper_b.hc.dpc import check_alignment

SH = '/home/user/sprint24/experiments/hc/d/FD001_shared.json'
INIT = '/home/user/sprint24/experiments/hc/tmp/wave1_init_A1_s101.pt'
DATA = '/home/user/rulb/data/CMAPSS'

sh = json.load(open(SH))
dev = torch.device('cpu')
kw = dict(subset='FD001', K=int(sh['K']), l_mean=float(sh['l_mean']),
          head_seed=int(sh['head_seed']), dev=dev, init_from=INIT, seed=101)

b0, _ = T.build_d('B0', **kw)
m5, _ = T.build_d('M5', **kw)

# 1. shared weights identical
sb, sm = b0.state_dict(), m5.state_dict()
common = [k for k in sb if k in sm]
extra = [k for k in sm if k not in sb]
# The damage head ('head.*') is dead for both arms (NOHEAD_ARMS excludes it from
# the optimiser and nothing reads it), and it is seeded differently for B0 vs M5
# by design. Compare only the parameters that actually run.
used = [k for k in common if not k.startswith('head.')]
dmax = max(float((sb[k].float() - sm[k].float()).abs().max()) for k in used)
dhead = max([float((sb[k].float() - sm[k].float()).abs().max())
             for k in common if k.startswith('head.')] or [0.0])
print('GATE1 used keys=%d extra=%s max|d|=%.3e (dead head max|d|=%.3e)'
      % (len(used), extra[:4], dmax, dhead))
assert dmax == 0.0

# head must be zero-init so it cannot disturb anything at step 0
w = m5.dpc.out.weight.detach()
print('GATE1b dpc.out zero-init: max|w|=%.3e  params=%d'
      % (float(w.abs().max()), sum(p.numel() for p in m5.dpc.parameters())))
assert float(w.abs().max()) == 0.0

# 2. step-0 prediction identity on a real engine
ebt = T.engine_batch('FD001', DATA, T.C.SPAN, T.C.VAL_CUTS, split='train')
with torch.no_grad():
    b0.eval(); m5.eval()
    n = int(ebt['lengths'][0])
    f0 = T.forward_engine(b0, 'B0', ebt, 0, n, dev, int(sh['K']))
    f5 = T.forward_engine(m5, 'M5', ebt, 0, n, dev, int(sh['K']))
    pd = float((f0['pred'] - f5['pred']).abs().max())
print('GATE2 step-0 max|pred_B0 - pred_M5| = %.3e' % pd)
assert pd == 0.0

# 3. DPC loss finite, non-zero, and alignment correct
check_alignment(m5.dpc)
loss, used = m5.dpc.loss(f5['z'])
print('GATE3 dpc loss=%.6f spans_used=%d z=%s' % (float(loss), used, tuple(f5['z'].shape)))
assert used == 3 and float(loss) > 0.0

acc = T.term_sums(m5, 'M5', f5, 125.0)
L = T.aggregate(acc, 1.0)
print('GATE3b terms=%s  dpc=%.6f' % (sorted(L), float(L['dpc'])))
print('M5 GATE OK')
