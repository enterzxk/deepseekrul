"""Why do the auxiliary-term gradients come out as exactly zero?

Reproduces the D4.5 measurement on one engine and prints, for each term, whether
the term requires grad and what the gradient norm is w.r.t. the damage head.
"""
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import (build_d, engine_tensors, forward_engine,
                                losses_for, trainable)

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
INIT = ROOT + '/experiments/hc/tmp/wave1_init_A1_s101.pt'

dev = torch.device('cuda:0')
net, _ = build_d('B3', 'FD001', 600, 209.525, 101, dev, INIT)
trainable(net, 'B3')
net.eval()
ebt = engine_batch('FD001', DATA, C.SPAN, C.VAL_CUTS, split='train')
n = int(ebt['lengths'][0])
fo = forward_engine(net, 'B3', ebt, 0, n, dev, 600)
print('delta.requires_grad', fo['delta'].requires_grad,
      'D.requires_grad', fo['D'].requires_grad,
      'pred.requires_grad', fo['pred'].requires_grad)
print('D[-1]', float(fo['D'][-1]), 'delta[0]', float(fo['delta'][0]))
total, L = losses_for(net, 'B3', fo, float(C.RUL_CAP), 4.5981552723381447e-10,
                      float(C.RUL_CAP) ** 2, float(C.RUL_CAP) ** 2)
params = [p for p in net.parameters() if p.requires_grad]
for k, v in L.items():
    print('%-9s requires_grad=%s value=%.6e' % (k, v.requires_grad, float(v)))
    gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
    nz = [g for g in gs if g is not None]
    print('           tensors_with_grad=%d/%d  norm=%.6e'
          % (len(nz), len(gs), float(sum(g.norm() ** 2 for g in nz)) ** 0.5))
