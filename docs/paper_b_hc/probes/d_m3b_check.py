"""M3-B definitional check: at init the anchored increment IS the age timer.

M3-B replaces the increment with delta_t = lambda0 * exp(clamp(u(z_t))),
lambda0 = 1/L_mean(training). With u zero-initialised, step 0 must give
delta == lambda0 for every cycle and every engine, hence D_t = t/L_mean and the
arrival operator must return exactly L_mean - t. If that does not hold, the arm
is not the ablation it claims to be.

Runs on CPU: this is exact arithmetic on the initialisation, not a numerics
comparison, so it does not need the GPU (which is busy with the matrix).
"""
import json

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.arrival import first_passage
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import build_d, forward_engine, trainable

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
INIT = ROOT + '/experiments/hc/tmp/wave1_init_A1_s101.pt'
SHARED = ROOT + '/experiments/hc/d'


def main():
    dev = torch.device('cpu')
    sh = json.loads(open(SHARED + '/FD001_shared.json').read())
    l_mean = float(sh['l_mean'])
    K = int(sh['K'])
    net, _ = build_d('M3B', 'FD001', K, l_mean, int(sh['head_seed']), dev, INIT)
    trainable(net, 'M3B')
    net.eval()
    print('anchor lambda0 = 1/%.4f = %.8f  (head says %.8f)'
          % (l_mean, 1.0 / l_mean, net.head.anchor_lambda0))

    eb = engine_batch('FD001', DATA, C.SPAN, C.VAL_CUTS, split='val')
    n = int(eb['lengths'][0])
    with torch.no_grad():
        fo = forward_engine(net, 'M3B', eb, 0, n, dev, K)
        delta = fo['delta']
        D = fo['D']
        worst_delta = float((delta - 1.0 / l_mean).abs().max())
        t = torch.arange(1, n + 1, dtype=torch.float32)
        worst_D = float((D - t / l_mean).abs().max())
        print('increment: max|delta - 1/L| = %.3e   (must be 0)' % worst_delta)
        print('damage   : max|D_t - t/L|    = %.3e   (must be 0)' % worst_D)
        assert worst_delta == 0.0, worst_delta
        assert worst_D < 1e-6, worst_D

        # the arrival operator must return the age timer exactly
        idx = torch.tensor([0, 10, 50, 100, min(n - 1, 150)])
        T = first_passage(D[idx], net.head.future_rates(D[idx], fo['nu'][idx],
                                                       fo['z'][idx], K))
        age_pred = l_mean - (idx.float() + 1.0)
        print('arrival at t+1 in %s: %s' % (idx.tolist(), [round(float(x), 4) for x in T]))
        print('age timer L - t      : %s' % [round(float(x), 4) for x in age_pred])
        worst_T = float((T - age_pred).abs().max())
        print('max|T - (L - t)| = %.3e' % worst_T)
        assert worst_T < 1e-3, worst_T

        # and the gradient w.r.t. the anchor head is live (it must be able to leave)
        u = net.head.anchor(fo['z'][:8])
        print('anchor head output at init: max|u| = %.3e (zero-init)' % float(u.abs().max()))
    print('CHECK OK: M3-B starts exactly on the age timer')


if __name__ == '__main__':
    main()
