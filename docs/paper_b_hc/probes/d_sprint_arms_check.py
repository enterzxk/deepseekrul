"""Definitional check for the sprint arms: at init, Baseline B IS the age prior.

Baseline B is "AgePrior(cycle) + SensorResidual(z_t)" with the residual head
initialised to output ~0. If the residual is not exactly zero at step 0, then
whatever the model does later is not attributable to the sensor evidence alone,
and the baseline is not the baseline. This checks the claim on the real code.

Also reports the prior's shape: A[t] = mean over engines alive at t of
min(n_e - t, cap), fitted on the training split only.
"""
import json

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import (ResidualHead, build_d, fit_age_prior,
                                forward_engine, trainable)

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
INIT = ROOT + '/experiments/hc/tmp/wave1_init_A1_s101.pt'
SHARED = ROOT + '/experiments/hc/d'


def main():
    dev = torch.device('cuda:0')
    sh = json.loads(open(SHARED + '/FD001_shared.json').read())
    cap = float(C.RUL_CAP)
    ebt = engine_batch('FD001', DATA, C.SPAN, C.VAL_CUTS, split='train')
    ebv = engine_batch('FD001', DATA, C.SPAN, C.VAL_CUTS, split='val')
    prior = fit_age_prior(ebt, cap)

    net, _ = build_d('AAR', 'FD001', int(sh['K']), float(sh['l_mean']),
                     int(sh['head_seed']), dev, INIT)
    net.register_buffer('age_prior', torch.from_numpy(prior).to(dev))
    net.residual = ResidualHead(net.d_model).to(dev)
    trainable(net, 'AAR')
    net.eval()

    i = 0
    n = int(ebv['lengths'][i])
    fo = forward_engine(net, 'AAR', ebv, i, n, dev, int(sh['K']))
    cyc = np.asarray(ebv['cycle_raw'][i, :n], np.float32)
    idx = torch.from_numpy(cyc).long().clamp(0, net.age_prior.numel() - 1).to(dev)
    prior_pred = net.age_prior[idx]
    worst_res = float((fo['pred'] - prior_pred).abs().max())
    print('residual at init: max|pred - prior| = %.3e  (must be 0)' % worst_res)
    assert worst_res == 0.0, worst_res

    # and the residual really is zero as a function, not just because the prior
    # happens to cancel it
    x = torch.from_numpy(np.asarray(ebv['x'][i, :8], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(ebv['mask'][i, :8], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(ebv['baseline'][i, :8], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(ebv['cond'][i, :8], np.int64)).to(dev)
    z = net.pooled_feature(x, m, b, torch.zeros(8, device=dev), cd)
    r = net.residual(z.float())
    print('residual head output: max|r| = %.3e  (must be 0)' % float(r.abs().max()))
    assert float(r.abs().max()) == 0.0

    d = np.diff(prior[1:])
    print('prior: A[1]=%.3f A[50]=%.3f A[200]=%.3f A[end]=%.3f  min=%.3f max=%.3f'
          % (prior[1], prior[50], prior[200], prior[-3], prior[1:].min(), prior[1:].max()))
    print('prior monotone non-increasing: %s  (max rise %.3f)'
          % (bool((d <= 1e-6).all()), float(d.max())))
    print('CHECK OK')


if __name__ == '__main__':
    main()
