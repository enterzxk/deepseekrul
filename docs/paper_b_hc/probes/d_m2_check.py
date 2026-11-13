"""M2 definitional check: at step 0 the CDE arm IS B0, bit for bit.

M2 adds a condition-matched evidence correction to the B0 backbone. The
correction head is zero-initialised, so the M2 prediction at step 0 must equal
B0's exactly. If it does not, the B0->M2 difference is confounded with a
different random head and the ablation says nothing.

CPU only: this is an equality check on the initialisation, not a numerics
comparison, and the GPUs are busy with the matrix.
"""
import json

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.train_d import build_d, forward_engine, trainable

ROOT = '/media/sangfor/vdb/user/zxk/deepseek-rul'
DATA = ROOT + '/data/CMAPSS'
INIT = ROOT + '/experiments/hc/tmp/wave1_init_A1_s101.pt'
SHARED = ROOT + '/experiments/hc/d'


def main():
    dev = torch.device('cpu')
    sh = json.loads(open(SHARED + '/FD001_shared.json').read())
    K = int(sh['K'])
    eb = engine_batch('FD001', DATA, C.SPAN, C.VAL_CUTS, split='val')
    i, n = 0, int(eb['lengths'][0])

    net_b0, _ = build_d('B0', 'FD001', K, None, int(sh['head_seed']), dev, INIT)
    net_m2, _ = build_d('M2', 'FD001', K, None, int(sh['head_seed']), dev, INIT)
    trainable(net_b0, 'B0')
    trainable(net_m2, 'M2')
    net_b0.eval()
    net_m2.eval()

    with torch.no_grad():
        fo_b0 = forward_engine(net_b0, 'B0', eb, i, n, dev, K)
        fo_m2 = forward_engine(net_m2, 'M2', eb, i, n, dev, K)
        worst = float((fo_m2['pred'] - fo_b0['pred']).abs().max())
        print('step-0 M2 vs B0: max|diff| = %.3e   (must be 0)' % worst)
        assert worst == 0.0, worst
        print('cde head output at init: max|corr| = %.3e'
              % float(net_m2.cde_head(fo_m2['q']).abs().max()))
        print('cde evidence q: shape %s  mean %.4f  std %.4f'
              % (tuple(fo_m2['q'].shape), float(fo_m2['q'].mean()),
                 float(fo_m2['q'].std())))

    n_b0 = sum(p.numel() for p in net_b0.parameters() if p.requires_grad)
    n_m2 = sum(p.numel() for p in net_m2.parameters() if p.requires_grad)
    n_cde = sum(p.numel() for p in list(net_m2.cde.parameters())
                + list(net_m2.cde_head.parameters()))
    print('trainable params: B0 %d  M2 %d  (CDE adds %d, %.1f%%)'
          % (n_b0, n_m2, n_cde, 100.0 * n_cde / max(n_b0, 1)))
    print('CHECK OK: M2 == B0 at step 0')


if __name__ == '__main__':
    main()
