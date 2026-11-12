"""Is the calibration a pure function of (init, subset, encoding)?

cal_audit.py found the pinned gain differs across seeds by 1e-11 (A2cal) to
1e-8 (A3bcal). That is numerically irrelevant but conceptually important: the
gain is supposed to depend on nothing but the shared init, the subset and the
encoding, because the seed spread is read as "training randomness only".

Hypothesis: train_hc calibrates before any ``.eval()``, so dropout is active on
CUDA and consumes the CUDA generator, whose state is seeded per run. The CPU-RNG
guard in calibrate_init_rms cannot see that -- it only snapshots the CPU
generator -- so the guard passes while the measurement is not actually seed-free.

This measures the gain for two seeds in train mode and in eval mode.

Usage: python docs/paper_b_hc/probes/cal_determinism.py
"""
from pathlib import Path

import torch

from paper_b.hc import data_hc
from paper_b.hc.scale_align import (build_hc_scaled, calibrate_init_rms,
                                    g2_config, load_shared_init)
from paper_b.hc.train_hc import set_seed

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
TMP = ROOT / 'experiments/hc/tmp'
DATA = str(ROOT / 'data/CMAPSS')


def run(seed, mode):
    cfg = g2_config('A2cal', 'FD001', seed, rho=1.0)
    cfg['data_dir'] = DATA
    set_seed(seed)                       # exactly what train_hc does before building
    net = build_hc_scaled(cfg).to(torch.device('cuda:0'))
    if mode == 'eval':
        net.eval()
    load_shared_init(net, TMP / 'wave1_init_A2_s101.pt')
    ds = data_hc.build('FD001', DATA, cfg['span'], cfg['val_cuts'],
                       cycle_source=cfg['cycle_source'])
    rec = calibrate_init_rms(net, ds, cfg, rho=1.0, dev='cuda:0')
    return rec['gain'], rec['rms_h'], rec['rms_term_before']


def main():
    print('%-6s %8s %18s %18s %14s' % ('mode', 'seed', 'gain', 'rms_h', 'rms_term'))
    out = {}
    for mode in ('train', 'eval'):
        for seed in (101, 202):
            g, h, t = run(seed, mode)
            out[(mode, seed)] = g
            print('%-6s %8d %18.10f %18.8f %14.8f' % (mode, seed, g, h, t))
    for mode in ('train', 'eval'):
        d = abs(out[(mode, 101)] - out[(mode, 202)])
        rel = d / out[(mode, 101)]
        print('%-6s seed spread: abs %.3e  rel %.3e  -> %s'
              % (mode, d, rel, 'BITWISE SEED-FREE' if d == 0.0 else 'depends on the seed'))


if __name__ == '__main__':
    main()
