"""G2 scale alignment: pin the cycle term to a fixed fraction of the backbone.

The measurement in ``docs/paper_b_hc/probes/g2_scale.py`` showed the two G2 arms
are not operating on the same scale at all:

    A2  (scalar)   ||cycle term|| / ||backbone h|| = 41-51x   <- term dominates
    A3a (Fourier)  ||cycle term|| / ||backbone h|| = 0.40x    <- term is a nudge

So A2 is closer to "a cycle regressor with a small sensor correction" than to
"sensor model + age prior", and the G2 comparison confounds encoding form with
magnitude. This module makes the magnitude a controlled constant.

Design (fixed, not learned, not adaptive):

    gain = rho * mean||h_base|| / mean||term||      measured once, on TRAINING
                                                    data, after init, no grad
    h_out = h_base + gain * cycle_enc(cycle)

`rho` is the target ratio: after pinning, ||term|| / ||h_base|| == rho by
construction. The gain is a registered buffer, not a parameter -- it cannot move
during training, and it is recorded in the manifest so a run can be reproduced.

Why not normalise per batch: a running statistic changes the objective mid-run
and makes the arms incomparable across epochs (the same argument as D's S_0).

Why a new module: ``model_hc.py``, ``config_hc.py`` and ``data_hc.py`` are all
under the v9 code lock. Nothing here edits them; ``DSPT_HC_Scaled`` subclasses
``DSPT_HC`` and only changes how the cycle term is added.

Arm table: only TWO new arms are needed. ``A2-raw`` and ``A3-raw`` already exist
as the Gate A runs (re-running a completed configuration is forbidden), so they
are referenced by run_id, not retrained.
"""
import json
from pathlib import Path

import numpy as np
import torch

from ..models.dspt import DSPT
from .config_hc import hc_config
from .data_hc import build
from .model_hc import DSPT_HC, build_hc

RHO_DEFAULT = 1.0

#: name -> (base variant, scale mode). 'raw' arms are existing Gate A runs.
G2_ARMS = {
    'A2raw': dict(base='A2', scale='raw'),
    'A2norm': dict(base='A2', scale='rms'),
    'A3raw': dict(base='A3a', scale='raw'),
    'A3norm': dict(base='A3a', scale='rms'),
}


class DSPT_HC_Scaled(DSPT_HC):
    """DSPT_HC with the cycle term multiplied by a fixed, pinned gain."""

    def __init__(self, cycle_gain=1.0, **kw):
        super().__init__(**kw)
        self.register_buffer('cycle_gain', torch.tensor(float(cycle_gain)))

    def encode(self, x, baseline, mask, cycle, cond=None):
        # Call the frozen backbone directly so the gain applies to the term only.
        # DSPT_HC is always built with use_cycle=False (see its __init__), so the
        # backbone ignores `cycle`; assert that instead of assuming it.
        assert getattr(self, 'use_cycle', False) is False, \
            'DSPT_HC_Scaled expects use_cycle=False; the backbone would add its own cycle term'
        h = DSPT.encode(self, x, baseline, mask, None, cond)
        if self.cycle_enc is None:
            return h
        return h + self.cycle_gain * self.cycle_enc(cycle).unsqueeze(1)


def _common(cfg):
    """The exact backbone kwargs ``build_hc`` uses.

    Duplicated on purpose: ``build_hc`` lives in the code-locked ``model_hc.py``
    and its mapping is not exported. ``assert_same_architecture`` below fails
    loudly if the two ever drift, so the copy cannot rot silently.
    """
    bb = cfg['backbone']
    return dict(num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
                dropout=cfg['dropout'], max_span=cfg['span'],
                ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
                hi_source=bb['hi_source'], hi_mono=bb['hi_mono'],
                use_pad_mask=True, n_cond=cfg['n_cond'])


def build_hc_scaled(cfg):
    """build_hc, but through the scaled subclass when the arm asks for it."""
    if cfg.get('scale_mode', 'raw') == 'raw' or cfg.get('variant') == 'A0':
        return build_hc(cfg)
    if cfg.get('frozen_injection'):
        raise ValueError('frozen injection (A3p) has no cycle_enc; '
                         'scale alignment does not apply to it')
    return DSPT_HC_Scaled(cycle_mode=cfg['cycle_mode'],
                          period_bank=cfg.get('period_bank', 'frozen'),
                          cycle_gain=1.0, **_common(cfg))


def assert_same_architecture(cfg):
    """The scaled model must differ from the frozen one only by the gain buffer."""
    a = build_hc(cfg)
    b = build_hc_scaled(cfg)
    sa = {k: tuple(v.shape) for k, v in a.state_dict().items()}
    sb = {k: tuple(v.shape) for k, v in b.state_dict().items() if k != 'cycle_gain'}
    assert sa == sb, ('scaled builder drifted from build_hc; '
                      'only-in-frozen=%s only-in-scaled=%s'
                      % (sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))))
    return True


def g2_config(arm, subset, seed, rho=RHO_DEFAULT):
    """Effective config for one G2 arm. Shape-compatible with hc_config."""
    if arm not in G2_ARMS:
        raise KeyError('unknown G2 arm %r; known: %s' % (arm, sorted(G2_ARMS)))
    spec = G2_ARMS[arm]
    cfg = hc_config(spec['base'], subset, seed)
    cfg = dict(cfg)
    cfg['variant'] = arm
    cfg['base_variant'] = spec['base']
    cfg['scale_mode'] = spec['scale']
    cfg['rho'] = float(rho)
    cfg['cycle_gain'] = 1.0
    return cfg


def measure_ratio(net, ds, n=512, dev='cuda:0'):
    """mean||term|| / mean||backbone h|| on the first n training windows."""
    dev = torch.device(dev)
    tr = ds['train']
    n = min(n, len(tr['y']))
    x = torch.from_numpy(np.asarray(tr['x'][:n], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(tr['mask'][:n], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(tr['baseline'][:n], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(tr['cond'][:n], np.int64)).to(dev)
    cy = torch.from_numpy(np.asarray(tr['cycle_hc'][:n], np.float32)).to(dev)
    with torch.no_grad():
        h = DSPT.encode(net, x, b, m, None, cd)
        term = net.cycle_enc(cy)
        # include the pinned gain: this is the term as the model actually adds it
        gain = getattr(net, 'cycle_gain', None)
        if gain is not None:
            term = term * gain
        n_h = float(h.norm(dim=-1).mean())
        n_t = float(term.norm(dim=-1).mean())
    return n_t / n_h if n_h else float('nan'), n_h, n_t, n


def pin_gain(net, ds, rho=RHO_DEFAULT, n=512, dev='cuda:0'):
    """Measure on training data and freeze the gain. Returns the record."""
    ratio, n_h, n_t, used = measure_ratio(net, ds, n=n, dev=dev)
    gain = rho / ratio if ratio else 1.0
    with torch.no_grad():
        net.cycle_gain.fill_(float(gain))
    rec = dict(rho=float(rho), gain=float(gain), ratio_before=float(ratio),
               h_norm=n_h, term_norm=n_t, n_windows=int(used),
               measured_on='train', when='after_init_before_training',
               source='paper_b/hc/scale_align.py::pin_gain')
    return rec


def main():
    """Verification only -- no training is launched here."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='A2norm')
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--rho', type=float, default=RHO_DEFAULT)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--data-dir',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/data/CMAPSS')
    ap.add_argument('--check-arch', action='store_true',
                    help='only verify the scaled builder matches build_hc')
    a = ap.parse_args()

    if a.check_arch:
        for arm in ('A2norm', 'A3norm'):
            cfg = g2_config(arm, a.subset, a.seed, rho=a.rho)
            assert_same_architecture(cfg)
            print('arch OK: %s matches build_hc (only the gain buffer differs)' % arm)
        return

    cfg = g2_config(a.arm, a.subset, a.seed, rho=a.rho)
    cfg['data_dir'] = a.data_dir
    net = build_hc_scaled(cfg).to(torch.device(a.device)).eval()
    ds = build(a.subset, cfg['data_dir'], cfg['span'], cfg['val_cuts'],
               cycle_source=cfg['cycle_source'])
    before, _, _, _ = measure_ratio(net, ds, dev=a.device)
    rec = pin_gain(net, ds, rho=a.rho, dev=a.device)
    after, _, _, _ = measure_ratio(net, ds, dev=a.device)
    rec['ratio_after'] = float(after)
    print(json.dumps(rec, indent=1))
    assert abs(after - a.rho) <= 1e-3 * max(1.0, a.rho), \
        'pinned ratio %.6f != rho %.6f' % (after, a.rho)
    print('OK: ||term||/||h|| %.4f -> %.4f (rho %.4f), gain %.6g'
          % (before, after, a.rho, rec['gain']))


if __name__ == '__main__':
    main()
