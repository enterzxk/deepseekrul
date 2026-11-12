"""G2 scale alignment: init-RMS calibration of the cycle term.

Measured confound (``docs/paper_b_hc/probes/g2_scale.py``): at the trained
checkpoint the cycle term is 41-51x the backbone norm for A2 (scalar) and 0.40x
for A3a (Fourier). At init it is 67.9x and 0.46x. Either way the two G2 arms are
not on the same scale, so the encoding-form comparison is confounded.

Codex's ruling (2026-11-12) fixed three things this module must respect:

  1. A2 uses **untruncated** cycle and A3a uses **250-clipped** cycle, so the
     original comparison varies encoding form, scale AND available age
     information at once. The new Fourier arm is therefore ``A3b`` (unclipped),
     not A3a.
  2. Calibration must be a strict RMS over **all** training windows, streamed,
     not a mean-vector-norm over the first 512 rows.
  3. It runs after the shared init is loaded and before the optimiser starts, it
     must not consume training randomness, and it stores a fingerprint.

Calibration:

    g = rho * RMS(h_base) / RMS(cycle term)      RMS over all elements
    h_out = h_base + g * cycle_enc(c)

``g`` is a registered buffer, not a parameter: it cannot move during training.
``rho`` is the target ratio, identical for every calibrated arm.

Why not normalise per batch: a running statistic changes the objective mid-run
and makes arms incomparable across epochs (same argument as D's S_0).

Why a new module: ``model_hc.py``, ``config_hc.py`` and ``data_hc.py`` are all
under the v9 code lock. Nothing here edits them.

Arm table: the ``-raw`` arms are the existing Gate A runs and are referenced by
run_id, never retrained (re-running a completed configuration is forbidden).
Only ``A2-cal`` and ``A3b-cal`` are new.
"""
import hashlib
import json

import numpy as np
import torch

from ..models.dspt import DSPT
from .config_hc import hc_config
from .data_hc import build
from .model_hc import DSPT_HC, build_hc

RHO_DEFAULT = 1.0
BS_DEFAULT = 512

#: name -> (base variant, mode). 'raw' arms are existing Gate A runs.
G2_ARMS = {
    'A2raw': dict(base='A2', mode='raw', label='scalar cycle, untruncated (existing run)'),
    'A3araw': dict(base='A3a', mode='raw', label='Fourier cycle, 250-clipped (existing run)'),
    'A2cal': dict(base='A2', mode='cal', label='scalar cycle, untruncated, init-RMS calibrated'),
    'A3bcal': dict(base='A3b', mode='cal', label='Fourier cycle, untruncated, init-RMS calibrated'),
}


class DSPT_HC_Scaled(DSPT_HC):
    """DSPT_HC with the cycle term multiplied by a fixed, calibrated gain."""

    def __init__(self, cycle_gain=1.0, **kw):
        super().__init__(**kw)
        self.register_buffer('cycle_gain', torch.tensor(float(cycle_gain)))

    def encode(self, x, baseline, mask, cycle, cond=None):
        # Call the frozen backbone directly so the gain applies to the term only.
        assert getattr(self, 'use_cycle', False) is False, \
            'DSPT_HC_Scaled expects use_cycle=False; the backbone would add its own cycle term'
        h = DSPT.encode(self, x, baseline, mask, None, cond)
        if self.cycle_enc is None:
            return h
        return h + self.cycle_gain * self.cycle_enc(cycle).unsqueeze(1)


def _common(cfg):
    """The exact backbone kwargs ``build_hc`` uses.

    Duplicated on purpose: ``build_hc`` lives in the code-locked ``model_hc.py``
    and its mapping is not exported. ``assert_same_architecture`` fails loudly if
    the two ever drift, so the copy cannot rot silently.
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
                         'calibration does not apply to it')
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
    cfg = dict(hc_config(spec['base'], subset, seed))
    cfg['variant'] = arm
    cfg['base_variant'] = spec['base']
    cfg['scale_mode'] = spec['mode']
    cfg['rho'] = float(rho)
    cfg['cycle_gain'] = 1.0
    return cfg


def _fingerprint(ds, cfg):
    """Exact identity of the data and config the calibration saw."""
    cyc = np.asarray(ds['train']['cycle_hc'], np.float32)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(cyc).tobytes())
    h.update(('%s|%s|%s|%s|%d' % (cfg.get('subset'), cfg.get('seed'),
                                  cfg.get('variant'), cfg.get('cycle_source'),
                                  len(cyc))).encode())
    return h.hexdigest()


def rms_stats(net, ds, dev='cuda:0', bs=BS_DEFAULT):
    """Strict RMS of the backbone output and of the (gain-scaled) cycle term.

    Streams over **every** training window; no sampling, no RNG.
    """
    dev = torch.device(dev)
    tr = ds['train']
    n = len(tr['y'])
    gain = getattr(net, 'cycle_gain', None)
    g = float(gain) if gain is not None else 1.0
    sum_h2 = sum_t2 = 0.0
    cnt_h = cnt_t = 0
    vnorm_h = vnorm_t = 0.0
    nv = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, min(i + bs, n))
            x = torch.from_numpy(np.asarray(tr['x'][sl], np.float32)).to(dev)
            m = torch.from_numpy(np.asarray(tr['mask'][sl], np.float32)).to(dev)
            b = torch.from_numpy(np.asarray(tr['baseline'][sl], np.float32)).to(dev)
            cd = torch.from_numpy(np.asarray(tr['cond'][sl], np.int64)).to(dev)
            cy = torch.from_numpy(np.asarray(tr['cycle_hc'][sl], np.float32)).to(dev)
            h = DSPT.encode(net, x, b, m, None, cd)
            t = net.cycle_enc(cy) * g
            sum_h2 += float((h * h).sum())
            cnt_h += h.numel()
            sum_t2 += float((t * t).sum())
            cnt_t += t.numel()
            vnorm_h += float(h.norm(dim=-1).mean())
            vnorm_t += float(t.norm(dim=-1).mean())
            nv += 1
    rms_h = (sum_h2 / cnt_h) ** 0.5
    rms_t = (sum_t2 / cnt_t) ** 0.5
    return dict(rms_h=rms_h, rms_term=rms_t,
                rms_ratio=(rms_t / rms_h if rms_h else float('nan')),
                vecnorm_ratio=((vnorm_t / nv) / (vnorm_h / nv) if vnorm_h else float('nan')),
                n_windows=int(n), n_batches=int(nv))


def calibrate_init_rms(net, ds, cfg, rho=RHO_DEFAULT, dev='cuda:0', bs=BS_DEFAULT):
    """Measure on all training windows and freeze the gain. Returns the record.

    Must be called after the shared init is loaded and before the optimiser is
    created. Asserts that neither the torch nor the numpy RNG advanced.
    """
    tp = torch.get_rng_state()
    np_state = np.random.get_state()
    before = rms_stats(net, ds, dev=dev, bs=bs)
    gain = rho / before['rms_ratio'] if before['rms_ratio'] else 1.0
    with torch.no_grad():
        net.cycle_gain.fill_(float(gain))
    after = rms_stats(net, ds, dev=dev, bs=bs)
    assert torch.equal(tp, torch.get_rng_state()), 'calibration consumed torch RNG'
    ns = np.random.get_state()
    assert ns[0] == np_state[0] and np.array_equal(ns[1], np_state[1]), \
        'calibration consumed numpy RNG'
    return dict(rho=float(rho), gain=float(gain),
                rms_h=before['rms_h'], rms_term_before=before['rms_term'],
                rms_ratio_before=before['rms_ratio'],
                rms_ratio_after=after['rms_ratio'],
                vecnorm_ratio_before=before['vecnorm_ratio'],
                n_windows=before['n_windows'], n_batches=before['n_batches'],
                subset=cfg.get('subset'), seed=cfg.get('seed'),
                variant=cfg.get('variant'), base_variant=cfg.get('base_variant'),
                cycle_source=cfg.get('cycle_source'),
                measured_on='train', when='after_init_before_optimizer',
                definition='strict RMS over all elements and all training windows',
                fingerprint=_fingerprint(ds, cfg),
                source='paper_b/hc/scale_align.py::calibrate_init_rms')


def main():
    """Verification only -- no training is launched here."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='A2cal')
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
        for arm in ('A2cal', 'A3bcal'):
            assert_same_architecture(g2_config(arm, a.subset, a.seed, rho=a.rho))
            print('arch OK: %s matches build_hc (only the gain buffer differs)' % arm)
        return

    cfg = g2_config(a.arm, a.subset, a.seed, rho=a.rho)
    cfg['data_dir'] = a.data_dir
    net = build_hc_scaled(cfg).to(torch.device(a.device)).eval()
    ds = build(a.subset, cfg['data_dir'], cfg['span'], cfg['val_cuts'],
               cycle_source=cfg['cycle_source'])
    rec = calibrate_init_rms(net, ds, cfg, rho=a.rho, dev=a.device)
    print(json.dumps(rec, indent=1))
    assert abs(rec['rms_ratio_after'] - a.rho) <= 1e-3 * max(1.0, a.rho), \
        'calibrated RMS ratio %.6f != rho %.6f' % (rec['rms_ratio_after'], a.rho)
    print('OK: strict RMS ratio %.6f -> %.6f (rho %.4f), gain %.6g'
          % (rec['rms_ratio_before'], rec['rms_ratio_after'], a.rho, rec['gain']))


if __name__ == '__main__':
    main()
