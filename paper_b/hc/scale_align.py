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
from pathlib import Path

import numpy as np
import torch

from ..models.dspt import DSPT
from .config_hc import hc_config
from .data_hc import build
from .model_hc import DSPT_HC, build_hc

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
RHO_DEFAULT = 1.0
BS_DEFAULT = 512

#: arm -> config_hc variant name. The '-raw' arms are existing Gate A runs and
#: are referenced by run_id, never retrained (re-running a completed
#: configuration is forbidden). Only the two '-cal' arms are new.
G2_ARMS = {
    'A2raw': dict(cfg='A2', mode='raw', label='scalar cycle, untruncated (existing run)'),
    'A3araw': dict(cfg='A3a', mode='raw', label='Fourier cycle, 250-clipped (existing run)'),
    'A2cal': dict(cfg='A2cal', mode='cal', label='scalar cycle, untruncated, init-RMS calibrated'),
    'A3bcal': dict(cfg='A3bcal', mode='cal', label='Fourier cycle, untruncated, init-RMS calibrated'),
}

#: shared backbone init each calibrated arm must start from (经理 ruling
#: 2026-11-12): same file as its base arm in wave 1, so the only difference
#: between arms is the calibrated gain.
DEFAULT_INIT = {
    'A2cal': 'experiments/hc/tmp/wave1_init_A2_s101.pt',
    'A3bcal': 'experiments/hc/tmp/init_v9_A3p_s101.pt',
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


def load_shared_init(net, path):
    """Load a shared backbone init with the same target-driven remap train_hc uses.

    ``pe_proj``/``pe_periods`` (frozen model) vs ``cycle_enc.proj``/``cycle_enc.periods``
    (subclass) means a plain strict=False load would silently leave the cycle
    projection at its own random draw. The remap is driven by what *this* model
    has. ``cycle_gain`` is the one key the init cannot carry: it is produced by
    the calibration that runs immediately after this load.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']   # --dump-init wraps it with provenance
    sd = dict(ckpt)
    want = net.state_dict()
    if 'pe_proj.weight' in want and 'pe_proj.weight' not in sd:
        sd = {('pe_proj.' + k.split('.', 1)[1]) if k.startswith('cycle_enc.proj.') else k: v
              for k, v in sd.items()}
    if 'cycle_enc.proj.weight' in want and 'cycle_enc.proj.weight' not in sd:
        sd = {('cycle_enc.proj.' + k.split('.', 1)[1]) if k.startswith('pe_proj.') else k: v
              for k, v in sd.items()}
    if 'cycle_enc.periods' in want and 'cycle_enc.periods' not in sd and 'pe_periods' in sd:
        sd['cycle_enc.periods'] = sd['pe_periods']   # copy, never pop
    if 'pe_periods' in want and 'pe_periods' not in sd and 'cycle_enc.periods' in sd:
        sd['pe_periods'] = sd['cycle_enc.periods']
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != 'cycle_gain']
    if missing or unexpected:
        raise RuntimeError('init_from mismatch: missing=%s unexpected=%s (after remap)'
                           % (sorted(missing)[:8], sorted(unexpected)[:8]))
    return dict(init_from=str(path), keys_loaded=len(sd), missing=0, unexpected=0)


def g2_config(arm, subset, seed, rho=RHO_DEFAULT):
    """Effective config for one G2 arm (config_hc carries the cal arms)."""
    if arm not in G2_ARMS:
        raise KeyError('unknown G2 arm %r; known: %s' % (arm, sorted(G2_ARMS)))
    cfg = dict(hc_config(G2_ARMS[arm]['cfg'], subset, seed))
    cfg['g2_arm'] = arm
    cfg['rho'] = float(rho)
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
                variant=cfg.get('variant'), g2_arm=cfg.get('g2_arm'),
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
    ap.add_argument('--init-from', default=None,
                    help='shared backbone init; defaults to DEFAULT_INIT[arm]')
    a = ap.parse_args()

    if a.check_arch:
        for arm in ('A2cal', 'A3bcal'):
            assert_same_architecture(g2_config(arm, a.subset, a.seed, rho=a.rho))
            print('arch OK: %s matches build_hc (only the gain buffer differs)' % arm)
        return

    cfg = g2_config(a.arm, a.subset, a.seed, rho=a.rho)
    cfg['data_dir'] = a.data_dir
    if a.arm not in DEFAULT_INIT and not a.init_from:
        raise SystemExit('arm %s has no default init; pass --init-from' % a.arm)
    init = a.init_from or str(ROOT / DEFAULT_INIT[a.arm])
    net = build_hc_scaled(cfg).to(torch.device(a.device)).eval()
    info = load_shared_init(net, init)          # init first ...
    ds = build(a.subset, cfg['data_dir'], cfg['span'], cfg['val_cuts'],
               cycle_source=cfg['cycle_source'])
    rec = calibrate_init_rms(net, ds, cfg, rho=a.rho, dev=a.device)   # ... then calibrate
    rec.update(info)
    print(json.dumps(rec, indent=1))
    assert abs(rec['rms_ratio_after'] - a.rho) <= 1e-3 * max(1.0, a.rho), \
        'calibrated RMS ratio %.6f != rho %.6f' % (rec['rms_ratio_after'], a.rho)
    print('OK: strict RMS ratio %.6f -> %.6f (rho %.4f), gain %.6g'
          % (rec['rms_ratio_before'], rec['rms_ratio_after'], a.rho, rec['gain']))


if __name__ == '__main__':
    main()
