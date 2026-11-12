"""Prove the loader merge changed nothing, and smoke a calibrated arm.

Two checks, both required before the v11 lock:

  A. EQUIVALENCE. The inline remap that lived in train_hc.py at 95a5898 is
     reproduced verbatim below as `old_inline_load`, and compared against
     `scale_align.load_shared_init` on all five canonical inits. Both loaders
     must produce bit-identical state dicts for every non-cal arm; the only
     intended difference is that the new one tolerates a missing `cycle_gain`.

  B. SMOKE. A calibrated arm must survive build -> load init -> calibrate ->
     one forward, without training. This is the check that was missing when
     A2cal died at ep=0 on `missing=['cycle_gain']`.

Reads only; writes nothing. Usage:
    python docs/paper_b_hc/probes/init_loader_equiv.py
"""
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import data_hc
from paper_b.hc.config_hc import hc_config
from paper_b.hc.model_hc import build_hc
from paper_b.hc.scale_align import (build_hc_scaled, calibrate_init_rms,
                                    g2_config, load_shared_init)

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
TMP = ROOT / 'experiments/hc/tmp'
DATA = str(ROOT / 'data/CMAPSS')

INITS = [('A1', 'wave1_init_A1_s101.pt'),
         ('A2', 'wave1_init_A2_s101.pt'),
         ('A4', 'wave1_init_A4_s101.pt'),
         ('A5', 'wave1_init_A5_s101.pt'),
         ('A3p', 'init_v9_A3p_s101.pt')]


def old_inline_load(net, path):
    """The block that lived in train_hc.py at 95a5898, logic verbatim."""
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    sd = dict(ckpt)
    want = net.state_dict()
    if 'pe_proj.weight' in want and 'pe_proj.weight' not in sd:
        sd = {('pe_proj.' + k.split('.', 1)[1]) if k.startswith('cycle_enc.proj.') else k: v
              for k, v in sd.items()}
    if 'cycle_enc.proj.weight' in want and 'cycle_enc.proj.weight' not in sd:
        sd = {('cycle_enc.proj.' + k.split('.', 1)[1]) if k.startswith('pe_proj.') else k: v
              for k, v in sd.items()}
    if 'cycle_enc.periods' in want and 'cycle_enc.periods' not in sd and 'pe_periods' in sd:
        sd['cycle_enc.periods'] = sd['pe_periods']
    if 'pe_periods' in want and 'pe_periods' not in sd and 'cycle_enc.periods' in sd:
        sd['pe_periods'] = sd['cycle_enc.periods']
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError('old loader mismatch %s %s' % (missing, unexpected))
    return net


def equivalence():
    print('A. loader equivalence (old inline vs scale_align.load_shared_init)')
    all_ok = True
    for name, fname in INITS:
        path = TMP / fname
        if not path.exists():
            print('  %-4s MISSING init %s' % (name, path))
            all_ok = False
            continue
        cfg = hc_config(name, 'FD001', 101)
        a = old_inline_load(build_hc(cfg), path)
        b = build_hc(cfg)
        load_shared_init(b, path)          # returns the record, not the net
        sa, sb = a.state_dict(), b.state_dict()
        if set(sa) != set(sb):
            print('  %-4s KEY SETS DIFFER' % name)
            all_ok = False
            continue
        worst = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
        same = all(torch.equal(sa[k], sb[k]) for k in sa)
        print('  %-4s %-28s tensors=%3d  bitwise=%s  max|diff|=%.1e'
              % (name, fname, len(sa), same, worst))
        all_ok &= same
    return all_ok


def cal_smoke():
    print('\nB. calibrated-arm smoke (build -> load init -> calibrate -> forward)')
    dev = torch.device('cuda:0')
    cfg = g2_config('A2cal', 'FD001', 101, rho=1.0)
    cfg['data_dir'] = DATA
    net = build_hc_scaled(cfg).to(dev).eval()
    info = load_shared_init(net, TMP / 'wave1_init_A2_s101.pt')
    ds = data_hc.build('FD001', DATA, cfg['span'], cfg['val_cuts'],
                       cycle_source=cfg['cycle_source'])
    rec = calibrate_init_rms(net, ds, cfg, rho=1.0, dev=dev)
    tr = ds['val']
    with torch.no_grad():
        p = net(torch.from_numpy(np.asarray(tr['x'][:8], np.float32)).to(dev),
                mask=torch.from_numpy(np.asarray(tr['mask'][:8], np.float32)).to(dev),
                baseline=torch.from_numpy(np.asarray(tr['baseline'][:8], np.float32)).to(dev),
                cycle=torch.from_numpy(np.asarray(tr['cycle_hc'][:8], np.float32)).to(dev),
                cond=torch.from_numpy(np.asarray(tr['cond'][:8], np.int64)).to(dev),
                mode='dual')['prediction']
    finite = bool(torch.isfinite(p).all())
    print('  init keys_loaded=%d missing=%d unexpected=%d'
          % (info['keys_loaded'], info['missing'], info['unexpected']))
    print('  gain=%.7f  ratio_before=%.6f  ratio_after=%.6f  finite=%s'
          % (rec['gain'], rec['rms_ratio_before'], rec['rms_ratio_after'], finite))
    ok = (finite and abs(rec['rms_ratio_after'] - 1.0) < 1e-3
          and abs(rec['gain'] - 0.0142921) < 1e-6)
    print('  expected gain 0.0142921 -> %s' % ('OK' if ok else 'MISMATCH'))
    return ok


if __name__ == '__main__':
    a = equivalence()
    b = cal_smoke()
    print('\nVERDICT: %s' % ('PASS' if (a and b) else 'FAIL'))
    raise SystemExit(0 if (a and b) else 1)
