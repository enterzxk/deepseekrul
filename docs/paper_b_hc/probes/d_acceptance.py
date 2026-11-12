"""Stage D acceptance: three quantities the reviewer asked for, on real data.

No training. Deterministic by construction: the whole run is in ``eval()``, so
dropout is off and every number below is reproducible from the seed of the
shared init alone. That is deliberate -- the parity check compares a replicated
pooling path against the frozen forward, and under ``train()`` the two draw
different dropout masks, which is exactly how incident 14 happened (a check that
passed while measuring nothing, or here: failed for the wrong reason).

Two phases:

  PHASE 1 (no_grad, eval)  -- pooling parity, then
      1. fraction of NOT-yet-failed samples whose damage already exceeds 1;
      2. fraction of UNCAPPED samples that never cross within K
         (zero-gradient samples: they carry no training signal).
  PHASE 2 (grad on, eval)  -- one backward, then
      3. gradient norm reaching the damage head and the future-rate head.

Usage: python docs/paper_b_hc/probes/d_acceptance.py [--subset FD001] [--engines 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.hi_d import DSPT_HC_Structured

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
DATA = str(ROOT / 'data/CMAPSS')
TMP = ROOT / 'experiments/hc/tmp'


def build_model(dev, fixed_rate=False):
    cfg = dict(C.hc_config('A1', 'FD001', 101))
    bb = cfg['backbone']
    net = DSPT_HC_Structured(
        fixed_rate=fixed_rate, K=200, cycle_mode='none',
        num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
        dropout=cfg['dropout'], max_span=cfg['span'],
        ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
        hi_source=bb['hi_source'], hi_mono=bb['hi_mono'],
        use_pad_mask=True, n_cond=cfg['n_cond'])
    ckpt = torch.load(TMP / 'wave1_init_A1_s101.pt', map_location='cpu',
                      weights_only=True)
    sd = dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [m for m in missing if not m.startswith('head.')]
    if bad or unexpected:
        raise RuntimeError('init mismatch: missing=%s unexpected=%s' % (bad, unexpected))
    # eval(): dropout off, so the parity check and both fractions are
    # deterministic and the two pooling paths see the same activations.
    return net.to(dev).eval(), sorted(missing)


def z_for_engine(net, eb, i, n, dev, chunk=128):
    outs = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        x = torch.from_numpy(np.asarray(eb['x'][i, lo:hi], np.float32)).to(dev)
        m = torch.from_numpy(np.asarray(eb['mask'][i, lo:hi], np.float32)).to(dev)
        b = torch.from_numpy(np.asarray(eb['baseline'][i, lo:hi], np.float32)).to(dev)
        cd = torch.from_numpy(np.asarray(eb['cond'][i, lo:hi], np.int64)).to(dev)
        cy = torch.zeros(x.shape[0], device=dev)
        outs.append(net.pooled_feature(x, m, b, cy, cd))
    return torch.cat(outs, 0)


def damage_and_arrival(net, z):
    """(n, d) -> D (n,), nu (n,), and the arrival operator's inputs."""
    delta = net.head.increment(z.unsqueeze(0)).reshape(-1)
    D = torch.cumsum(delta, dim=0)
    nu = net.current_rate(delta.unsqueeze(0)).reshape(-1)
    return delta, D, nu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--engines', type=int, default=4)
    ap.add_argument('--grad-engines', type=int, default=2)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fixed-rate', action='store_true')
    a = ap.parse_args()
    dev = torch.device(a.device)

    net, new_keys = build_model(dev, fixed_rate=a.fixed_rate)
    eb = engine_batch(a.subset, DATA, C.SPAN, C.VAL_CUTS, split='val')
    print('engines %d (using %d), mode=eval, new head tensors %d'
          % (len(eb['lengths']), a.engines, len(new_keys)))

    x = torch.from_numpy(np.asarray(eb['x'][0, :8], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][0, :8], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][0, :8], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][0, :8], np.int64)).to(dev)
    parity = net.assert_pooling_parity(x, m, b, torch.zeros(8, device=dev), cd)
    print('pooling parity (eval): max|reg_head(z) - forward| = %.3e' % parity)

    n_alive = n_ge1 = n_unc = n_not_hit = 0
    d0_vals = []
    with torch.no_grad():
        for i in range(a.engines):
            n = int(eb['lengths'][i])
            z = z_for_engine(net, eb, i, n, dev)
            delta, D, nu = damage_and_arrival(net, z)
            # damage at the FIRST observed cycle = the first increment, since
            # D_0 = 0 by construction. If this is ~0.69 (softplus(0)) the head
            # puts an engine past the threshold within two cycles.
            d0_vals.append(float(delta[0]))
            r = torch.from_numpy(np.asarray(eb['r'][i, :n], np.float32)).to(dev)
            idx = torch.arange(0, n, max(1, n // 40), device=dev)
            T = net.arrival(D[idx], nu[idx], z[idx])
            capped = r[idx] >= C.RUL_CAP
            alive = r[idx] > 0
            n_alive += int(alive.sum())
            n_ge1 += int((D[idx][alive] >= 1.0).sum())
            n_unc += int((~capped).sum())
            n_not_hit += int((T[~capped] >= net.K - 1e-6).sum())

    losses = []
    for i in range(min(a.grad_engines, a.engines)):
        n = int(eb['lengths'][i])
        z = z_for_engine(net, eb, i, n, dev)
        _, D, nu = damage_and_arrival(net, z)
        r = torch.from_numpy(np.asarray(eb['r'][i, :n], np.float32)).to(dev)
        idx = torch.arange(0, n, max(1, n // 40), device=dev)
        T = net.arrival(D[idx], nu[idx], z[idx])
        r_i = r[idx]
        capped = r_i >= C.RUL_CAP
        losses.append(torch.where(capped,
                                  torch.clamp(C.RUL_CAP - T, min=0.0) ** 2,
                                  (T - r_i) ** 2).mean())
    total = torch.stack(losses).mean()
    total.backward()
    g_delta = float(sum(p.grad.norm() for p in net.head.delta.parameters()
                        if p.grad is not None))
    g_rate = float(sum(p.grad.norm() for p in net.head.rate.parameters()
                       if p.grad is not None))

    rec = dict(subset=a.subset, engines=a.engines, mode='eval',
               fixed_rate=a.fixed_rate, loss=float(total),
               pooling_parity=parity,
               frac_alive_with_D_ge1=(n_ge1 / n_alive if n_alive else None),
               frac_uncapped_not_hit=(n_not_hit / n_unc if n_unc else None),
               n_alive=n_alive, n_uncapped=n_unc,
               grad_damage_head=g_delta, grad_rate_head=g_rate,
               damage_at_first_cycle=dict(
                   min=float(min(d0_vals)), median=float(np.median(d0_vals)),
                   max=float(max(d0_vals)),
                   frac_ge1=float(np.mean([v >= 1.0 for v in d0_vals]))))
    print(json.dumps(rec, indent=1))
    ok = (g_delta > 0) and (g_rate > 0) and parity < 1e-4
    print('\nACCEPTANCE %s (deterministic, eval mode)' % ('PASS' if ok else 'FAIL'))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
