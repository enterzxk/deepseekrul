"""Stage D acceptance v2 (2026-11-13): bias init, reproducible head, S_0.

What changed from the 403db2b version, and why each change exists:

  * **Reproducible head.** The v1 probe loaded only the shared backbone and left
    ``head.*`` at whatever the global RNG happened to hold, so two runs of the
    same probe could disagree. The head is now built under its own seed and
    fingerprinted; every number below is reproducible from
    (backbone init, head seed, subset, K) alone.
  * **Bias initialization.** ``b0 = log(e^{1/L_mean} - 1)`` so that
    ``softplus(b0) = 1/L_mean``. ``L_mean`` is the mean run-to-failure length of
    the TRAINING engines of the frozen split -- never validation or test.
  * **S_0 calibration.** The smoothness normalization constant is measured once
    on the corrected init (advisor ruling: one measurement, shared by all arms)
    and reported together with whether it hit the 1e-12 floor.
  * **Per-subset reporting.** First-cycle increment, endpoint damage,
    premature-crossing ratio and miss ratio, all in ``eval()``. The three
    "quantity" numbers are never pooled across subsets.
  * **Loss uses the uncropped output.** The training-form loss is the censored
    hinge on the arrival operator's own output, with no extra clamp at
    ``rul_cap``. The clipped-label metric is a separate diagnostic.
  * **Separated verdicts.** Gradient-path pass is reported on its own line and
    never promoted to overall acceptance.

Modes: PHASE 1 (no_grad, eval) for the deterministic quantities, PHASE 2
(grad enabled, eval) for the backward. ``eval()`` does not disable autograd, so
phase 2 is deliberately not wrapped in ``no_grad``/``inference_mode``.

Usage:
  python docs/paper_b_hc/probes/d_acceptance.py --subsets FD001,FD002,FD003,FD004
  python docs/paper_b_hc/probes/d_acceptance.py --fixed-rate --no-b0 --K 200
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.arrival import first_passage
from paper_b.hc.data_d import engine_batch, train_lifetime_mean
from paper_b.hc.hi_d import DSPT_HC_Structured, mean_life_bias

ROOT = Path('/media/sangfor/vdb/user/zxk/deepseek-rul')
DATA = str(ROOT / 'data/CMAPSS')
TMP = ROOT / 'experiments/hc/tmp'
OUT = TMP / 'd_acceptance'


def build_model(dev, fixed_rate=False, l_mean=None, head_seed=101):
    cfg = dict(C.hc_config('A1', 'FD001', 101))
    bb = cfg['backbone']
    net = DSPT_HC_Structured(
        fixed_rate=fixed_rate, K=cfg.get('K', 200), cycle_mode='none',
        num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
        dropout=cfg['dropout'], max_span=cfg['span'],
        ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
        hi_source=bb['hi_source'], hi_mono=bb['hi_mono'],
        use_pad_mask=True, n_cond=cfg['n_cond'],
        l_mean=l_mean, head_seed=head_seed)
    ckpt = torch.load(TMP / 'wave1_init_A1_s101.pt', map_location='cpu',
                      weights_only=True)
    sd = dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [m for m in missing if not m.startswith('head.')]
    if bad or unexpected:
        raise RuntimeError('init mismatch: missing=%s unexpected=%s' % (bad, unexpected))
    # The backbone is frozen by design, so only the head may receive gradient.
    # This makes the gradient check stricter: anything reaching head.* must have
    # travelled through the head's own path.
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith('head.'))
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


def damage_path(net, z):
    """(n, d) -> delta (n,), D (n,), nu (n,) along one engine's trajectory."""
    delta = net.head.increment(z.unsqueeze(0)).reshape(-1)
    D = torch.cumsum(delta, dim=0)
    nu = net.current_rate(delta.unsqueeze(0)).reshape(-1)
    return delta, D, nu


def calibrate_s0(net, ebt, dev, n_batches=8, bs=8, seed=0, floor=1e-12):
    """S_0 = E[(delta_{t+1} - delta_t)^2] on the corrected init, detached.

    B_0 is 8 fixed-seed batches of 8 TRAINING engines, full trajectories: the
    smoothness term is a per-trajectory second difference, so the reference
    sample has to be trajectories, not windows. Measured once, never per epoch
    (the spec forbids re-measuring: that would make the objective drift).
    """
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    E = len(ebt['lengths'])
    vals = []
    with torch.no_grad():
        for _ in range(n_batches):
            idx = torch.randint(0, E, (bs,), generator=g).tolist()
            for i in idx:
                n = int(ebt['lengths'][i])
                z = z_for_engine(net, ebt, i, n, dev)
                d = net.head.increment(z.unsqueeze(0)).reshape(-1)
                if d.numel() < 2:
                    continue
                vals.append(float(((d[1:] - d[:-1]) ** 2).mean()))
    raw = float(np.mean(vals))
    return dict(raw=raw, used=max(raw, floor), floor=floor,
                floor_hit=bool(raw < floor), n_batches=n_batches, batch=bs,
                seed=int(seed), n_terms=len(vals))


def run_subset(subset, a, dev):
    # The lifetime prior is always measured -- even under --no-b0, where it is
    # used only as the reference for damage_over_age_prior. Otherwise the
    # pre-fix record would have no scale metric to compare against.
    lrec = train_lifetime_mean(subset, DATA)
    lmean = lrec['mean']
    net, new_keys = build_model(dev, fixed_rate=a.fixed_rate,
                               l_mean=None if a.no_b0 else lmean,
                               head_seed=a.head_seed)
    net.K = int(a.K)
    init = net.head.init_report()
    init['head_seed'] = a.head_seed
    init['n_new_tensors'] = len(new_keys)
    init['b0_formula'] = None if a.no_b0 else 'log(expm1(1/L_mean))'
    init['rate_out_zero_init'] = not a.no_b0

    eb = engine_batch(subset, DATA, C.SPAN, C.VAL_CUTS, split='val')
    n_eng = len(eb['lengths'])
    use = n_eng if a.engines <= 0 else min(a.engines, n_eng)

    x = torch.from_numpy(np.asarray(eb['x'][0, :8], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][0, :8], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][0, :8], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][0, :8], np.int64)).to(dev)
    parity = net.assert_pooling_parity(x, m, b, torch.zeros(8, device=dev), cd)

    # ---- PHASE 1: deterministic quantities, eval(), no_grad ----------------
    n_alive = n_ge1 = n_unc = n_not_hit = 0
    d0_vals, dend_vals, dsum = [], [], 0.0
    prior_sum = 0.0
    fr_vals, t_unc = [], []
    per_engine = []
    with torch.no_grad():
        for i in range(use):
            n = int(eb['lengths'][i])
            z = z_for_engine(net, eb, i, n, dev)
            delta, D, nu = damage_path(net, z)
            r = torch.from_numpy(np.asarray(eb['r'][i, :n], np.float32)).to(dev)
            age = torch.from_numpy(np.asarray(eb['cycle_raw'][i, :n], np.float32)).to(dev)
            d0_vals.append(float(delta[0]))
            dend_vals.append(float(D[-1]))
            idx = torch.arange(0, n, max(1, n // 40), device=dev)
            nu_hat = net.head.future_rates(D[idx], nu[idx], z[idx], net.K)
            T = first_passage(D[idx], nu_hat)
            capped = r[idx] >= C.RUL_CAP
            alive = r[idx] > 0
            n_alive += int(alive.sum())
            n_ge1 += int((D[idx][alive] >= 1.0).sum())
            n_unc += int((~capped).sum())
            n_not_hit += int((T[~capped] >= net.K - 1e-6).sum())
            if (~capped).any():
                t_unc.append(T[~capped])
            fr_vals.append(float(nu_hat.median()))
            if lmean:
                dsum += float(D[idx][alive].sum())
                prior_sum += float((age[idx][alive] / lmean).sum())
            per_engine.append(dict(engine=int(eb['engine_id'][i]), n=n,
                                   delta0=float(delta[0]), D_end=float(D[-1]),
                                   mean_rate=float(delta.mean()),
                                   median_future_rate=float(nu_hat.median()),
                                   rate_times_K=float(delta.mean()) * net.K))

    # ---- S_0 on the corrected init, shared by every arm --------------------
    ebt = engine_batch(subset, DATA, C.SPAN, C.VAL_CUTS, split='train')
    s0 = calibrate_s0(net, ebt, dev, seed=a.s0_seed)

    # ---- PHASE 2: gradient path, eval(), autograd ON ----------------------
    losses, maes_raw, maes_clip = [], [], []
    for i in range(min(a.grad_engines, use)):
        n = int(eb['lengths'][i])
        z = z_for_engine(net, eb, i, n, dev)
        _, D, nu = damage_path(net, z)
        r = torch.from_numpy(np.asarray(eb['r'][i, :n], np.float32)).to(dev)
        idx = torch.arange(0, n, max(1, n // 40), device=dev)
        T = net.arrival(D[idx], nu[idx], z[idx])
        r_i = r[idx]
        capped = r_i >= C.RUL_CAP
        # Training form: censored hinge on the operator's own output, no extra
        # clamp at rul_cap (a cap here would silently zero the gradient of every
        # prediction above 125).
        losses.append(torch.where(capped,
                                  torch.clamp(C.RUL_CAP - T, min=0.0) ** 2,
                                  (T - r_i) ** 2).mean())
        unc = ~capped
        if unc.any():
            maes_raw.append(float((T[unc] - r_i[unc]).abs().mean()))
            # Evaluation form: the frozen protocol clips both sides at rul_cap
            # before scoring. Reported separately, never mixed into the loss.
            maes_clip.append(float((torch.clamp(T[unc], max=C.RUL_CAP)
                                    - torch.clamp(r_i[unc], max=C.RUL_CAP)).abs().mean()))
    total = torch.stack(losses).mean()
    total.backward()
    g_delta = float(sum(p.grad.norm() for p in net.head.delta.parameters()
                        if p.grad is not None))
    g_rate = float(sum(p.grad.norm() for p in net.head.rate.parameters()
                       if p.grad is not None))
    g_head = float(sum(p.grad.norm() for p in net.head.parameters()
                       if p.grad is not None))
    n_grad_tensors = int(sum(1 for p in net.head.parameters() if p.grad is not None))
    n_head_tensors = int(sum(1 for p in net.head.parameters()))

    mean_rate = float(np.mean([e['mean_rate'] for e in per_engine]))
    T_unc = torch.cat(t_unc) if t_unc else None
    rec = dict(
        subset=subset, mode='eval', K=net.K, engines_used=use, engines_total=n_eng,
        fixed_rate=a.fixed_rate, head=init, train_lifetime=lrec,
        pooling_parity=parity, loss_train_form=float(total),
        s0=s0,
        first_cycle_increment=dict(
            min=float(min(d0_vals)), median=float(np.median(d0_vals)),
            max=float(max(d0_vals)),
            expected_if_b0=1.0 / lmean),
        endpoint_damage=dict(
            min=float(min(dend_vals)), median=float(np.median(dend_vals)),
            max=float(max(dend_vals)),
            frac_ge1=float(np.mean([v >= 1.0 for v in dend_vals]))),
        damage_over_age_prior=(dsum / prior_sum) if prior_sum else None,
        frac_alive_with_D_ge1=(n_ge1 / n_alive if n_alive else None),
        frac_uncapped_not_hit=(n_not_hit / n_unc if n_unc else None),
        n_alive=n_alive, n_uncapped=n_unc,
        mean_rate=mean_rate,
        # The trap is about the rate the ARRIVAL OPERATOR integrates, i.e. the
        # future rate, not the observed one: nu_hat * K is the total damage the
        # horizon can accumulate from D=0. Both are recorded; the gate uses the
        # future rate because that is what decides whether a path can cross.
        rate_times_K=float(np.median(fr_vals)) * net.K,
        observed_rate_times_K=mean_rate * net.K,
        zero_gradient_trap=bool(float(np.median(fr_vals)) * net.K <= 1.0),
        future_rate=dict(
            median_across_engines=float(np.median(fr_vals)),
            min_across_engines=float(min(fr_vals)),
            max_across_engines=float(max(fr_vals)),
            ratio_to_observed=float(np.median(fr_vals)) / mean_rate,
            expected_if_b0=1.0 / lmean),
        arrival_uncensored=dict(
            min=float(T_unc.min()) if T_unc is not None else None,
            median=float(T_unc.median()) if T_unc is not None else None,
            max=float(T_unc.max()) if T_unc is not None else None,
            frac_at_cap=(float((T_unc >= net.K - 1e-6).float().mean())
                         if T_unc is not None else None)),
        rul_diag_uncensored=dict(
            mae_uncropped=float(np.mean(maes_raw)) if maes_raw else None,
            mae_clipped_labels=float(np.mean(maes_clip)) if maes_clip else None,
            note='init state, not a trained model'),
        grads=dict(damage_head=g_delta, rate_head=g_rate, head_total=g_head,
                   tensors_with_grad=n_grad_tensors,
                   tensors_total=n_head_tensors,
                   rate_out_layer_grad=bool(any(
                       p.grad is not None and float(p.grad.abs().sum()) > 0
                       for p in net.head.rate[2].parameters()))),
        per_engine=per_engine,
    )
    rec['checks'] = {
        'pooling_parity': parity < 1e-4,
        'gradient_path_damage': g_delta > 0.0,
        'gradient_path_rate': (g_rate > 0.0) if not a.fixed_rate else None,
        'no_zero_gradient_trap': not rec['zero_gradient_trap'],
        'scale_on_prior': (None if rec['damage_over_age_prior'] is None
                           else 0.5 < rec['damage_over_age_prior'] < 2.0),
        'miss_ratio_low': (None if rec['frac_uncapped_not_hit'] is None
                           else rec['frac_uncapped_not_hit'] < 0.5),
        's0_above_floor': not s0['floor_hit'],
    }
    rec['info'] = dict(
        all_head_tensors_have_grad=(n_grad_tensors == n_head_tensors),
        note='zero-init last layer: the rate head first layer gets its gradient '
             'from the second step on')
    gp = [v for k, v in rec['checks'].items() if k.startswith('gradient_path')]
    rec['verdict'] = dict(
        gradient_path='PASS' if all(v for v in gp if v is not None) else 'FAIL',
        overall='PASS' if all(v for v in rec['checks'].values()
                              if v is not None) else 'FAIL')
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subsets', default='FD001,FD002,FD003,FD004')
    ap.add_argument('--engines', type=int, default=0, help='0 = all val engines')
    ap.add_argument('--grad-engines', type=int, default=2)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fixed-rate', action='store_true')
    ap.add_argument('--no-b0', action='store_true',
                    help='pre-fix state: default PyTorch bias, still seeded')
    ap.add_argument('--head-seed', type=int, default=101)
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--s0-seed', type=int, default=0)
    ap.add_argument('--tag', default='b0fix')
    a = ap.parse_args()
    dev = torch.device(a.device)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for subset in [s.strip() for s in a.subsets.split(',') if s.strip()]:
        rec = run_subset(subset, a, dev)
        rows.append(rec)
        tag = a.tag + ('_b2' if a.fixed_rate else '_b3') + ('_nob0' if a.no_b0 else '')
        (OUT / ('%s_%s.json' % (tag, subset))).write_text(
            json.dumps(rec, indent=1), encoding='utf-8')
        print('%-6s K=%d %s  b0=%s  d0=%.6f  D_end_med=%.3f  alive_D>=1=%.4f  '
              'miss=%.4f  nu*K=%.3f  nuhat=%.6f  T_med=%.1f  S0=%.3e%s  '
              'gD=%.3e gR=%.3e  GP=%s  ALL=%s'
              % (subset, rec['K'], 'B2' if a.fixed_rate else 'B3',
                 ('%.6f' % rec['head']['b0']) if rec['head']['b0'] is not None else 'none',
                 rec['first_cycle_increment']['median'],
                 rec['endpoint_damage']['median'],
                 rec['frac_alive_with_D_ge1'] or 0.0,
                 rec['frac_uncapped_not_hit'] or 0.0,
                 rec['rate_times_K'],
                 rec['future_rate']['median_across_engines'],
                 rec['arrival_uncensored']['median'] or 0.0,
                 rec['s0']['raw'],
                 ' FLOOR' if rec['s0']['floor_hit'] else '',
                 rec['grads']['damage_head'], rec['grads']['rate_head'],
                 rec['verdict']['gradient_path'], rec['verdict']['overall']))
    print('\nrecord -> %s' % OUT)
    print(json.dumps([dict(subset=r['subset'], checks=r['checks'],
                           verdict=r['verdict']) for r in rows], indent=1))


if __name__ == '__main__':
    main()
