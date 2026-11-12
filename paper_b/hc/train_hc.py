"""Stage C trainer: the frozen training loop, with a swappable cycle path.

The loop below is a transcription of ``paper_b.training.train.train_one`` (same
optimiser, schedule, loss terms, gradient clip, early-stopping rule and
checkpoint choice) because the frozen module may not be edited and the Stage C
comparison is only meaningful if everything except the cycle path is identical.
``paper_b/hc/verify_loop.py`` checks the transcription numerically against the
frozen trainer on synthetic data.

Audit fixes in this file (see CODE_REVIEW_20261111_test_lead.md):
  A1  model selection uses the engine-level mean RMSE, and the output carries
      ``val_engine_mean_rmse`` (main) + ``val_pooled_rmse`` (secondary). The
      misleading ``val_engine_rmse`` name is not used at all here.
  B3  epochs/patience/batch/lr/weight_decay are read from the effective config
      and asserted equal to the protocol constants, so config.json cannot lie.
      A deliberate smaller budget (smoke / verify_loop) sets
      ``cfg['budget_override']`` and the reason is recorded in config.json.
  C1  the cycle intervention perturbs the RAW cycle and re-maps through
      ``cycle_vector``; adding delta to an already-clipped scalar produced
      inputs (e.g. 260) that the model never saw during training.
  C3  the lifetime breakdown averages per-engine RMSE with equal weight and
      splits engines with ``np.array_split`` so 20 engines give 7/7/6.
  C4  the cycle key lookup no longer evaluates ``d['cycle']`` eagerly.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..training.metrics import per_engine_rmse, rmse, summarize
from . import data_hc
from .config_hc import BATCH, EPOCHS, LR, PATIENCE, WEIGHT_DECAY
from .model_hc import build_hc


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _now():
    import datetime
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def dump_init(cfg, log=print):
    """Export the initial weights of this configuration and stop.

    Deliberately separate from training (2026-11-11 ruling): the dump run must
    not double as a training run, otherwise the ledger cannot tell which record
    produced the artefact and which one consumed it. The payload carries the
    provenance needed to tie it to one code baseline: variant / subset / seed /
    span / created_utc / code_md5 / git_sha / baseline_version.
    """
    set_seed(cfg['seed'])
    net = build_hc(cfg)
    p = Path(cfg['dump_init'])
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit(f'REFUSED: init dump {p} already exists; a dump is written once')
    prov = cfg.get('provenance') or {}
    payload = dict(
        variant=cfg['variant'], subset=cfg['subset'], seed=cfg['seed'],
        span=cfg['span'], created_utc=_now(),
        code_md5=cfg.get('code_md5') or prov.get('code_md5'),
        git_sha=cfg.get('git_sha') or prov.get('git_sha'),
        baseline_version=cfg.get('baseline_version') or prov.get('baseline_version'),
        state_dict={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        note='initial weights only; never a trained checkpoint')
    torch.save(payload, p)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    log('  dump_init=%s variant=%s seed=%s sha256=%s' % (p, cfg['variant'], cfg['seed'], sha))
    return p, sha


def _t(d, dev, key):
    return torch.from_numpy(np.ascontiguousarray(d[key])).to(dev)


def _cycle_of(d):
    """C4: no eager evaluation of the fallback key."""
    return np.asarray(d['cycle_hc'] if 'cycle_hc' in d else d['cycle'], np.float32)


def predict_hc(net, d, dev, bs=512):
    net.eval()
    n = len(d['y'])
    out = []
    cyc = _cycle_of(d)
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                o = net(torch.from_numpy(d['x'][sl]).to(dev),
                        mask=torch.from_numpy(d['mask'][sl]).to(dev),
                        baseline=torch.from_numpy(d['baseline'][sl]).to(dev),
                        cycle=(torch.from_numpy(cyc[sl]).to(dev)
                               if getattr(net, 'needs_cycle', True) else None),
                        cond=torch.from_numpy(d['cond'][sl]).to(dev), mode='dual')
            out.append(o['prediction'].reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def cycle_intervention(net, d, dev, cycle_source, delta=10.0, bs=512):
    """Hold sensor evidence fixed, advance the age, measure the RUL response.

    The delta is applied in the RAW cycle space and then pushed back through
    ``cycle_vector`` so the perturbed input is on the same scale and clipping
    regime the model was trained on (audit C1). Returns the mean signed delta
    (negative is physically correct), the mean absolute delta, and the fraction
    of windows moving in the correct direction. For a sensor-only model this is
    exactly zero, which is the point of reporting it.
    """
    raw = np.asarray(d['cycle_raw'], np.float32)
    d2 = dict(d)
    d2['cycle_hc'] = data_hc.cycle_vector(raw + float(delta), cycle_source)
    p0 = predict_hc(net, d, dev, bs)
    p1 = predict_hc(net, d2, dev, bs)
    diff = p1 - p0
    return dict(delta=float(delta), cycle_source=cycle_source,
                mean_delta=float(diff.mean()),
                mean_abs_delta=float(np.abs(diff).mean()),
                frac_negative=float((diff < 0).mean()),
                frac_zero=float((np.abs(diff) < 1e-6).mean()))


def _breakdowns(pv, y, unit, cut):
    out = {}
    bins = [(0, 25), (25, 50), (50, 100), (100, 1000)]
    rb = {}
    for lo, hi in bins:
        sel = (y >= lo) & (y < hi)
        if sel.any():
            rb[f'{lo}-{hi}'] = dict(n=int(sel.sum()), rmse=rmse(pv[sel], y[sel]),
                                    bias=float(np.mean(pv[sel] - y[sel])))
    out['rul_bins'] = rb

    life = {}
    for u in np.unique(unit):
        sel = unit == u
        life[int(u)] = dict(n=int(sel.sum()), rmse=rmse(pv[sel], y[sel]),
                            cut_max=int(cut[sel].max()))
    if life:
        # C3: equal-weight mean of per-engine RMSE (same aggregation as the main
        # metric), and a split that actually covers every engine.
        order = sorted(life, key=lambda k: life[k]['cut_max'])
        groups = dict(zip(('short', 'mid', 'long'), np.array_split(np.array(order), 3)))
        out['lifetime_tertiles'] = {
            g: dict(n_engines=len(ids),
                    rmse=float(np.mean([life[int(i)]['rmse'] for i in ids])))
            for g, ids in groups.items() if len(ids)}
    out['per_engine'] = {str(k): v for k, v in life.items()}
    return out


def train_hc(cfg, device='cuda:0', out_dir=None, log=print):
    # B3: the effective config is the single source of truth; the protocol
    # constants are only a drift alarm. A run that deliberately uses a smaller
    # budget (smoke, verify_loop) must say so through ``budget_override`` so the
    # exemption is visible in config.json instead of being an invisible hack.
    epochs, patience = int(cfg['epochs']), int(cfg['patience'])
    batch, lr, wd = int(cfg['batch']), float(cfg['lr']), float(cfg['weight_decay'])
    if not cfg.get('budget_override'):
        for name, got, want in (('epochs', epochs, EPOCHS), ('patience', patience, PATIENCE),
                                ('batch', batch, BATCH), ('lr', lr, LR),
                                ('weight_decay', wd, WEIGHT_DECAY)):
            if got != want:
                raise AssertionError(f'cfg {name}={got} != protocol {want}')

    data_dir = cfg['data_dir']
    set_seed(cfg['seed'])
    t0 = time.time()

    ds = data_hc.build(cfg['subset'], data_dir, cfg['span'], cfg['val_cuts'],
                       cycle_source=cfg['cycle_source'])
    tr, va = ds['train'], ds['val']
    dev = torch.device(device)
    X, Y, MK, BS, CD = (_t(tr, dev, k) for k in ('x', 'y', 'mask', 'baseline', 'cond'))
    CY = torch.from_numpy(_cycle_of(tr)).to(dev)
    HT = torch.from_numpy(tr['hi']).to(dev)

    torch.backends.cudnn.benchmark = True
    # Calibrated arms must go through DSPT_HC_Scaled, otherwise `cycle_gain`
    # does not exist and the calibration below would fail (loudly, not silently).
    if cfg.get('scale_mode') == 'cal':
        from .scale_align import build_hc_scaled
        net = build_hc_scaled(cfg).to(dev)
    else:
        net = build_hc(cfg).to(dev)
    if cfg.get('init_from'):
        # Shared-initialisation control (A3a vs A3p). The two variants name the
        # same Linear(8->d) and the same period buffer differently -- ``pe_proj``
        # / ``pe_periods`` in the frozen model, ``cycle_enc.proj`` /
        # ``cycle_enc.periods`` in the subclass -- so the remap must be driven by
        # what THIS model has, not by what the dump happens to contain. A plain
        # strict=False load would silently leave the cycle projection at its own
        # random draw; anything left over raises.
        # Single source of truth: scale_align.load_shared_init owns the remap and
        # is the only place that knows ``cycle_gain`` cannot come from an init
        # file. The copy that used to live here had drifted from it -- it did not
        # exclude cycle_gain -- so every calibrated arm died at ep=0 with
        # ``init_from mismatch: missing=['cycle_gain']``.
        from .scale_align import load_shared_init
        info = load_shared_init(net, cfg['init_from'])
        log('  init_from=%s loaded=%d missing=0 unexpected=0 (remapped)'
            % (info['init_from'], info['keys_loaded']))
    # G2 init-RMS calibration. Order matters and is asserted by the caller:
    # the shared init is loaded above, the optimiser is created below, so the
    # gain is a property of the init and never of an optimisation step. The
    # record is written to calibration.json further down, where `out` exists.
    calib = None
    if cfg.get('scale_mode') == 'cal':
        from .scale_align import calibrate_init_rms
        calib = calibrate_init_rms(net, ds, cfg, rho=float(cfg.get('rho', 1.0)),
                                   dev=dev)
        log('  calibration: rms_ratio %.6f -> %.6f rho=%.3g gain=%.6g'
            % (calib['rms_ratio_before'], calib['rms_ratio_after'],
               calib['rho'], calib['gain']))
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    g = torch.Generator(device=dev).manual_seed(cfg['seed'])
    bb = cfg['backbone']

    def val_metrics():
        """(pooled, engine_mean, preds). A1: selection uses engine_mean."""
        p = predict_hc(net, va, dev)
        return rmse(p, va['y']), per_engine_rmse(va['unit'], va['y'], p), p

    best, best_state, best_ep, history = float('inf'), None, -1, []
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(X), generator=g, device=dev)
        tot, nb = 0.0, 0
        for idx in perm.split(batch):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = net(X[idx], mask=MK[idx], baseline=BS[idx],
                          cycle=(CY[idx] if net.needs_cycle else None),
                          cond=CD[idx], mode='dual')
                y = Y[idx].reshape(-1, 1)
                e = out['prediction'] - y
                loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()
                if getattr(net, 'dual_span', False) and cfg['aux_dual']:
                    eA = out['y_A'] - y
                    eB = out['y_B'] - y
                    loss = loss + 0.25 * (eA.square().mean() + eB.square().mean())
                if bb['hi_source'] != 'none' and bb['hi_lambda'] > 0:
                    ht = HT[idx].reshape(-1)
                    loss = loss + bb['hi_lambda'] * torch.mean((out['hi'].reshape(-1) - ht) ** 2)
                    if bb['hi_mono'] != 'none' and bb['hi_mono_coef'] > 0:
                        hs = out['hi_seq']
                        step = (MK[idx][:, 1:] * MK[idx][:, :-1])
                        diff = hs[:, 1:] - hs[:, :-1]
                        pen = torch.relu(diff) if bb['hi_mono'] == 'correct' else torch.relu(-diff)
                        loss = loss + bb['hi_mono_coef'] * (pen * step).sum() / step.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        pooled, eng, _ = val_metrics()
        # 郑总 P2-3: one field vocabulary across both trainers. The frozen
        # trainer's ``val_rmse`` key is the engine-level mean; here the names say
        # what they are, so a merged curve cannot silently pick the wrong column.
        history.append(dict(epoch=ep, train_loss=tot / max(nb, 1),
                            val_engine_mean_rmse=eng, val_pooled_rmse=pooled))
        if ep % 5 == 0 or ep == 1:
            log('  ep%03d loss=%.4f val_eng=%.4f pooled=%.4f best=%.4f@%d'
                % (ep, tot / max(nb, 1), eng, pooled, best, best_ep))
        if eng < best - 1e-4:
            best, best_ep = eng, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)

    pv = predict_hc(net, va, dev)
    pooled = rmse(pv, va['y'])
    eng = per_engine_rmse(va['unit'], va['y'], pv)
    m = summarize(pv, va['y'])
    m.update(val_engine_mean_rmse=eng, val_pooled_rmse=pooled,
             best_val_engine_mean_rmse=best, best_ep=best_ep,
             params=int(nparam), epochs_run=len(history),
             elapsed_s=round(time.time() - t0, 1),
             select_on_val=True)
    if eng > pooled + 1e-9:
        raise AssertionError(f'Jensen violated: engine_mean {eng} > pooled {pooled}')
    per_cut = {}
    for c in np.unique(va['cut']):
        sel = va['cut'] == c
        per_cut[int(c)] = dict(n=int(sel.sum()), rmse=rmse(pv[sel], va['y'][sel]))
    m['per_cut'] = per_cut
    m.update(_breakdowns(pv, va['y'], va['unit'], va['cut']))
    m['cycle_intervention'] = cycle_intervention(net, va, dev, cfg['cycle_source'])
    m['cycle_stats'] = dict(
        train_min=float(_cycle_of(tr).min()), train_max=float(_cycle_of(tr).max()),
        val_min=float(_cycle_of(va).min()), val_max=float(_cycle_of(va).max()),
        n_train_above_clip=int((np.asarray(tr['cycle_raw']) > 250).sum()),
        n_val_above_clip=int((np.asarray(va['cycle_raw']) > 250).sum()))

    out = Path(out_dir)
    (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    if calib is not None:
        (out / 'calibration.json').write_text(json.dumps(calib, indent=1))
    torch.save(net.state_dict(), out / 'ckpt' / 'best.pt')
    (out / 'metrics.json').write_text(json.dumps(m, indent=1))
    (out / 'history.json').write_text(json.dumps(history, indent=0))
    import pandas as pd
    pd.DataFrame(dict(engine_id=va['unit'], cut=va['cut'], cycle=va['cycle_raw'],
                      y_true=va['y'], y_pred=pv, split='val')).to_csv(
        out / 'preds.csv', index=False)
    log('DONE %s %s s%d val_eng=%.4f pooled=%.4f ep=%d params=%d %.1fs'
        % (cfg['variant'], cfg['subset'], cfg['seed'], eng, pooled,
           best_ep, nparam, m['elapsed_s']))
    return m, ds
