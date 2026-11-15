"""Unified trainer for Paper B (DSPT and Vanilla baseline).

Protocol (frozen):
  split by engine id, 80/20, split_seed=7, fold=0
  model seeds 101/202/303 (stage 1) and 101..505 (stage 2)
  RUL cap 125, endpoint RUL=0 at the last training cycle
  early stopping on engine-equal truncated validation RMSE
    (cuts 0.4/0.55/0.7/0.85/0.95, from paper_b/configs/protocol.json)
  no EMA selection (single, predetermined checkpoint rule)

Sensor-augmentation study additions (all inert when cfg['augment'] is None):
  * optional SensorAugmentor wiring: whole-sensor mask and/or residual-scale
    noise applied to the batch before the forward pass;
  * the checkpoint rule reads the plan's engine-equal RMSE,
    sqrt(mean_u MSE_u), instead of the historical mean of per-engine roots;
    both are written to metrics.json;
  * last.pt with optimiser / scheduler / permutation generator / augmentor
    generator states so an epoch-boundary resume reproduces the run.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch

from .data import prepare_train
from .metrics import (engine_equal_mae, engine_equal_nasa_mean, engine_equal_rmse,
                      per_engine_rmse, rmse, summarize)
from ..models import build_dspt, build_vanilla

DEFAULTS = dict(
    model='dspt', subset='FD001', seed=101, span=70, epochs=40, patience=12,
    batch=512, lr=1e-3, weight_decay=1e-4, d_model=64, layers=2, dropout=0.2,
    mode='dual', ms_mode='full', sa_mode='off', hi_source='none', hi_mono='none',
    hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=True, fp32=False,
    use_pad_mask=True,
    input_mode='raw', aux_dual=True, instnorm=False, rul_cap=125.0,
    split_seed=7, fold=0, val_frac=0.2, n_cond=6, cuts=(0.4, 0.55, 0.7, 0.85, 0.95),
    legacy_rul=False, global_hi=False, legacy_ms_pad=False, legacy_ema=False,
    augment=None, selection_metric='engine_equal_rmse', resume=False,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build(cfg):
    if cfg['model'] == 'dspt':
        return build_dspt(num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
                          dropout=cfg['dropout'], max_span=cfg['span'],
                          instnorm=cfg['instnorm'], ms_mode=cfg['ms_mode'],
                          sa_mode=cfg['sa_mode'], hi_source=cfg['hi_source'],
                          hi_mono=cfg['hi_mono'], use_pad_mask=cfg['use_pad_mask'],
                          use_cycle=cfg['use_cycle'], n_cond=cfg['n_cond'],
                          legacy_ms_pad=cfg['legacy_ms_pad'], legacy_ema=cfg['legacy_ema'])
    return build_vanilla(num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
                         dropout=cfg['dropout'], max_span=cfg['span'],
                         input_mode=cfg['input_mode'], use_cycle=cfg['use_cycle'])


def _tensors(d, dev, keys):
    return [torch.from_numpy(np.ascontiguousarray(d[k])).to(dev) for k in keys]


def _amp(dev, fp32):
    """Autocast context, or a no-op when the run is pinned to FP32.

    BF16 is not reproducible across processes on this stack (measured: the same
    weights re-run over 12 runs drifted, up to 1.12 cycles at FP32). Any
    comparison that has to survive re-execution therefore runs FP32 with TF32
    and cudnn benchmark off.
    """
    if fp32:
        return contextlib.nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)


def predict(net, d, dev, mode='dual', bs=512, fp32=False, sensor_keep=None):
    net.eval()
    n = len(d['y'])
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            sl = slice(i, i + bs)
            with _amp(dev, fp32):
                o = net(torch.from_numpy(d['x'][sl]).to(dev),
                    mask=torch.from_numpy(d['mask'][sl]).to(dev),
                    baseline=torch.from_numpy(d['baseline'][sl]).to(dev),
                    cycle=(torch.from_numpy(d['cycle'][sl]).to(dev) if net.use_cycle else None),
                    cond=torch.from_numpy(d['cond'][sl]).to(dev), mode=mode,
                    sensor_keep=(torch.from_numpy(sensor_keep[sl]).to(dev)
                                 if sensor_keep is not None else None))
            out.append(o['prediction'].reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def _gpu_uuid(dev):
    try:
        return str(torch.cuda.get_device_properties(dev).uuid)
    except Exception:
        return ''


def _git_head(repo):
    try:
        import subprocess
        return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ''


def _augmentor(cfg, dev):
    from ..sensor_aug.augmentation import SensorAugmentor
    from ..sensor_aug.noise_stats import effective_scale
    a = cfg['augment']
    scale = None
    if float(a.get('noise_alpha', 0.0)) > 0:
        stats = json.loads(Path(a['noise_stats_path']).read_text())
        scale = np.asarray(effective_scale(stats, float(a['noise_alpha']))['s'], np.float32)
    return SensorAugmentor(mask_prob=float(a.get('mask_prob', 0.0)), noise_scale=scale,
                           n_sensors=14, seed=cfg['seed'], device=str(dev))


def train_one(cfg, device='cuda:0', out_root=None):
    cfg = {**DEFAULTS, **cfg}
    data_dir = Path(cfg['data_dir'])
    set_seed(cfg['seed'])
    t0 = time.time()
    dev = torch.device(device)
    out = Path(out_root or 'paper_b/results/raw') / cfg['tag'] / cfg['subset'] / f"s{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / 'status.json').write_text(json.dumps(
        {'status': 'RUNNING', 'pid': os.getpid(), 'started': time.time(),
         'device': str(dev), 'visible': os.environ.get('CUDA_VISIBLE_DEVICES', '')}, indent=1))

    ds = prepare_train(cfg['subset'], data_dir, span=cfg['span'],
                       split_seed=cfg['split_seed'], fold=cfg['fold'],
                       val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                       legacy_rul=cfg['legacy_rul'], global_hi=cfg['global_hi'])
    tr, va = ds['train'], ds['val']
    X, Y, MK, BS, CY, CD = _tensors(tr, dev, ('x', 'y', 'mask', 'baseline', 'cycle', 'cond'))
    CDSEQ = torch.from_numpy(tr['cond_seq']).to(dev)
    HT = torch.from_numpy(tr['hi']).to(dev)
    VX, VY, VMK, VBS, VCY, VCD = _tensors(va, dev, ('x', 'y', 'mask', 'baseline', 'cycle', 'cond'))

    torch.backends.cudnn.benchmark = not cfg['fp32']
    if cfg['fp32']:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
    net = _build(cfg).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=1e-5)
    g = torch.Generator(device=dev).manual_seed(cfg['seed'])
    aug = _augmentor(cfg, dev) if cfg['augment'] else None

    def val_metrics():
        """Return (pooled, engine_equal, engine_mean_roots, predictions).

        Selection uses engine_equal = sqrt(mean_u MSE_u), the study protocol's
        engine-equal RMSE (plan section 10.1). engine_mean_roots is the
        historical Paper B quantity mean_u sqrt(MSE_u); it is reported for
        continuity and never used for selection.
        """
        # Validation always runs in FP32, whatever precision the training used:
        # the reported metric and the checkpoint rule then share one
        # deterministic evaluation path (the historical pipeline reported BF16
        # numbers; the difference measured on one checkpoint was 0.11%).
        p = predict(net, va, dev, cfg['mode'], fp32=True)
        return (rmse(p, va['y']), engine_equal_rmse(va['unit'], va['y'], p),
                per_engine_rmse(va['unit'], va['y'], p), p)

    best, best_state, best_ep, history = float('inf'), None, -1, []
    streams = []
    start_ep = 1
    if cfg['resume'] and (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sch.load_state_dict(ck['sch'])
        # Scheduler.load_state_dict restores last_epoch but does NOT push the
        # decayed learning rate back into the optimizer, so the resumed epoch
        # would otherwise train at the initial LR. Recompute it here.
        for pg, lr in zip(opt.param_groups, sch.get_lr()):
            pg['lr'] = lr
        g.set_state(ck['perm_state'])
        best, best_ep, best_state = ck['best'], ck['best_ep'], ck['best_state']
        history = ck['history']
        start_ep = int(ck['epoch']) + 1
        if aug is not None and ck.get('aug') is not None:
            aug.load_state_dict(ck['aug'])
        rng = ck.get('rng') or {}
        if rng.get('torch') is not None:
            torch.set_rng_state(rng['torch'])
        if rng.get('cuda') and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
        if rng.get('numpy') is not None:
            np.random.set_state(rng['numpy'])
        if rng.get('python') is not None:
            random.setstate(rng['python'])
        print(f'  [resume] from epoch {start_ep} (best {best:.4f}@{best_ep})', flush=True)

    for ep in range(start_ep, cfg['epochs'] + 1):
        net.train()
        dbg = None
        if cfg.get('debug'):
            import hashlib
            rng_fp = (hashlib.sha256(torch.cuda.get_rng_state_all()[0].numpy().tobytes())
                      .hexdigest()[:8] if torch.cuda.is_available() else '-')
            opt_fp = float(sum(float(v['exp_avg'].double().sum())
                               for v in opt.state.values() if 'exp_avg' in v))
            dbg = dict(epoch=ep,
                       wsum=float(sum(float(q.double().sum()) for q in net.parameters())),
                       rng=rng_fp, opt=opt_fp)
        perm = torch.randperm(len(X), generator=g, device=dev)
        tot, nb = 0.0, 0
        for idx in perm.split(cfg['batch']):
            xb, mk = X[idx], MK[idx]
            keep = None
            if aug is not None:
                if aug.mask_gen is not None:
                    keep = aug.sample_mask(int(idx.shape[0]), device=dev)
                if aug.noise_gen is not None:
                    xb = aug.add_noise(xb, mk, CDSEQ[idx])
            if dbg is not None and nb == 0:
                dbg['batch0_x_sum'] = float(xb.double().sum())
                dbg['batch0_kept'] = (-1 if keep is None else int(keep.sum()))
            opt.zero_grad(set_to_none=True)
            with _amp(dev, cfg['fp32']):
                o = net(xb, mask=mk, baseline=BS[idx],
                        cycle=(CY[idx] if net.use_cycle else None), cond=CD[idx],
                        mode=cfg['mode'], sensor_keep=keep)
                y = Y[idx].reshape(-1, 1)
                e = o['prediction'] - y
                loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()
                if getattr(net, 'dual_span', False) and cfg['aux_dual']:
                    eA = o['y_A'] - y
                    eB = o['y_B'] - y
                    loss = loss + 0.25 * (eA.square().mean() + eB.square().mean())
                if cfg['hi_source'] != 'none' and cfg['hi_lambda'] > 0:
                    ht = HT[idx].reshape(-1)
                    loss = loss + cfg['hi_lambda'] * torch.mean((o['hi'].reshape(-1) - ht) ** 2)
                    if cfg['hi_mono'] != 'none' and cfg['hi_mono_coef'] > 0:
                        hs = o['hi_seq']
                        step = (MK[idx][:, 1:] * MK[idx][:, :-1])
                        diff = hs[:, 1:] - hs[:, :-1]
                        # HI is a HEALTH index: it decreases with life, so the
                        # correct monotone penalty punishes increases (diff > 0).
                        pen = torch.relu(diff) if cfg['hi_mono'] == 'correct' else torch.relu(-diff)
                        loss = loss + cfg['hi_mono_coef'] * (pen * step).sum() / step.sum().clamp_min(1.0)
            if dbg is not None and nb == 0:
                dbg['batch0_loss'] = float(loss.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sch.step()
        if dbg is not None:
            dbg['perm0'] = [int(x) for x in perm[:4].tolist()]
            dbg['lr'] = float(opt.param_groups[0]['lr'])
            streams.append(dbg)
        pooled, eq, mor, _ = val_metrics()
        history.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_engine_equal=eq,
                            val_engine_mean_roots=mor, val_pooled_rmse=pooled,
                            lr=float(opt.param_groups[0]['lr'])))
        if ep % 5 == 0 or ep == 1:
            print('  ep%03d loss=%.4f eq=%.4f roots=%.4f pooled=%.4f best=%.4f@%d'
                  % (ep, tot / max(nb, 1), eq, mor, pooled, best, best_ep), flush=True)
        if eq < best - 1e-4:
            best, best_ep = eq, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= cfg['patience']:
            break
        # Every random stream that can affect the remaining epochs is saved:
        # the batch permutation generator, the augmentor generators, and the
        # global RNGs that drive model-internal dropout.
        torch.save(dict(epoch=ep, model=net.state_dict(), opt=opt.state_dict(),
                        sch=sch.state_dict(), best=best, best_ep=best_ep,
                        best_state=best_state, history=history, perm_state=g.get_state(),
                        rng=dict(torch=torch.get_rng_state(),
                                 cuda=(torch.cuda.get_rng_state_all()
                                       if torch.cuda.is_available() else []),
                                 numpy=np.random.get_state(), python=random.getstate()),
                        aug=(aug.state_dict() if aug is not None else None), cfg=cfg),
                   out / 'last.pt')
        if cfg.get('stop_after_epoch') and ep >= int(cfg['stop_after_epoch']):
            print('  [stop] after epoch %d (simulated interruption)' % ep, flush=True)
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    pv = predict(net, va, dev, cfg['mode'], fp32=True)
    pooled = rmse(pv, va['y'])
    eq = engine_equal_rmse(va['unit'], va['y'], pv)
    mor = per_engine_rmse(va['unit'], va['y'], pv)
    m = summarize(pv, va['y'])
    m.update(val_engine_equal_rmse=eq, val_engine_equal_mae=engine_equal_mae(va['unit'], va['y'], pv),
             val_engine_equal_nasa_mean=engine_equal_nasa_mean(va['unit'], va['y'], pv),
             val_engine_mean_roots_rmse=mor,
             val_engine_rmse=pooled, val_pooled_rmse=pooled, best_val_rmse=best,
             best_ep=best_ep, params=int(nparam), epochs_run=len(history),
             elapsed_s=round(time.time() - t0, 1), gpu_uuid=_gpu_uuid(dev),
             visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
             selection_metric=cfg['selection_metric'])
    per_cut = {}
    for c in np.unique(va['cut']):
        sel = va['cut'] == c
        per_cut[int(c)] = dict(n=int(sel.sum()), rmse=rmse(pv[sel], va['y'][sel]))
    m['per_cut'] = per_cut

    torch.save(net.state_dict(), out / 'best.pt')
    (out / 'config.json').write_text(json.dumps(cfg, indent=1, default=str))
    try:
        import yaml
        (out / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:
        pass
    (out / 'metrics.json').write_text(json.dumps(m, indent=1))
    import pandas as pd
    pd.DataFrame(history).to_csv(out / 'train_history.csv', index=False)
    pd.DataFrame(dict(unit=va['unit'], cut=va['cut'], y_true=va['y'], y_pred=pv)).to_csv(
        out / 'val_pred.csv', index=False)
    meta = dict(
        study_id=cfg.get('study_id', ''), run_id=cfg.get('run_id', ''),
        tag=cfg['tag'], subset=cfg['subset'], seed=cfg['seed'],
        host=socket.gethostname(), git_head=_git_head(Path(__file__).resolve().parents[2]),
        torch=torch.__version__, cuda=torch.version.cuda, python=os.sys.version.split()[0],
        gpu_uuid=_gpu_uuid(dev), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        n_train_windows=int(len(X)), n_val_windows=int(len(va['y'])),
        train_units=[int(x) for x in ds['train_units']],
        val_units=[int(x) for x in ds['val_units']],
        augment=cfg['augment'], noise_stats_path=(cfg['augment'] or {}).get('noise_stats_path', ''),
        mask_hash=(cfg['augment'] or {}).get('mask_hash', ''),
        finished=time.time(), elapsed_s=m['elapsed_s'],
    )
    (out / 'metadata.json').write_text(json.dumps(meta, indent=1))
    if aug is not None:
        (out / 'aug_stats.json').write_text(json.dumps(aug.stats(), indent=1))
    if cfg.get('debug'):
        (out / 'debug_streams.json').write_text(json.dumps(streams, indent=1))
    (out / 'status.json').write_text(json.dumps(
        {'status': 'COMPLETE', 'best_ep': best_ep, 'epochs_run': len(history),
         'val_engine_equal_rmse': eq, 'val_engine_mean_roots_rmse': mor,
         'finished': time.time()}, indent=1))
    print('DONE %s %s s%d eq=%.4f roots=%.4f ep=%d params=%d %.1fs' % (
        cfg['tag'], cfg['subset'], cfg['seed'], eq, mor, best_ep, nparam, m['elapsed_s']),
        flush=True)
    return m
