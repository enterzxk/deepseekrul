#!/usr/bin/env python3
"""Independent audit: does the sensor augmentation actually take effect?

Read-only. Runs on CPU, allocates no GPU memory, touches no run directory.

Checks (each is a hard assertion; a failure invalidates the 48-cell grid):
  A1  mask zeroes the hidden sensor in ALL THREE input channels (raw, deviation,
      causal EMA) simultaneously, not just the raw channel
  A2  no hidden value reaches the prediction: d(loss)/d(x) is exactly 0 on the
      hidden channels, and the prediction changes when a sensor is hidden
  A3  the p=0.20 mask is a strict superset of the p=0.10 mask for the same seed
      (the pairing claim in augmentation.py's docstring)
  A4  at least one sensor survives per sample; observed hide rate ~= nominal p
  A5  noise is injected only where the time mask is 1, is never applied in place,
      and its per-step scale comes from the frozen training-side matrix
  A6  creating the augmentor does not perturb model initialisation, so E0 and
      E1/E2 with the same seed start from identical weights (paired design)
  A7  the validation / inference path is untouched: predict() twice returns
      bit-identical output, and it never consumes the augmentor

Usage:
  python scripts/audit_mask_effect.py --data-dir /path/to/CMAPSS [--subset FD001]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paper_b.training.data import prepare_train                      # noqa: E402
from paper_b.training.train import _build, predict, set_seed          # noqa: E402
from paper_b.sensor_aug.augmentation import SensorAugmentor           # noqa: E402
from paper_b.sensor_aug.noise_stats import effective_scale            # noqa: E402

BASE = dict(
    model='dspt', span=70, epochs=40, patience=12, batch=512, lr=1e-3,
    weight_decay=1e-4, d_model=64, layers=2, dropout=0.2, mode='dual',
    ms_mode='full', sa_mode='basic', hi_source='hf', hi_mono='correct',
    hi_lambda=0.5, hi_mono_coef=0.1, use_cycle=True, use_pad_mask=True,
    input_mode='raw', aux_dual=True, instnorm=False, rul_cap=125, fp32=True,
    split_seed=17, fold=0, val_frac=0.2, n_cond=6,
    cuts=[0.4, 0.55, 0.7, 0.85, 0.95],
    legacy_rul=False, global_hi=False, legacy_ms_pad=False, legacy_ema=False,
)

RESULTS = []


def check(name, ok, detail):
    RESULTS.append(dict(check=name, ok=bool(ok), detail=detail))
    print(('PASS  ' if ok else 'FAIL  ') + name + ' :: ' + detail, flush=True)
    return ok


def sd_hash(net):
    h = hashlib.sha256()
    for k in sorted(net.state_dict()):
        h.update(k.encode())
        h.update(net.state_dict()[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--stats', default=None,
                    help='noise_stats_<subset>.json from the frozen build')
    ap.add_argument('--ckpt', default=None,
                    help='best.pt of a completed run, for the trained-model check')
    a = ap.parse_args()

    torch.set_grad_enabled(True)
    dev = torch.device('cpu')
    cfg = {**BASE, 'subset': a.subset, 'seed': 101, 'data_dir': a.data_dir}

    ds = prepare_train(a.subset, Path(a.data_dir), span=cfg['span'],
                       split_seed=cfg['split_seed'], fold=cfg['fold'],
                       val_frac=cfg['val_frac'], rul_cap=cfg['rul_cap'],
                       n_cond=cfg['n_cond'], cuts=tuple(cfg['cuts']),
                       legacy_rul=cfg['legacy_rul'], global_hi=cfg['global_hi'])
    tr, va = ds['train'], ds['val']
    n = min(a.batch, len(tr['x']))
    x0 = torch.from_numpy(np.ascontiguousarray(tr['x'][:n]))
    mk = torch.from_numpy(np.ascontiguousarray(tr['mask'][:n]))
    bs = torch.from_numpy(np.ascontiguousarray(tr['baseline'][:n]))
    cy = torch.from_numpy(np.ascontiguousarray(tr['cycle'][:n]))
    cd = torch.from_numpy(np.ascontiguousarray(tr['cond'][:n]))
    cdseq = torch.from_numpy(np.ascontiguousarray(tr['cond_seq'][:n]))
    B, L, N = x0.shape
    print(f'[{a.subset}] batch B={B} L={L} N={N}  split_seed={cfg["split_seed"]}', flush=True)

    # ---- A6: augmentor construction must not move the init ------------------
    set_seed(101)
    netA = _build(cfg).eval()
    hA = sd_hash(netA)
    set_seed(101)
    _ = SensorAugmentor(mask_prob=0.20, seed=101, device='cpu')
    netB = _build(cfg).eval()
    hB = sd_hash(netB)
    check('A6 init unchanged by augmentor construction', hA == hB,
          f'with={hA} without={hB}')

    net = netA
    net.eval()
    feats_cap = {}
    net.sensor_in.register_forward_hook(
        lambda m, inp, out: feats_cap.__setitem__('f', inp[0].detach()))

    # ---- A1 / A4: mask hides all three channels -----------------------------
    for p in (0.10, 0.20):
        aug = SensorAugmentor(mask_prob=p, seed=101, device='cpu')
        aug.reset_stats()
        keep = aug.sample_mask(B, device='cpu')
        st = aug.stats()
        rate_ok = abs(st['actual_hide_rate'] - p) < 0.02
        check(f'A4a hide rate p={p:.2f}', rate_ok,
              f'actual={st["actual_hide_rate"]:.4f} nominal={p} '
              f'slots={st["mask_slots"]} resampled={st["resampled_samples"]}')
        check(f'A4b at least one sensor kept p={p:.2f}',
              bool(keep.any(dim=1).all()),
              f'min kept per sample={int(keep.sum(dim=1).min())}/{N}')

        with torch.no_grad():
            net(x0, mask=mk, baseline=bs, cycle=cy, cond=cd, mode=cfg['mode'],
                sensor_keep=keep)
        f = feats_cap['f']                                  # (B,L,N,3)
        hidden = ~keep
        per_ch = []
        for c, nm in enumerate(('raw', 'deviation', 'ema')):
            v = f[..., c].permute(0, 2, 1)[hidden]           # (B,N,L) masked
            per_ch.append(float(v.abs().max()) if v.numel() else 0.0)
        check(f'A1 all three channels zeroed at hidden sensors p={p:.2f}',
              max(per_ch) == 0.0 and f.shape[-1] == 3,
              f'n_channels={f.shape[-1]} max|raw|={per_ch[0]:.3e} '
              f'max|dev|={per_ch[1]:.3e} max|ema|={per_ch[2]:.3e}')

        # ---- A2: no gradient path from a hidden sensor ----------------------
        xg = x0.clone().requires_grad_(True)
        out = net(xg, mask=mk, baseline=bs, cycle=cy, cond=cd,
                  mode=cfg['mode'], sensor_keep=keep)
        out['prediction'].sum().backward()
        g = xg.grad.permute(0, 2, 1)                        # (B,N,L)
        hid = float(g[hidden].abs().max())
        kep = float(g[keep].abs().max())
        check(f'A2 no gradient through hidden sensors p={p:.2f}',
              hid == 0.0 and kep > 0.0,
              f'max|grad_hidden|={hid:.3e} max|grad_kept|={kep:.3e}')

        with torch.no_grad():
            base = net(x0, mask=mk, baseline=bs, cycle=cy, cond=cd,
                       mode=cfg['mode'], sensor_keep=None)['prediction']
            masked = net(x0, mask=mk, baseline=bs, cycle=cy, cond=cd,
                         mode=cfg['mode'], sensor_keep=keep)['prediction']
        d = float((masked - base).abs().max())
        check(f'A2b masking changes the prediction p={p:.2f}',
              np.isfinite(d) and d > 1e-3, f'max|dpred|={d:.4f} cycles')

    # ---- A3: p=0.20 superset of p=0.10 -------------------------------------
    k10 = SensorAugmentor(mask_prob=0.10, seed=101, device='cpu').sample_mask(B, 'cpu')
    k20 = SensorAugmentor(mask_prob=0.20, seed=101, device='cpu').sample_mask(B, 'cpu')
    # superset means: hidden at p=0.10 => hidden at p=0.20
    viol = int(((~k10) & k20).sum())
    check('A3 p=0.20 mask is a superset of p=0.10', viol == 0,
          f'violations={viol} hidden10={int((~k10).sum())} '
          f'hidden20={int((~k20).sum())} new_in_20={int(((~k20) & k10).sum())}')

    # ---- A2c: sensitivity of a *trained* model to the same masks -----------
    if a.ckpt and Path(a.ckpt).exists():
        net.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True))
        net.eval()
        with torch.no_grad():
            base = net(x0, mask=mk, baseline=bs, cycle=cy, cond=cd,
                       mode=cfg['mode'], sensor_keep=None)['prediction']
            rows = []
            for p in (0.10, 0.20):
                aug = SensorAugmentor(mask_prob=p, seed=101, device='cpu')
                keep = aug.sample_mask(B, device='cpu')
                pred = net(x0, mask=mk, baseline=bs, cycle=cy, cond=cd,
                           mode=cfg['mode'], sensor_keep=keep)['prediction']
                dd = (pred - base).abs()
                rows.append(dict(p=p, mean=round(float(dd.mean()), 4),
                                 p95=round(float(dd.quantile(0.95)), 4),
                                 max=round(float(dd.max()), 4)))
        check('A2c trained model responds to masking',
              all(r['mean'] > 0.01 for r in rows), json.dumps(rows))
    else:
        check('A2c trained model responds to masking', False,
              'no checkpoint given; pass --ckpt to enable')

    # ---- A5: noise placement and scale -------------------------------------
    if a.stats and Path(a.stats).exists():
        stats = json.loads(Path(a.stats).read_text())
    else:
        cand = REPO / 'experiments' / 'paper_b_sensor_generalization' / 'shared' / \
            f'noise_stats_{a.subset}.json'
        stats = json.loads(cand.read_text()) if cand.exists() else None
    if stats is None:
        check('A5 noise', False, 'no noise_stats json found; skipped')
    else:
        scale = np.asarray(effective_scale(stats, 0.50)['s'], np.float32)
        aug = SensorAugmentor(mask_prob=0.0, noise_scale=scale, seed=101, device='cpu')
        xc = x0.clone()
        xn = aug.add_noise(x0, mk, cdseq)
        pad = mk == 0
        pad_untouched = bool((xn[pad] == xc[pad]).all()) if pad.any() else True
        inplace = bool((x0 != xc).sum() == 0)
        check('A5a padded steps receive no noise', pad_untouched,
              f'padded_values={int(pad.sum())}')
        check('A5b add_noise is not in place', inplace, f'input_changed={not inplace}')
        # Replay the first noise draw from the documented seed and rebuild it
        # from the frozen matrix: noise == scale[cond_seq] * eps * time_mask.
        g3 = torch.Generator(device='cpu').manual_seed(800000 + 101)
        eps3 = torch.randn(xn.shape, generator=g3, device='cpu', dtype=xn.dtype)
        n3 = torch.from_numpy(scale)[cdseq.long()] * eps3 * mk.unsqueeze(-1)
        check('A5c noise = scale[cond_seq] * eps * time_mask',
              bool(torch.equal(xn, x0 + n3)),
              f'max|xn-(x0+n)|={float((xn - (x0 + n3)).abs().max()):.3e}')
        rms = float(((xn - x0)[mk.bool()].double() ** 2).mean().sqrt())
        exp_rms = float((torch.from_numpy(scale)[cdseq.long()][mk.bool()].double() ** 2).mean().sqrt())
        check('A5d injected noise rms matches frozen scale',
              abs(rms - exp_rms) / max(exp_rms, 1e-12) < 0.10,
              f'rms={rms:.6f} expected~{exp_rms:.6f}')
        # sigma source: training engines only
        tr_units = set(int(u) for u in stats.get('train_units', []))
        va_units = set(int(u) for u in stats.get('val_units', []))
        check('A5e noise stats built from training engines only',
              bool(tr_units) and not (tr_units & va_units),
              f'train_units={len(tr_units)} val_units={len(va_units)} '
              f'overlap={len(tr_units & va_units)} split_seed={stats.get("split_seed")}')

    # ---- A7: validation path untouched -------------------------------------
    with torch.no_grad():
        p1 = predict(net, va, dev, cfg['mode'], fp32=True)
        p2 = predict(net, va, dev, cfg['mode'], fp32=True)
    check('A7a predict() is deterministic and noise-free', bool(np.array_equal(p1, p2)),
          f'max|p1-p2|={float(np.abs(p1 - p2).max()):.3e} n={len(p1)}')
    import inspect
    sig = inspect.signature(predict)
    check('A7b predict() has no augmentor argument',
          'augmentor' not in sig.parameters and 'aug' not in sig.parameters,
          f'params={list(sig.parameters)}')

    bad = [r for r in RESULTS if not r['ok']]
    out = dict(subset=a.subset, batch=int(B), n_checks=len(RESULTS),
               n_failed=len(bad), failed=bad, results=RESULTS)
    dest = REPO / 'docs' / 'paper_b_sensor_generalization' / f'MASK_EFFECT_AUDIT_{a.subset}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f'\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed -> {dest}', flush=True)
    print('VERDICT:', 'AUGMENTATION EFFECTIVE' if not bad else 'AUGMENTATION BROKEN', flush=True)
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
