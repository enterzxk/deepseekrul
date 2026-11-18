"""Post-hoc causal temporal aggregation for Stage D B0 checkpoints.

This is an inference-only module: it never trains and never uses labels to fit
anything. For each validation cut ``c`` it builds the K causal windows ending at
``c, c-delta, ..., c-(K-1)*delta``, predicts RUL at each, aligns each prediction
to the cut (a window ending d cycles earlier must predict d more cycles), and
aggregates.

Why alignment is mandatory: a window ending at ``c-d`` has true RUL
``RUL(c) + d``. Averaging raw predictions without subtracting ``d`` biases the
estimate upward by roughly ``delta*(K-1)/2`` cycles, which is larger than any
effect we are looking for.

The script writes a JSON with one row per (K, delta, aggregator) and prints a
compact table. ``K=1`` must reproduce the checkpoint's ``preds.csv``; that
assertion is the parity check that the model/data path is the trained one.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from paper_b.hc import config_hc as C
from paper_b.hc.data_d import engine_batch
from paper_b.hc.hi_d import DSPT_HC_Structured


def build_model(subset, seed, ckpt, dev):
    cfg = C.hc_config('A1', subset, seed)
    bb = cfg['backbone']
    net = DSPT_HC_Structured(
        fixed_rate=False, K=600, l_mean=None, head_seed=None,
        cycle_mode='none', num_sensors=14, d_model=cfg['d_model'],
        num_layers=cfg['layers'], dropout=cfg['dropout'], max_span=cfg['span'],
        ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
        hi_source=bb['hi_source'], hi_mono=bb['hi_mono'],
        use_pad_mask=True, n_cond=cfg['n_cond'])
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    net.load_state_dict(sd, strict=True)
    return net.to(dev).eval()


def predict_windows(net, eb, i, idxs, dev):
    """Predict RUL for the windows whose 0-based end index is in ``idxs``."""
    idxs = np.asarray(idxs, np.int64)
    x = torch.from_numpy(np.asarray(eb['x'][i, idxs], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][i, idxs], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][i, idxs], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][i, idxs], np.int64)).to(dev)
    cy = torch.zeros(len(idxs), device=dev)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=(dev.type == 'cuda')):
            z = net.pooled_feature(x, m, b, cy, cd)
        p = net.reg_head(z.float()).reshape(-1)
    return p.detach().cpu().numpy()


def aggregate(aligned, idxs, cut, how):
    """Aggregate aligned predictions. ``aligned`` is RUL at ``cut`` per window."""
    if len(aligned) == 0:
        raise ValueError('no window')
    if how == 'mean':
        return float(aligned.mean())
    if how == 'median':
        return float(np.median(aligned))
    if how == 'trim':
        if len(aligned) < 3:
            return float(aligned.mean())
        s = np.sort(aligned)
        return float(s[1:-1].mean())
    if how == 'recency':
        # newest window gets the largest weight; geometric decay over window age
        w = 0.7 ** np.arange(len(aligned))
        return float((aligned * w).sum() / w.sum())
    if how == 'linfit':
        # Estimate the failure cycle a_t = p_t + (t+1) = aligned + cut, fit a
        # line over t, evaluate at the cut, convert back to RUL. This is the
        # only aggregator that can correct a systematic drift in the
        # prediction trajectory.
        a = aligned + cut
        if len(a) < 2:
            return float(aligned.mean())
        coef = np.polyfit(idxs.astype(np.float64), a, 1)
        a_hat = np.polyval(coef, cut - 1)
        return float(a_hat - cut)
    raise ValueError(how)


def nasa_score(pred, y):
    d = np.asarray(pred, np.float64) - np.asarray(y, np.float64)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                                np.exp(d / 10.0) - 1.0)))


def evaluate_config(rows, cfg_name):
    """rows: list of dict(engine, cut, y_cap, y_raw, pred)."""
    by_engine = {}
    for r in rows:
        by_engine.setdefault(r['engine'], []).append(r)
    eng_rmse = []
    eng_nasa = []
    all_p, all_y = [], []
    unc_p, unc_y = [], []
    for u, rs in by_engine.items():
        p = np.array([r['pred'] for r in rs], np.float64)
        y = np.array([r['y_cap'] for r in rs], np.float64)
        p = np.clip(p, 0.0, 125.0)
        eng_rmse.append(float(np.sqrt(np.mean((p - y) ** 2))))
        eng_nasa.append(nasa_score(p, y))
        all_p.append(p); all_y.append(y)
        yr = np.array([r['y_raw'] for r in rs], np.float64)
        m = yr < 125.0
        if m.any():
            unc_p.append(p[m]); unc_y.append(yr[m])
    p = np.concatenate(all_p); y = np.concatenate(all_y)
    out = dict(config=cfg_name,
               engine_mean_rmse=float(np.mean(eng_rmse)),
               engine_rmse_sd=float(np.std(eng_rmse, ddof=1)) if len(eng_rmse) > 1 else 0.0,
               pooled_rmse=float(np.sqrt(np.mean((p - y) ** 2))),
               nasa_mean=float(np.mean(eng_nasa)),
               mae=float(np.mean(np.abs(p - y))),
               bias=float(np.mean(p - y)),
               n_engines=len(eng_rmse))
    if unc_p:
        up = np.concatenate(unc_p); uy = np.concatenate(unc_y)
        out['rmse_uncensored'] = float(np.sqrt(np.mean((up - uy) ** 2)))
        out['n_uncensored'] = int(len(up))
    if any('gated' in r for r in rows):
        out['gate_rate'] = float(np.mean([r.get('gated', 0) for r in rows]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--root', default=str(Path.home() / 'sprint24/experiments/hc/_gate'))
    ap.add_argument('--data-dir', default=str(Path.home() / 'rulb/data/CMAPSS'))
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--K', default='1,3,5,7')
    ap.add_argument('--delta', default='5,10')
    ap.add_argument('--agg', default='mean,median,trim,recency,linfit')
    ap.add_argument('--match-cond', action='store_true',
                    help='keep only windows whose operating-condition label equals the cut condition')
    ap.add_argument('--cap-mode', choices=['none', 'drop', 'clamp'], default='none',
                    help='how to treat windows whose prediction is at the RUL cap')
    ap.add_argument('--gate-tau', default='',
                    help='comma-separated thresholds for reliability-gated aggregation; '
                         'empty disables the gate')
    ap.add_argument('--gate-mode', choices=['mad', 'shift', 'rel'], default='mad',
                    help='gate statistic: MAD of aligned predictions, |median-newest|, '
                         'or MAD/|median|')
    ap.add_argument('--gate-shift-tau', default='',
                    help='comma-separated extra shift thresholds (cycles); aggregation '
                         'requires BOTH the gate statistic and |median-newest| to pass')
    ap.add_argument('--random-rate', type=float, default=None,
                    help='placebo gate: aggregate with this probability regardless of '
                         'the dispersion statistic (same rate as the real gate)')
    ap.add_argument('--unaligned', action='store_true',
                    help='control: aggregate raw predictions without the age alignment')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    root = Path(a.root)
    run = 'HC-D-B0-%s-s%d-D1' % (a.subset, a.seed)
    ckpt = Path(a.ckpt) if a.ckpt else root / run / 'ckpt' / 'best.pt'
    if not ckpt.exists():
        raise SystemExit('missing checkpoint %s' % ckpt)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    net = build_model(a.subset, a.seed, ckpt, dev)
    eb = engine_batch(a.subset, a.data_dir, C.SPAN, C.VAL_CUTS, split='val')
    Ks = [int(x) for x in a.K.split(',')]
    deltas = [int(x) for x in a.delta.split(',')]
    aggs = a.agg.split(',')
    kmax = max(Ks)

    # parity: K=1 must reproduce the checkpoint's own preds.csv. Stage D writes
    # the cut *index* (0..4) in ``preds.csv`` while ``cut_t`` stores the cycle;
    # support both spellings so the check actually runs on d/d80 roots.
    ref = {}
    ref_idx = {}
    p = root / run / 'preds.csv'
    if p.exists():
        import pandas as pd
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            uid = int(r['engine_id'])
            cut = int(r['cut'])
            ref[(uid, cut)] = float(r['y_pred'])
            ref_idx.setdefault(uid, []).append(float(r['y_pred']))

    results = []
    parity = []
    rng = np.random.default_rng(12345)
    for i in range(len(eb['lengths'])):
        n = int(eb['lengths'][i])
        uid = int(eb['engine_id'][i])
        cuts = [int(c) for c in eb['cut_t'][i] if int(c) > 0]
        for j, c in enumerate(cuts):
            cur_cond = int(eb['cond'][i, c - 1])
            for di, delta in enumerate(deltas):
                idxs = [c - 1 - j * delta for j in range(kmax)]
                idxs = [t for t in idxs if t >= 0]
                if a.match_cond:
                    same = [t for t in idxs if int(eb['cond'][i, t]) == cur_cond]
                    idxs = same if same else [c - 1]
                preds = predict_windows(net, eb, i, idxs, dev)
                # K=1 is the newest window regardless of delta; check parity once.
                if di == 0 and not a.match_cond and a.cap_mode == 'none':
                    refv = ref.get((uid, c))
                    if refv is None and j < len(ref_idx.get(uid, [])):
                        refv = ref_idx[uid][j]
                    if refv is not None:
                        parity.append(abs(float(preds[0]) - refv))
                for K in Ks:
                    k = min(K, len(idxs))
                    sel_idx = np.array(idxs[:k], np.int64)
                    sel_pred = preds[:k]
                    aligned = sel_pred - (c - (sel_idx + 1))
                    if a.unaligned:
                        aligned = sel_pred.copy()
                    if a.cap_mode == 'clamp':
                        aligned = np.where(sel_pred >= 124.5, 125.0, aligned)
                    elif a.cap_mode == 'drop':
                        keep = sel_pred < 124.5
                        if keep.any():
                            sel_idx, aligned = sel_idx[keep], aligned[keep]
                        else:
                            sel_idx = np.array([c - 1], np.int64)
                            aligned = np.array([min(float(sel_pred[0]), 125.0)])
                    for how in aggs:
                        val = aggregate(aligned, sel_idx, c, how)
                        suffix = (',cm' if a.match_cond else '') + (
                            ',cap=%s' % a.cap_mode if a.cap_mode != 'none' else '')
                        results.append(dict(engine=uid, cut=c,
                                            y_cap=float(min(n - c, 125)),
                                            y_raw=float(n - c), pred=val,
                                            K=K, delta=delta, agg=how,
                                            variant=suffix))
                    if a.gate_tau:
                        med = float(np.median(aligned))
                        disp = float(np.median(np.abs(aligned - med)))
                        newest = float(preds[0])
                        if a.gate_mode == 'mad':
                            stat = disp
                        elif a.gate_mode == 'shift':
                            stat = abs(med - newest)
                        else:
                            stat = disp / max(abs(med), 1.0)
                        taus = [float(x) for x in a.gate_tau.split(',')]
                        for tau in taus:
                            val = med if stat <= tau else newest
                            suffix = (',cm' if a.match_cond else '') + (
                                ',cap=%s' % a.cap_mode if a.cap_mode != 'none' else '') + (
                                ',gate=%g' % tau) + (
                                ',gm=%s' % a.gate_mode if a.gate_mode != 'mad' else '')
                            results.append(dict(engine=uid, cut=c,
                                                y_cap=float(min(n - c, 125)),
                                                y_raw=float(n - c), pred=val,
                                                K=K, delta=delta, agg='gate',
                                                variant=suffix,
                                                gated=int(stat <= tau)))
                        if a.gate_shift_tau:
                            shift = abs(med - newest)
                            for tau in taus:
                                for gs in [float(x) for x in a.gate_shift_tau.split(',')]:
                                    val = med if (stat <= tau and shift <= gs) else newest
                                    suffix = (',cm' if a.match_cond else '') + (
                                        ',cap=%s' % a.cap_mode if a.cap_mode != 'none' else '') + (
                                        ',gate=%g' % tau) + (
                                        ',gm=%s' % a.gate_mode if a.gate_mode != 'mad' else '') + (
                                        ',gs=%g' % gs)
                                    results.append(dict(engine=uid, cut=c,
                                                        y_cap=float(min(n - c, 125)),
                                                        y_raw=float(n - c), pred=val,
                                                        K=K, delta=delta, agg='gate',
                                                        variant=suffix,
                                                        gated=int(stat <= tau and shift <= gs)))
                    if a.random_rate is not None:
                        med = float(np.median(aligned))
                        newest = float(preds[0])
                        u = float(rng.random())
                        val = med if u <= a.random_rate else newest
                        suffix = (',cm' if a.match_cond else '') + (
                            ',cap=%s' % a.cap_mode if a.cap_mode != 'none' else '') + (
                            ',rg=%g' % a.random_rate)
                        results.append(dict(engine=uid, cut=c,
                                            y_cap=float(min(n - c, 125)),
                                            y_raw=float(n - c), pred=val,
                                            K=K, delta=delta, agg='randomgate',
                                            variant=suffix,
                                            gated=int(u <= a.random_rate)))
    # group by (K,delta,agg)
    groups = {}
    for r in results:
        key = (r['K'], r['delta'], r['agg'], r['variant'])
        groups.setdefault(key, []).append(r)
    table = [evaluate_config(v, 'K=%d,delta=%d,%s%s' % (k[0], k[1], k[2], k[3]))
             for k, v in sorted(groups.items())]
    out = dict(subset=a.subset, seed=a.seed, ckpt=str(ckpt), n_engines=int(len(eb['lengths'])),
               parity_max_abs_diff=(float(np.max(parity)) if parity else None),
               parity_n=len(parity), table=table)
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
