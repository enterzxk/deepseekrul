"""Stage D trainer: engine-level damage recursion, three losses, four arms.

Why this is a separate module and not a flag on ``train_hc.py``: the Stage C loop
trains on independent windows, so the recursion ``D_t = sum_{u<=t} delta_u``
cannot be formed (a window carries no time index). Stage D must walk whole
trajectories, so the batching unit changes from "window" to "engine", and the
loss gains two engine-level terms. Keeping them in one function would mean two
different batching rules behind one name.

The four arms differ ONLY in what produces the RUL prediction and whether the
damage path is used as an auxiliary signal:

  B0  direct head, no damage path at all            L_RUL
  B1  direct head + damage path as auxiliary        L_RUL + lam_T L_term + lam_S L_smooth/S0
  B2  arrival only, fixed rate nu_t                 L_RUL + lam_T L_term + lam_S L_smooth/S0
  B3  arrival only, learned rate g_phi(., k)        same as B2

B2/B3 must not carry a bypass RUL head (task book section 7 D5): the regression
head's parameters are excluded from the optimiser, so it cannot act as one.

Numerics that are not negotiable:

  * The encoder runs under bf16 autocast, but the damage recursion runs in
    float32. Each increment is ~5e-3 and the path sums ~500 of them to reach 1;
    bf16's 8-bit mantissa would cost percent-level accuracy in the accumulated
    damage, which is exactly the quantity the threshold comparison reads.
  * ``S0`` is read from the shared preparation record, never recomputed per
    epoch: a moving normaliser makes the objective drift.
  * The training loss uses the raw operator output. Clipping at ``rul_cap``
    happens only in evaluation (D4.1/D4.6).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..training.metrics import per_engine_rmse, rmse, summarize
from . import config_hc as C
from .arrival import first_passage
from .data_d import engine_batch, train_lifetime_mean
from .hi_d import DSPT_HC_Structured

ARMS = ('B0', 'B1', 'B2', 'B3')
HI_ARMS = ('B1', 'B2', 'B3')
ARRIVAL_ARMS = ('B2', 'B3')


def _now():
    import datetime
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_d(arm, subset, K, l_mean, head_seed, dev, init_from, seed=101):
    """Build one arm. The head init is seeded and identical for B1/B2/B3."""
    cfg = dict(C.hc_config('A1', subset, seed))
    bb = cfg['backbone']
    net = DSPT_HC_Structured(
        fixed_rate=(arm == 'B2'), K=int(K), cycle_mode='none',
        num_sensors=14, d_model=cfg['d_model'], num_layers=cfg['layers'],
        dropout=cfg['dropout'], max_span=cfg['span'],
        ms_mode=bb['ms_mode'], sa_mode=bb['sa_mode'],
        hi_source=bb['hi_source'], hi_mono=bb['hi_mono'],
        use_pad_mask=True, n_cond=cfg['n_cond'],
        l_mean=(None if arm == 'B0' else l_mean),
        head_seed=(None if arm == 'B0' else head_seed))
    ck = torch.load(init_from, map_location='cpu', weights_only=True)
    sd = dict(ck['state_dict'] if 'state_dict' in ck else ck)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [m for m in missing if not m.startswith('head.')]
    if bad or unexpected:
        raise RuntimeError('init mismatch: missing=%s unexpected=%s' % (bad, unexpected))
    return net.to(dev), sorted(missing)


def trainable(net, arm):
    """B0 never touches the damage head; B2/B3 never touch the regression head.

    The exclusion is the enforcement of D5's "no bypass RUL head" constraint:
    an unused head whose parameters are still optimised is a bypass waiting to
    be used. ``requires_grad_`` is set in BOTH directions -- leaving the excluded
    head at its default True would keep it in ``net.parameters()`` and it would
    silently be trained (and would still count as trainable in the audit).
    """
    out = []
    for name, p in net.named_parameters():
        keep = True
        if arm == 'B0' and name.startswith('head.'):
            keep = False
        if arm in ARRIVAL_ARMS and name.startswith('reg_head.'):
            keep = False
        p.requires_grad_(keep)
        if keep:
            out.append(p)
    return out


def module_counts(net, arm):
    def n(prefix):
        return int(sum(p.numel() for k, p in net.named_parameters()
                       if k.startswith(prefix)))
    return dict(backbone=n('') - n('head.') - n('reg_head.'),
                damage_head=n('head.'), reg_head=n('reg_head.'),
                trainable=int(sum(p.numel() for p in net.parameters() if p.requires_grad)))


# --------------------------------------------------------------------------- #
# tensors
# --------------------------------------------------------------------------- #
def engine_tensors(eb, i, n, dev):
    x = torch.from_numpy(np.asarray(eb['x'][i, :n], np.float32)).to(dev)
    m = torch.from_numpy(np.asarray(eb['mask'][i, :n], np.float32)).to(dev)
    b = torch.from_numpy(np.asarray(eb['baseline'][i, :n], np.float32)).to(dev)
    cd = torch.from_numpy(np.asarray(eb['cond'][i, :n], np.int64)).to(dev)
    cy = torch.zeros(n, device=dev)
    r = torch.from_numpy(np.asarray(eb['r'][i, :n], np.float32)).to(dev)
    return x, m, b, cy, cd, r


def forward_engine(net, arm, eb, i, n, dev, K):
    """One engine: z_t (fp32), damage path, and the RUL prediction."""
    x, m, b, cy, cd, r = engine_tensors(eb, i, n, dev)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                        enabled=(dev.type == 'cuda')):
        z = net.pooled_feature(x, m, b, cy, cd)
    # fp32 from here on: the recursion sums ~500 increments of ~5e-3.
    z = z.float()
    if arm == 'B0':
        return dict(pred=net.reg_head(z).reshape(-1), r=r, n=n)
    delta = net.head.increment(z.unsqueeze(0)).reshape(-1)
    D = torch.cumsum(delta, dim=0)
    nu = net.current_rate(delta.unsqueeze(0)).reshape(-1)
    out = dict(delta=delta, D=D, nu=nu, r=r, n=n, z=z)
    if arm == 'B1':
        out['pred'] = net.reg_head(z).reshape(-1)
    else:
        out['pred'] = first_passage(D, net.head.future_rates(D, nu, z, K))
    return out


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def rul_loss(pred, r, cap):
    """Censored hinge (D4.1). Uncropped: no clamp of pred at cap."""
    capped = r >= cap
    return torch.where(capped, torch.clamp(cap - pred, min=0.0) ** 2,
                       (pred - r) ** 2)


def losses_for(net, arm, fo, cap, s0, lam):
    L = dict(rul=rul_loss(fo['pred'], fo['r'], cap).mean())
    if arm in HI_ARMS:
        L['terminal'] = (fo['D'][-1] - 1.0) ** 2
        if fo['delta'].numel() > 1:
            L['smooth'] = ((fo['delta'][1:] - fo['delta'][:-1]) ** 2).mean() / s0
        else:
            L['smooth'] = torch.zeros((), device=fo['delta'].device)
    total = L['rul']
    if 'terminal' in L:
        total = total + lam * L['terminal'] + lam * L['smooth']
    return total, L


# --------------------------------------------------------------------------- #
# preparation: l_mean, b0, S0, head fingerprint  (once per subset)
# --------------------------------------------------------------------------- #
def calibrate_s0(net, ebt, dev, K, n_batches=8, bs=8, seed=0, floor=1e-12):
    """S0 = E[(delta_{t+1} - delta_t)^2] on the initial weights, detached.

    Two things here are not style:

      * ``net.eval()`` is mandatory. With dropout on, the second difference is
        measured on a randomly masked network, S0 comes out ~2.7x too large and
        seed-dependent -- the same failure shape as incident 14. The assertion
        below turns that into a crash instead of a wrong normaliser.
      * No bf16 autocast. ``probes/d_acceptance.py`` measures S0 in fp32, and the
        two records must agree digit for digit or an auditor cannot tell whether
        the run used the calibrated value. bf16 is a training-speed choice.
    """
    if net.training:
        raise AssertionError('S0 must be measured with dropout off: call net.eval() first')
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    E = len(ebt['lengths'])
    vals = []
    with torch.no_grad():
        for _ in range(n_batches):
            for i in torch.randint(0, E, (bs,), generator=g).tolist():
                n = int(ebt['lengths'][i])
                x, m, b, cy, cd, _ = engine_tensors(ebt, i, n, dev)
                z = net.pooled_feature(x, m, b, cy, cd).float()
                d = net.head.increment(z.unsqueeze(0)).reshape(-1)
                if d.numel() > 1:
                    vals.append(float(((d[1:] - d[:-1]) ** 2).mean()))
    raw = float(np.mean(vals))
    return dict(raw=raw, used=max(raw, floor), floor=floor,
                floor_hit=bool(raw < floor), n_batches=n_batches, batch=bs,
                seed=int(seed), n_terms=len(vals), dropout='eval')


def prepare(subset, data_dir, K, head_seed, dev, init_from, out_dir):
    """Write the shared record every arm of this subset reads."""
    lrec = train_lifetime_mean(subset, data_dir)
    net, _ = build_d('B3', subset, K, lrec['mean'], head_seed, dev, init_from)
    net.eval()  # S0 is a property of the init, not of a dropout draw
    ebt = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='train')
    s0 = calibrate_s0(net, ebt, dev, K)
    rec = dict(subset=subset, K=int(K), head_seed=int(head_seed),
               l_mean=lrec['mean'], train_n=lrec['n'],
               train_units=lrec['units'], train_lengths=lrec['lengths'],
               b0=net.head.init_b0, softplus_b0=float(F.softplus(
                   torch.tensor(net.head.init_b0))),
               head_fingerprint=net.head.fingerprint(),
               s0=s0, created_utc=_now(),
               note='measured once per subset; B1/B2/B3 read the same numbers')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / ('%s_shared.json' % subset)
    if p.exists():
        raise SystemExit('REFUSED: %s already exists; the shared record is written once' % p)
    p.write_text(json.dumps(rec, indent=1), encoding='utf-8')
    return p, rec


# --------------------------------------------------------------------------- #
# evaluation (D4.6 six quantities)
# --------------------------------------------------------------------------- #
def evaluate(net, arm, eb, dev, K, cap):
    """Cut-cycle evaluation: 5 cuts per engine, engine-mean RMSE (frozen metric)."""
    net.eval()
    preds, ys, units, cuts = [], [], [], []
    with torch.no_grad():
        for i in range(len(eb['lengths'])):
            n = int(eb['lengths'][i])
            fo = forward_engine(net, arm, eb, i, n, dev, K)
            ct = eb['cut_t'][i]
            cv = eb['cut_valid'][i]
            for j in range(len(ct)):
                if cv[j] <= 0:
                    continue
                t = int(ct[j])
                preds.append(float(fo['pred'][t - 1]))
                ys.append(float(n - t))
                units.append(int(eb['engine_id'][i]))
                cuts.append(j)
    preds = np.asarray(preds, np.float64)
    ys = np.asarray(ys, np.float64)
    units = np.asarray(units)
    cuts = np.asarray(cuts)
    clipped = np.clip(preds, 0.0, cap)
    m = dict(n=int(ys.size),
             val_engine_mean_rmse=per_engine_rmse(units, ys, clipped),
             val_pooled_rmse=rmse(clipped, ys))
    unc = ys < cap
    capd = ~unc
    m['frac_capped'] = float(capd.mean()) if ys.size else None
    m['rmse_uncensored'] = float(np.sqrt(np.mean((clipped[unc] - ys[unc]) ** 2))) if unc.any() else None
    # rmse_uncapped: uncapped truth, but only where the truth is inside the
    # model's horizon -- beyond K the operator's output cannot represent it and
    # including those points would flatter the model (D4.6 item 4).
    inK = ys <= K
    m['rmse_uncapped'] = (float(np.sqrt(np.mean((preds[inK] - ys[inK]) ** 2)))
                          if inK.any() else None)
    m['frac_r_gt_K'] = float((~inK).mean()) if ys.size else None
    m['boundary_hit'] = float((preds[capd] >= cap).mean()) if capd.any() else None
    m['exact_boundary'] = float((np.abs(preds[capd] - cap) < 1e-6).mean()) if capd.any() else None
    m['rmse_by_censoring'] = dict(
        uncensored=(float(np.sqrt(np.mean((clipped[unc] - ys[unc]) ** 2))) if unc.any() else None),
        censored=(float(np.sqrt(np.mean((clipped[capd] - ys[capd]) ** 2))) if capd.any() else None))
    m['per_cut'] = {int(c): dict(n=int((cuts == c).sum()),
                                 rmse=float(np.sqrt(np.mean((clipped[cuts == c] - ys[cuts == c]) ** 2))))
                    for c in np.unique(cuts)}
    m['_preds'] = preds
    m['_ys'] = ys
    m['_units'] = units
    m['_cuts'] = cuts
    return m


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def term_grads(net, arm, ebt, dev, K, cap, s0, lam, n_eng=3):
    """D4.5: gradient contribution of each loss term, on a fixed sample.

    Reports BOTH readings, because the spec is ambiguous and the two disagree:

      unweighted  g_i = ||grad_theta L_i||          (D4.5's literal wording)
      weighted    g_i = ||grad_theta lambda_i L_i|| (what the optimiser sees)

    The literal reading has a defect: it ignores lambda entirely, so it would
    prescribe a 1755x up-weight of L_terminal and a 45x up-weight of L_smooth --
    which then makes the weighted smoothness gradient 1563x the RUL gradient.
    The weighted reading is the one that describes the actual update, so the
    ratio used for the gate is the weighted one; the unweighted one is kept for
    the record.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    keys = ('rul', 'terminal', 'smooth') if arm in HI_ARMS else ('rul',)
    acc = {k: 0.0 for k in keys}
    for i in range(min(n_eng, len(ebt['lengths']))):
        n = int(ebt['lengths'][i])
        fo = forward_engine(net, arm, ebt, i, n, dev, K)
        _, L = losses_for(net, arm, fo, cap, s0, lam)
        for k, v in L.items():
            if k not in acc:
                continue
            gs = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
            acc[k] += float(sum(g.norm() ** 2 for g in gs if g is not None)) ** 0.5
    m = float(min(n_eng, len(ebt['lengths'])))
    unweighted = {k: v / m for k, v in acc.items()}
    weighted = {k: (v if k == 'rul' else v * lam) for k, v in unweighted.items()}
    return dict(unweighted=unweighted, weighted=weighted)


def train_arm(arm, subset, seed, data_dir, shared_path, out_dir, dev,
              init_from, K, epochs=None, patience=None, batch=None, lr=None,
              weight_decay=None, lam=None, log=print, budget_override=None):
    sh = json.loads(Path(shared_path).read_text(encoding='utf-8'))
    K = int(sh['K'] if K is None else K)
    s0 = float(sh['s0']['used'])
    l_mean = float(sh['l_mean'])
    lam = float(C.RUL_CAP ** 2 if lam is None else lam)
    epochs = int(C.EPOCHS if epochs is None else epochs)
    patience = int(C.PATIENCE if patience is None else patience)
    batch = int(C.BATCH if batch is None else batch)
    lr = float(C.LR if lr is None else lr)
    weight_decay = float(C.WEIGHT_DECAY if weight_decay is None else weight_decay)
    cap = float(C.RUL_CAP)

    set_seed(seed)
    t0 = time.time()
    dev = torch.device(dev)
    net, new_keys = build_d(arm, subset, K, l_mean, int(sh['head_seed']), dev,
                            init_from, seed=seed)
    params = trainable(net, arm)
    nparam = module_counts(net, arm)
    ebt = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='train')
    ebv = engine_batch(subset, data_dir, C.SPAN, C.VAL_CUTS, split='val')
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    E = len(ebt['lengths'])

    best, best_state, best_ep, history = float('inf'), None, -1, []
    for ep in range(1, epochs + 1):
        net.train()
        order = torch.randperm(E, generator=g).tolist()
        opt.zero_grad(set_to_none=True)
        pts, tot = 0, 0.0
        for i in order:
            n = int(ebt['lengths'][i])
            fo = forward_engine(net, arm, ebt, i, n, dev, K)
            loss, _ = losses_for(net, arm, fo, cap, s0, lam)
            # weight by the point count so the accumulated gradient is the mean
            # over the ~`batch` points, not the sum of per-engine means
            (loss * (n / float(batch))).backward()
            pts += n
            tot += float(loss.detach()) * n
            if pts >= batch:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                pts = 0
        if pts > 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        sch.step()

        ev = evaluate(net, arm, ebv, dev, K, cap)
        tg = term_grads(net, arm, ebt, dev, K, cap, s0, lam)
        w = tg['weighted']
        mx = max(w.values()) if w else 0.0
        ratio = {k: (v / mx if mx > 0 else None) for k, v in w.items()}
        tl = tot / max(1, _epoch_points(ebt))
        history.append(dict(epoch=ep, train_loss=tl,
                            val_engine_mean_rmse=ev['val_engine_mean_rmse'],
                            val_pooled_rmse=ev['val_pooled_rmse'],
                            grad_terms_unweighted=tg['unweighted'],
                            grad_terms_weighted=w, grad_ratio=ratio))
        if ep % 5 == 0 or ep == 1:
            log('  ep%03d loss=%.4f val_eng=%.4f g_ratio=%s'
                % (ep, tl, ev['val_engine_mean_rmse'],
                   {k: ('%.3e' % v if v is not None else None)
                    for k, v in ratio.items()}))
        if ev['val_engine_mean_rmse'] < best - 1e-4:
            best, best_ep = ev['val_engine_mean_rmse'], ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)

    ev = evaluate(net, arm, ebv, dev, K, cap)
    preds, ys = ev.pop('_preds'), ev.pop('_ys')
    units = ev.pop('_units')
    cuts = ev.pop('_cuts')
    out = Path(out_dir)
    (out / 'ckpt').mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / 'ckpt' / 'best.pt')
    m = dict(arm=arm, subset=subset, seed=int(seed), K=K, s0=s0, lam=lam,
             l_mean=l_mean, head_fingerprint=sh['head_fingerprint'],
             params=nparam, epochs_run=len(history), best_ep=best_ep,
             best_val_engine_mean_rmse=best, elapsed_s=round(time.time() - t0, 1),
             select_on_val=True, budget=dict(epochs=epochs, patience=patience,
                                             batch_points=batch, lr=lr,
                                             weight_decay=weight_decay),
             budget_override=budget_override,
             created_utc=_now())
    m.update(ev)
    m['grad_terms_last_unweighted'] = history[-1]['grad_terms_unweighted']
    m['grad_terms_last_weighted'] = history[-1]['grad_terms_weighted']
    m['grad_ratio_last'] = history[-1]['grad_ratio']
    (out / 'metrics.json').write_text(json.dumps(m, indent=1), encoding='utf-8')
    (out / 'history.json').write_text(json.dumps(history, indent=1), encoding='utf-8')
    import pandas as pd
    pd.DataFrame(dict(engine_id=units, cut=cuts, y_true=ys, y_pred=preds,
                      split='val')).to_csv(out / 'preds.csv', index=False)
    log('DONE %s %s s%d val_eng=%.4f ep=%d params=%d %.1fs'
        % (arm, subset, seed, ev['val_engine_mean_rmse'], best_ep,
           nparam['trainable'], m['elapsed_s']))
    return m


def _epoch_points(ebt):
    return int(np.asarray(ebt['lengths']).sum())


# --------------------------------------------------------------------------- #
# four-arm execution config audit
# --------------------------------------------------------------------------- #
def audit(arms=ARMS, subset='FD001', data_dir=None, K=600, head_seed=101,
          dev='cpu', init_from=None, l_mean=None, root=None):
    """Build all four arms and report exactly what differs.

    The point is not to print a table -- it is to make the differences a fact of
    the built objects, not of the document. Anything that differs beyond the
    three intended axes (RUL source, HI auxiliary, rate mode) is a bug.

    ``l_mean`` should come from the shared preparation record: with a dummy
    value the head fingerprint will not match the acceptance record, and the
    audit would then be certifying a different init than the one that trains.
    """
    if l_mean is None:
        p = Path(root or '.') / ('%s_shared.json' % subset)
        if not p.exists():
            raise SystemExit('run `prepare` first, or pass l_mean: %s missing' % p)
        l_mean = float(json.loads(p.read_text(encoding='utf-8'))['l_mean'])
    rows = {}
    for arm in arms:
        net, new_keys = build_d(arm, subset, K, l_mean, head_seed, dev, init_from)
        trainable(net, arm)
        rows[arm] = dict(
            params=module_counts(net, arm),
            has_damage_head=arm != 'B0',
            damage_head_trained=arm in HI_ARMS,
            reg_head_trained=arm in ('B0', 'B1'),
            uses_arrival=arm in ARRIVAL_ARMS,
            fixed_rate=(arm == 'B2'),
            head_fingerprint=(None if arm == 'B0' else net.head.fingerprint()),
            new_tensors=len(new_keys),
            K=int(net.K))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'train', 'audit'])
    ap.add_argument('--subset', default='FD001')
    ap.add_argument('--arm', default='B3', choices=ARMS)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--K', type=int, default=600)
    ap.add_argument('--head-seed', type=int, default=101)
    ap.add_argument('--data-dir', default='/media/sangfor/vdb/user/zxk/deepseek-rul/data/CMAPSS')
    ap.add_argument('--init-from',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/tmp/'
                            'wave1_init_A1_s101.pt')
    ap.add_argument('--root',
                    default='/media/sangfor/vdb/user/zxk/deepseek-rul/experiments/hc/d')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--budget-override', default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    root = Path(a.root)

    if a.stage == 'prepare':
        p, rec = prepare(a.subset, a.data_dir, a.K, a.head_seed, dev, a.init_from, root)
        print(json.dumps(dict(path=str(p), l_mean=rec['l_mean'], b0=rec['b0'],
                              softplus_b0=rec['softplus_b0'],
                              head_fingerprint=rec['head_fingerprint'],
                              s0=rec['s0']['used'],
                              s0_floor_hit=rec['s0']['floor_hit']), indent=1))
    elif a.stage == 'audit':
        rows = audit(subset=a.subset, data_dir=a.data_dir, K=a.K,
                     head_seed=a.head_seed, dev=dev, init_from=a.init_from,
                     root=root)
        print(json.dumps(rows, indent=1))
        fps = {k: v['head_fingerprint'] for k, v in rows.items() if k != 'B0'}
        if len(set(fps.values())) != 1:
            raise SystemExit('head init differs across B1/B2/B3: %s' % fps)
        print('\nhead init identical across B1/B2/B3: %s' % next(iter(fps.values())))
    else:
        shared = root / ('%s_shared.json' % a.subset)
        out = root / ('%s_%s_s%d' % (a.arm, a.subset, a.seed))
        train_arm(a.arm, a.subset, a.seed, a.data_dir, shared, out, dev,
                  a.init_from, a.K, epochs=a.epochs,
                  budget_override=a.budget_override)


if __name__ == '__main__':
    main()

