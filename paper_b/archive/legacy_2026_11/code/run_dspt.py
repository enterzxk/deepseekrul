"""DSPT runner: padded 100-step windows, early stopping, engine-disjoint folds."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = Path('/home/user/rulb')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
from dataset import load_cmapss_train, OperatingConditionScaler, SELECTED_SENSOR_COLUMNS
from candidates.dspt import build, MAX_SPAN, SHORT_SPAN
from sklearn.cluster import KMeans

BATCH = 256
CYCLE_SCALE = 250.0

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def rul_curve(n, max_rul=125.0, legacy=False):
    if legacy:
        r = float(n) - np.arange(n, dtype=np.float64)
        return np.clip(r, 0.0, max_rul)
    """True RUL at step t is n-t (cycles remaining), capped at max_rul."""
    r = float(n - 1) - np.arange(n, dtype=np.float64)
    return np.clip(r, 0.0, max_rul)

def windows_for(frame, values, labels, pop_ref, tau=2.0, max_rul=125.0, span=MAX_SPAN, hi_map=None):
    xs, ys, ms, bs, cs, hs = [], [], [], [], [], []
    for uid, g in frame.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        clusters = np.asarray(labels[pos], dtype=np.int64)
        cycles = g['cycle'].to_numpy(np.int64)
        rul = rul_curve(len(seq), max_rul, legacy=legacy_rul)
        for t in range(len(seq)):
            end = t + 1; m = min(end, span)
            w = seq[end - m:end]
            pad = span - m
            x = np.zeros((span, seq.shape[1]), np.float32)
            x[pad:] = w
            mask = np.zeros(span, np.float32); mask[pad:] = 1.0
            cl = int(clusters[t])
            early = np.where((cycles <= 30) & (clusters == cl) & (np.arange(len(seq)) <= t))[0]
            si = early[-5:] if early.size > 5 else early
            n = int(si.size)
            if n > 0:
                xbar = seq[si].mean(axis=0)
            else:
                xbar = np.zeros(seq.shape[1], np.float32)
            ref = (n/(n+tau))*xbar + (tau/(n+tau))*pop_ref[cl]
            xs.append(x); ys.append(rul[t]); ms.append(mask); bs.append(ref.astype(np.float32))
            cs.append(min(cycles[t]/CYCLE_SCALE, 1.0))
            hs.append(float(hi_map.get((int(uid), int(cycles[t])), 0.0)) if hi_map else 0.0)
    return (np.stack(xs), np.asarray(ys, np.float32), np.stack(ms),
            np.stack(bs), np.asarray(cs, np.float32), np.asarray(hs, np.float32))

def evaluate(net, data, device, bs=512, no_cycle=False, mode='dual'):
    net.eval(); x, y, mk, b, c = data[0], data[1], data[2], data[3], data[4]; preds = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            cyc = None if no_cycle else torch.from_numpy(c[i:i+bs]).to(device)
            out = net(torch.from_numpy(x[i:i+bs]).to(device), mask=torch.from_numpy(mk[i:i+bs]).to(device),
                      baseline=torch.from_numpy(b[i:i+bs]).to(device), cycle=cyc)
            key = 'y_B' if mode == 'long' else ('y_A' if mode in ('short','repeat') else 'prediction')
            preds.append(out[key].reshape(-1).cpu().numpy())
    p = np.concatenate(preds)
    return float(np.sqrt(np.mean((p - y)**2))), p


def fit_hi(train_units, tr_frame, tr_scaled, seed=42):
    """Health indicator fitted ONLY on the training engines of this fold.
    Returns dict (unit, cycle) -> hi in [0,1]. No validation/test information used."""
    from sklearn.cluster import KMeans
    d = tr_frame[tr_frame.unit.isin(train_units)].sort_values(['unit','cycle']).reset_index(drop=True)
    pos = d.index.to_numpy(np.int64)
    X = np.asarray(tr_scaled[pos], dtype=np.float64)
    ops = d[['op_1','op_2','op_3']].to_numpy(np.float64)
    nreg = len(np.unique(ops, axis=0))
    k = int(min(6, max(1, nreg)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(ops)
    lab = km.labels_
    Z = X.copy()
    for c in range(k):
        m = lab == c
        if m.sum() < 2: continue
        mu = Z[m].mean(0); sd = Z[m].std(0) + 1e-9
        Z[m] = (Z[m] - mu) / sd
    C = np.cov(Z.T)
    w, V = np.linalg.eigh(C)
    pc1 = Z @ V[:, -1]
    cyc = d['cycle'].to_numpy(float)
    if np.corrcoef(pc1, cyc)[0,1] > 0: pc1 = -pc1
    lo, hi = np.percentile(pc1, 1), np.percentile(pc1, 99)
    val = np.clip((pc1 - lo)/(hi - lo + 1e-9), 0.0, 1.0)
    return {(int(u), int(c)): float(v) for u, c, v in zip(d['unit'].to_numpy(), cyc, val)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset'); ap.add_argument('fold', type=int)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--split-seed', type=int, default=7)
    ap.add_argument('--mode', default='dual', choices=['dual','short','long','repeat','slot'])
    ap.add_argument('--instnorm', action='store_true')
    ap.add_argument('--wshift', action='store_true')
    ap.add_argument('--span', type=int, default=100)
    ap.add_argument('--rul-cap', type=float, default=125.0)
    ap.add_argument('--full-data', action='store_true')
    ap.add_argument('--no-cycle', action='store_true')
    ap.add_argument('--hi-lambda', type=float, default=0.0)
    ap.add_argument('--hi-mono', type=float, default=0.0)
    ap.add_argument('--no-ms', action='store_true')
    ap.add_argument('--legacy-pad', action='store_true')
    ap.add_argument('--legacy-rul', action='store_true')
    ap.add_argument('--legacy-hi', action='store_true')
    ap.add_argument('--no-sa', action='store_true')
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    _out = ROOT/'experiments'/a.tag/a.subset
    if (_out/f'dspt_s{a.seed}.csv').exists():
        print('SKIP %s (exists)' % a.tag, flush=True); return
    dev = torch.device('cuda')
    raw = load_cmapss_train(ROOT/'data'/'CMAPSS', a.subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(a.split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size*0.2)))
    if getattr(a, "full_data", False):
        val_units = sorted(int(x) for x in units)
    else:
        val_units = sorted(int(x) for x in sh[a.fold*n_val:(a.fold+1)*n_val])
    if getattr(a, "full_data", False):
        train_units = sorted(int(x) for x in units)
    else:
        train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit','cycle']).reset_index(drop=True)
    va = raw[raw.unit.isin(val_units)].sort_values(['unit','cycle']).reset_index(drop=True)
    cols = list(SELECTED_SENSOR_COLUMNS); op = ['op_1','op_2','op_3']
    scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=a.split_seed)
    tr_s = scaler.fit_transform(tr); va_s = scaler.transform(va)
    km = KMeans(n_clusters=6, random_state=a.split_seed, n_init=10).fit(tr[op].to_numpy(np.float64))
    lt = km.labels_.astype(np.int64); lv = km.predict(va[op].to_numpy(np.float64)).astype(np.int64)
    pop = np.zeros((6, tr_s.shape[1]), np.float32)
    em = tr['cycle'].to_numpy(np.int64) <= 15
    for k in range(6):
        sel = em & (lt == k)
        pop[k] = tr_s[sel].mean(axis=0) if sel.any() else tr_s.mean(axis=0)
    HI = None
    if getattr(a, 'hi_lambda', 0.0) > 0:
        if getattr(a, 'legacy_hi', False):
            hid = pd.read_csv(ROOT/'data'/'HI'/f'hi_{a.subset}.csv')
            HI = hid.set_index(['unit','cycle'])['hi_raw'].to_dict()
            print('HI LEGACY global', len(HI), flush=True)
        else:
            HI = fit_hi(train_units, tr, tr_s, a.split_seed)
        print('HI fitted on %d TRAIN engines only: %d entries' % (len(train_units), len(HI)), flush=True)
    TR = windows_for(tr, tr_s, lt, pop, max_rul=a.rul_cap, span=a.span, hi_map=HI, legacy_rul=getattr(a,'legacy_rul',False)); VA = windows_for(va, va_s, lv, pop, max_rul=a.rul_cap, span=a.span, hi_map=HI, legacy_rul=getattr(a,'legacy_rul',False))
    print('windows train %d val %d' % (len(TR[0]), len(VA[0])), flush=True)
    SW = None
    if a.wshift:
        CAPW = a.rul_cap
        lens = np.array([len(g) for _, g in tr.groupby("unit")], float)
        ys = np.arange(0, int(CAPW) + 1)
        p_te = np.array([min((lens > y).sum(), (lens > CAPW).sum()) for y in ys], float)
        p_te = p_te / p_te.sum()
        Yr = np.round(np.asarray(TR[1], np.float64)).astype(int)
        p_tr = np.bincount(np.clip(Yr, 0, int(CAPW)), minlength=int(CAPW) + 1).astype(float)
        p_tr = p_tr / p_tr.sum()
        wv = np.where(p_tr > 1e-12, p_te / np.maximum(p_tr, 1e-12), 0.0)
        wv = wv / (p_tr * wv).sum()
        SW = torch.from_numpy(wv[np.clip(Yr, 0, int(CAPW))].astype(np.float32)).to(dev)
        print("  wshift: weight range [%.3f, %.3f] ESS %.1f%%" % (float(wv.min()), float(wv.max()), 100*float((p_tr*wv).sum()**2/((p_tr*wv**2).sum()))), flush=True)
    net = build(14, a.d_model, a.layers, a.dropout, instnorm=getattr(a, 'instnorm', False), max_span=a.span, use_ms=not getattr(a, 'no_ms', False), use_sa=not getattr(a, 'no_sa', False), use_pad_mask=not getattr(a, 'legacy_pad', False)).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    g = torch.Generator(device='cuda').manual_seed(a.seed)
    X, Y, MK, BS, CY, HIT = [torch.from_numpy(v).to(dev) for v in TR]
    best, best_state, best_ep = float('inf'), None, -1
    ema_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    EMA_DECAY = 0.995
    for ep in range(1, a.epochs+1):
        net.train()
        perm = torch.randperm(len(X), generator=g, device='cuda')
        for idx in perm.split(BATCH):
            opt.zero_grad(set_to_none=True)
            out = net(X[idx], mask=MK[idx], baseline=BS[idx], cycle=(None if getattr(a, "no_cycle", False) else CY[idx]), mode=a.mode)
            if a.mode == "short":
                out["prediction"] = out["y_A"]
            elif a.mode == "long":
                out["prediction"] = out["y_B"]
            elif a.mode == "repeat":
                out["y_B"] = out["y_A"]
                out["prediction"] = out["y_A"]
            y = Y[idx].reshape(-1, 1)
            e = out['prediction'] - y
            sw = None if SW is None else SW[idx].reshape(-1, 1)
            def wm(v):
                return v.mean() if sw is None else (v * sw).sum() / sw.sum()
            loss = wm(e.square()) + 0.5 * wm(torch.where(e > 0, 1.8, 1.0).mul(e.square()))
            eA = out['y_A'] - y; eB = out['y_B'] - y
            loss = loss + 0.5 * (wm(eA.square()) + wm(eB.square())) / 2
            if a.hi_lambda > 0:
                hh = out['hi'].reshape(-1)
                htarget = HIT[idx].reshape(-1)
                loss = loss + a.hi_lambda * torch.mean((hh - htarget)**2)
                if a.hi_mono > 0:
                    hseq = out['hi_seq']
                    loss = loss + a.hi_mono * torch.mean(torch.relu(hseq[:, 1:] - hseq[:, :-1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for _k, _v in net.state_dict().items():
                    if _v.dtype.is_floating_point:
                        ema_state[_k].mul_(EMA_DECAY).add_(_v.detach(), alpha=1.0-EMA_DECAY)
                    else:
                        ema_state[_k].copy_(_v)
        sch.step()
        vr, _ = evaluate(net, VA, dev, no_cycle=getattr(a, "no_cycle", False), mode=a.mode)
        if ep % 5 == 0 or ep == 1:
            print("ep%03d val=%.4f best=%.4f@%d" % (ep, vr, best, best_ep), flush=True)
        if getattr(a, "full_data", False):
            best, best_ep = vr, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        elif vr < best - 1e-4:
            best, best_ep = vr, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        elif (not getattr(a, "full_data", False)) and ep - best_ep >= a.patience:
            break
    if best_state is not None: net.load_state_dict(best_state)
    _bs = {k: t.detach().clone() for k, t in net.state_dict().items()}
    _vb = evaluate(net, VA, dev, no_cycle=getattr(a, 'no_cycle', False), mode=a.mode)[0]
    net.load_state_dict(ema_state)
    _ve = evaluate(net, VA, dev, no_cycle=getattr(a, 'no_cycle', False), mode=a.mode)[0]
    if _ve < _vb:
        print("  using EMA weights (%.4f < %.4f)" % (_ve, _vb), flush=True)
    else:
        net.load_state_dict(_bs)
    def ev(mode):
        net.eval(); ps=[]
        with torch.no_grad():
            for i in range(0,len(VA[0]),512):
                o = net(torch.from_numpy(VA[0][i:i+512]).to(dev), mode=a.mode,
                        mask=torch.from_numpy(VA[2][i:i+512]).to(dev),
                        baseline=torch.from_numpy(VA[3][i:i+512]).to(dev),
                        cycle=torch.from_numpy(VA[4][i:i+512]).to(dev))
                p = o["y_A"] if mode=="short" else (o["y_B"] if mode=="long" else (o["y_A"] if mode=="repeat" else o["prediction"]))
                ps.append(p.reshape(-1).cpu().numpy())
        return np.concatenate(ps)
    pv = ev(a.mode); vr = float(np.sqrt(np.mean((pv - VA[1])**2)))
    out_dir = ROOT/'experiments'/a.tag/a.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir/f'dspt_s{a.seed}.pt')
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'unit': np.concatenate([[u]*int((va.unit==u).sum()) for u in val_units]),
                  'y_true': VA[1], 'y_pred': pv}).to_csv(out_dir/f'dspt_s{a.seed}.csv', index=False)
    (out_dir/f'dspt_s{a.seed}.json').write_text(json.dumps({'subset':a.subset,'fold':a.fold,'seed':a.seed,
        'params':nparam,'best_val':best,'best_ep':best_ep,'layers':a.layers,'dropout':a.dropout}))
    print('DONE %s f%d s%d params=%d best_val=%.4f ep=%d' % (a.subset, a.fold, a.seed, nparam, vr, best_ep), flush=True)

if __name__ == '__main__':
    main()