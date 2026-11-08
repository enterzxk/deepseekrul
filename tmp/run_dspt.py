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

def rul_curve(n, max_rul=125.0):
    r = max_rul - np.arange(n, dtype=np.float64)
    return np.clip(r, 0.0, max_rul)

def windows_for(frame, values, labels, pop_ref, tau=2.0, max_rul=125.0):
    xs, ys, ms, bs, cs = [], [], [], [], []
    for uid, g in frame.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        clusters = np.asarray(labels[pos], dtype=np.int64)
        cycles = g['cycle'].to_numpy(np.int64)
        rul = rul_curve(len(seq), max_rul)
        for t in range(len(seq)):
            end = t + 1; m = min(end, MAX_SPAN)
            w = seq[end - m:end]
            pad = MAX_SPAN - m
            x = np.zeros((MAX_SPAN, seq.shape[1]), np.float32)
            x[pad:] = w
            mask = np.zeros(MAX_SPAN, np.float32); mask[pad:] = 1.0
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
    return (np.stack(xs), np.asarray(ys, np.float32), np.stack(ms),
            np.stack(bs), np.asarray(cs, np.float32))

def evaluate(net, data, device, bs=512):
    net.eval(); x, y, mk, b, c = data; preds = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out = net(torch.from_numpy(x[i:i+bs]).to(device), mask=torch.from_numpy(mk[i:i+bs]).to(device),
                      baseline=torch.from_numpy(b[i:i+bs]).to(device), cycle=torch.from_numpy(c[i:i+bs]).to(device))
            preds.append(out['prediction'].reshape(-1).cpu().numpy())
    p = np.concatenate(preds)
    return float(np.sqrt(np.mean((p - y)**2))), p

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
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    set_seed(a.seed)
    dev = torch.device('cuda')
    raw = load_cmapss_train(ROOT/'data'/'CMAPSS', a.subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(a.split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size*0.2)))
    val_units = sorted(int(x) for x in sh[a.fold*n_val:(a.fold+1)*n_val])
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
    TR = windows_for(tr, tr_s, lt, pop); VA = windows_for(va, va_s, lv, pop)
    print('windows train %d val %d' % (len(TR[0]), len(VA[0])), flush=True)
    net = build(14, a.d_model, a.layers, a.dropout).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    g = torch.Generator(device='cuda').manual_seed(a.seed)
    X, Y, MK, BS, CY = [torch.from_numpy(v) for v in TR]
    best, best_state, best_ep = float('inf'), None, -1
    for ep in range(1, a.epochs+1):
        net.train()
        perm = torch.randperm(len(X), generator=g, device='cuda')
        for idx in perm.split(BATCH):
            opt.zero_grad(set_to_none=True)
            out = net(X[idx].to(dev), mask=MK[idx].to(dev), baseline=BS[idx].to(dev), cycle=CY[idx].to(dev))
            y = Y[idx].to(dev).reshape(-1, 1)
            e = out['prediction'] - y
            loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()
            eA = out['y_A'] - y; eB = out['y_B'] - y
            loss = loss + 0.5 * (eA.square().mean() + eB.square().mean()) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sch.step()
        vr, _ = evaluate(net, VA, dev)
        if vr < best - 1e-4:
            best, best_ep = vr, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        elif ep - best_ep >= a.patience:
            break
    if best_state is not None: net.load_state_dict(best_state)
    vr, pv = evaluate(net, VA, dev)
    out_dir = ROOT/'experiments'/a.tag/a.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'unit': np.concatenate([[u]*int((va.unit==u).sum()) for u in val_units]),
                  'y_true': VA[1], 'y_pred': pv}).to_csv(out_dir/f'dspt_s{a.seed}.csv', index=False)
    (out_dir/f'dspt_s{a.seed}.json').write_text(json.dumps({'subset':a.subset,'fold':a.fold,'seed':a.seed,
        'params':nparam,'best_val':best,'best_ep':best_ep,'layers':a.layers,'dropout':a.dropout}))
    print('DONE %s f%d s%d params=%d best_val=%.4f ep=%d' % (a.subset, a.fold, a.seed, nparam, vr, best_ep), flush=True)

if __name__ == '__main__':
    main()