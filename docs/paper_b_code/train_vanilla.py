"""Train the vanilla-transformer ablation models.

Input to the model is ONLY the normalized sensor sequence + mask.
No 3-channel features, no baseline deviation, no cycle input.
HI labels are fitted per fold on TRAIN engines only.
"""
import argparse, json, sys
import numpy as np, pandas as pd, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'candidates'))
from dataset import (load_cmapss_train, load_cmapss_files,
                     OperatingConditionScaler, SELECTED_SENSOR_COLUMNS)
from candidates.dspt_vanilla import build, MAX_SPAN
from sklearn.cluster import KMeans

BATCH = 256


def rul_curve(n, max_rul=125.0):
    return np.clip(float(n - 1) - np.arange(n, dtype=np.float64), 0.0, max_rul)


def windows_for(frame, values, max_rul=125.0, span=MAX_SPAN, hi_map=None):
    xs, ys, ms, hs = [], [], [], []
    for uid, g in frame.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        cycles = g['cycle'].to_numpy(np.int64)
        rul = rul_curve(len(seq), max_rul)
        for t in range(len(seq)):
            end = t + 1; m = min(end, span)
            x = np.zeros((span, seq.shape[1]), np.float32)
            x[span - m:] = seq[end - m:end]
            mask = np.zeros(span, np.float32); mask[span - m:] = 1.0
            xs.append(x); ys.append(rul[t]); ms.append(mask)
            hs.append(float(hi_map.get((int(uid), int(cycles[t])), 0.0)) if hi_map else 0.0)
    return (np.stack(xs), np.asarray(ys, np.float32), np.stack(ms), np.asarray(hs, np.float32))


def fit_hi(train_units, tr_frame, tr_scaled, seed=42):
    d = tr_frame[tr_frame.unit.isin(train_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    pos = d.index.to_numpy(np.int64)
    X = np.asarray(tr_scaled[pos], dtype=np.float64)
    ops = d[['op_1', 'op_2', 'op_3']].to_numpy(np.float64)
    k = int(min(6, max(1, len(np.unique(ops, axis=0)))))
    lab = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(ops).labels_
    Z = X.copy()
    for c in range(k):
        mm = lab == c
        if mm.sum() < 2: continue
        Z[mm] = (Z[mm] - Z[mm].mean(0)) / (Z[mm].std(0) + 1e-9)
    w, V = np.linalg.eigh(np.cov(Z.T))
    pc1 = Z @ V[:, -1]
    cyc = d['cycle'].to_numpy(float)
    if np.corrcoef(pc1, cyc)[0, 1] > 0: pc1 = -pc1
    lo, hi = np.percentile(pc1, 1), np.percentile(pc1, 99)
    val = np.clip((pc1 - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return {(int(u), int(c)): float(v) for u, c, v in zip(d['unit'].to_numpy(), cyc, val)}


def evaluate(net, data, device, bs=512):
    net.eval()
    x, y, mk = data[0], data[1], data[2]
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            o = net(torch.from_numpy(x[i:i+bs]).to(device), mask=torch.from_numpy(mk[i:i+bs]).to(device))
            preds.append(o['prediction'].reshape(-1).cpu().numpy())
    p = np.concatenate(preds)
    return float(np.sqrt(np.mean((p - y) ** 2))), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--split-seed', type=int, default=7)
    ap.add_argument('--span', type=int, default=70)
    ap.add_argument('--rul-cap', type=float, default=125.0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--ms', action='store_true')
    ap.add_argument('--sa', action='store_true')
    ap.add_argument('--hi', action='store_true')
    ap.add_argument('--hi-lambda', type=float, default=0.5)
    ap.add_argument('--hi-mono', type=float, default=0.1)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    dev = torch.device('cuda')
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    raw = load_cmapss_train(ROOT / 'data' / 'CMAPSS', a.subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(a.split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size * 0.2)))
    val_units = sorted(int(x) for x in sh[a.fold * n_val:(a.fold + 1) * n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    va = raw[raw.unit.isin(val_units)].sort_values(['unit', 'cycle']).reset_index(drop=True)
    cols = list(SELECTED_SENSOR_COLUMNS)
    scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=a.split_seed)
    tr_s = scaler.fit_transform(tr); va_s = scaler.transform(va)

    HI = None
    if a.hi:
        HI = fit_hi(train_units, tr, tr_s, a.split_seed)
        print('HI fitted on %d TRAIN engines only: %d entries' % (len(train_units), len(HI)), flush=True)
    TR = windows_for(tr, tr_s, max_rul=a.rul_cap, span=a.span, hi_map=HI)
    VA = windows_for(va, va_s, max_rul=a.rul_cap, span=a.span, hi_map=HI)
    print('windows train %d val %d' % (len(TR[0]), len(VA[0])), flush=True)

    net = build(14, a.d_model, a.layers, a.dropout, max_span=a.span,
                use_ms=a.ms, use_sa=a.sa, use_hi=a.hi).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    g = torch.Generator(device='cuda').manual_seed(a.seed)
    X, Y, MK, HT = [torch.from_numpy(v).to(dev) for v in TR]
    best, best_state, best_ep = float('inf'), None, -1
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(X), generator=g, device='cuda')
        for idx in perm.split(BATCH):
            opt.zero_grad(set_to_none=True)
            out = net(X[idx], mask=MK[idx])
            y = Y[idx].reshape(-1, 1)
            e = out['prediction'] - y
            loss = e.square().mean() + 0.5 * torch.where(e > 0, 1.8, 1.0).mul(e.square()).mean()
            if a.hi:
                loss = loss + a.hi_lambda * torch.mean((out['hi'].reshape(-1) - HT[idx].reshape(-1)) ** 2)
                if a.hi_mono > 0:
                    hs = out['hi_seq']
                    loss = loss + a.hi_mono * torch.mean(torch.relu(hs[:, 1:] - hs[:, :-1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sch.step()
        vr, _ = evaluate(net, VA, dev)
        if ep % 5 == 0 or ep == 1:
            print('ep%03d val=%.4f best=%.4f@%d' % (ep, vr, best, best_ep), flush=True)
        if vr < best - 1e-4:
            best, best_ep = vr, ep
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        elif ep - best_ep >= a.patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    _, pv = evaluate(net, VA, dev)
    vr = float(np.sqrt(np.mean((pv - VA[1]) ** 2)))
    out_dir = ROOT / 'experiments' / a.tag / a.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_dir / ('v_s%d.pt' % a.seed))
    pd.DataFrame({'unit': np.concatenate([[u] * int((va.unit == u).sum()) for u in val_units]),
                  'y_true': VA[1], 'y_pred': pv}).to_csv(out_dir / ('v_s%d.csv' % a.seed), index=False)
    (out_dir / ('v_s%d.json' % a.seed)).write_text(json.dumps({
        'subset': a.subset, 'fold': a.fold, 'seed': a.seed, 'params': nparam,
        'best_val': best, 'best_ep': best_ep, 'span': a.span,
        'ms': a.ms, 'sa': a.sa, 'hi': a.hi}))
    print('DONE %s f%d s%d params=%d best_val=%.4f ep=%d' % (a.subset, a.fold, a.seed, nparam, vr, best_ep), flush=True)


if __name__ == '__main__':
    main()
