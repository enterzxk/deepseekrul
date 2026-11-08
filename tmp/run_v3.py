"""v3 runner: frozen recipe + optional monotonicity (A) and pinball (B) terms."""
import argparse, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent; ROOT = PARENT.parent
sys.path[:0] = [str(HERE), str(PARENT), str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
import run_fold as base
import run_ida as ida
BATCH = 256

def build_pairs(meta, min_gap=10, seed=0):
    rng = np.random.default_rng(seed)
    unit = meta['unit'].to_numpy(); cyc = meta['end_cycle'].to_numpy()
    n = len(unit); partner = np.arange(n); gap = np.zeros(n)
    g = defaultdict(list)
    for i, (u, c) in enumerate(zip(unit, cyc)): g[u].append((c, i))
    for u, lst in g.items():
        lst.sort()
        cs = np.array([c for c, _ in lst]); ids = np.array([i for _, i in lst])
        for c, i in lst:
            j = np.searchsorted(cs, c + min_gap)
            if j < len(cs):
                p = int(rng.integers(j, len(cs)))
                partner[i] = ids[p]; gap[i] = cs[p] - c
    return partner, gap

def train(arm, net, tr_raw, va_raw, target, meta, seed, epochs=80, l1=0.0, l2=0.0, l3=0.0, tau=0.9):
    if target.exists(): return False
    ida.set_seed(seed)
    g = torch.Generator(device='cuda').manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    tr = [base.device_array(a) for a in tr_raw[:4]]
    partner, gap = build_pairs(meta, 10, seed)
    pidx = torch.as_tensor(partner, device='cuda', dtype=torch.long)
    gt = torch.as_tensor(gap, device='cuda', dtype=torch.float32)
    use_pairs = (l1 > 0) or (l2 > 0)
    def fwd(data, idx):
        return net(data[0][idx], None, data[2][idx], data[3][idx])['prediction'].reshape(-1)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(tr[0]), generator=g, device='cuda').split(BATCH):
            opt.zero_grad(set_to_none=True)
            pred = fwd(tr, idx)
            e = pred - tr[1][idx]
            loss = e.square().mean() + .5 * torch.where(e > 0, 1.8, 1.).mul(e.square()).mean()
            if use_pairs:
                p2 = fwd(tr, pidx[idx])
                if l1 > 0: loss = loss + l1 * torch.relu(p2 - pred).square().mean()
                if l2 > 0:
                    w = (pred > 125).float()
                    loss = loss + l2 * (((pred - p2 - gt[idx]) ** 2) * w).mean()
            if l3 > 0:
                loss = loss + l3 * torch.maximum(tau * e, (tau - 1) * e).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
        sch.step()
    va = [base.device_array(a) for a in va_raw[:4]]
    net.eval()
    with torch.no_grad():
        raw = torch.cat([fwd(va, i) for i in torch.arange(len(va[0]), device='cuda').split(BATCH)])
    raw = raw.cpu().numpy()
    assert np.isfinite(raw).all()
    mu = va_raw[3][:, 0].astype(np.float64) * 250.0
    frame = meta.assign(y_true=va_raw[1], y_pred=raw, raw_output=raw, mu_T=mu)
    tmp = target.with_suffix('.partial'); frame.to_csv(tmp, index=False); os.replace(tmp, target)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset'); ap.add_argument('fold', type=int)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--arms', default='M0,FULL')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--l1', type=float, default=0.0)
    ap.add_argument('--l2', type=float, default=0.0)
    ap.add_argument('--l3', type=float, default=0.0)
    ap.add_argument('--tau', type=float, default=0.9)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    dest = HERE / f'runs_v3_{a.tag}' / a.subset / str(a.fold); dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(a.subset)
    df = df.sort_values(['unit','cycle']).reset_index(drop=True)
    train, valid = folds[a.fold]
    tables = base.prior_tables(df, a.subset, train, valid, a.fold)
    tr, va = ida.prep_fold(df, a.subset, a.fold, train, valid, window=a.window)
    xtr, ytr, etr, mtr = tr[:4]; xva, yva, eva, mva = va[:4]
    lives = df.groupby('unit').cycle.max()
    mtr_df = pd.DataFrame(mtr, columns=['unit','end_cycle'])
    meta = pd.DataFrame(mva, columns=['unit','end_cycle'])
    meta['short'] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta['true_life'] = meta.unit.map(lives)
    for arm in [x for x in a.arms.split(',') if x]:
        body = ida.CandidateModel(num_sensors=14, window_size=a.window, d_model=64, num_layers=1,
                                  dropout=0.15, use_baseline=True, use_ema_features=True,
                                  use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
        net = ida.PriorWrapper(body, 'M1' if arm == 'FULL' else 'M0', d_model=64)
        ida.set_seed(a.seed); net = net.cuda()
        tgt = dest / f'{arm}_{a.seed}.csv'
        w = train(arm, net,
                  [xtr, ytr, etr, base.prior_values(mtr, tables['FULL'])],
                  [xva, yva, eva, base.prior_values(mva, tables['FULL'])],
                  tgt, mtr_df, a.seed, a.epochs, a.l1, a.l2, a.l3, a.tau)
        print('done', a.subset, a.fold, arm, a.tag, 'trained' if w else 'cached', flush=True)

if __name__ == '__main__':
    main()