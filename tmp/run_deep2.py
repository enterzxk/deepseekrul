"""Phase-A runner: same frozen recipe, configurable dropout / weight decay / width."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent; ROOT = PARENT.parent
sys.path[:0] = [str(HERE), str(PARENT), str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
import run_fold as base
import run_ida as ida
import run_full_fold3 as r3
from deep_body import DeepBody
BATCH = 256

def build(arm, window, d_model, layers, extra, dropout):
    body = ida.CandidateModel(num_sensors=14, window_size=window, d_model=d_model, num_layers=layers,
                              dropout=dropout, use_baseline=True, use_ema_features=True,
                              use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
    deep = DeepBody(body, d_model, dropout=dropout, heads=4, extra_layers=extra)
    return ida.PriorWrapper(deep, 'M1' if arm == 'FULL' else 'M0', d_model=d_model)

def train2(arm, net, train_data, valid_data, target, metadata, seed, epochs=80, wd=1e-4):
    if target.exists(): return False
    ida.set_seed(seed)
    g = torch.Generator(device='cuda').manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    tr = [base.device_array(a) for a in train_data[:4]]
    def fwd(data, idx):
        x, y, extras, prior = (v[idx] for v in data[:4])
        return net(x, None, extras, prior)['prediction'].reshape(-1)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(tr[0]), generator=g, device='cuda').split(BATCH):
            opt.zero_grad(set_to_none=True)
            e = fwd(tr, idx) - tr[1][idx]
            loss = e.square().mean() + .5 * torch.where(e > 0, 1.8, 1.).mul(e.square()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
        sch.step()
    va = [base.device_array(a) for a in valid_data[:4]]
    net.eval()
    with torch.no_grad():
        raw = torch.cat([net(va[0][i], None, va[2][i], va[3][i])['prediction'].reshape(-1)
                         for i in torch.arange(len(va[0]), device='cuda').split(BATCH)])
    raw = raw.cpu().numpy()
    assert np.isfinite(raw).all()
    mu = valid_data[3][:, 0].astype(np.float64) * 250.0
    frame = metadata.assign(y_true=valid_data[1], y_pred=raw, raw_output=raw, mu_T=mu)
    tmp = target.with_suffix('.partial'); frame.to_csv(tmp, index=False); os.replace(tmp, target)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset'); ap.add_argument('fold', type=int)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--arms', default='M0')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--extra-layers', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--tag', default='sweep')
    a = ap.parse_args()
    dest = HERE / f'runs_sweep_{a.tag}' / a.subset / str(a.fold); dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(a.subset)
    df = df.sort_values(['unit','cycle']).reset_index(drop=True)
    train, valid = folds[a.fold]
    tables = base.prior_tables(df, a.subset, train, valid, a.fold)
    tr, va = ida.prep_fold(df, a.subset, a.fold, train, valid, window=a.window)
    xtr, ytr, etr, mtr = tr[:4]; xva, yva, eva, mva = va[:4]
    lives = df.groupby('unit').cycle.max()
    meta = pd.DataFrame(mva, columns=['unit','end_cycle'])
    meta['short'] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta['true_life'] = meta.unit.map(lives)
    for arm in [x for x in a.arms.split(',') if x]:
        net = build(arm, a.window, a.d_model, a.layers, a.extra_layers, a.dropout)
        n = sum(p.numel() for p in net.parameters())
        ida.set_seed(a.seed); net = net.cuda()
        tgt = dest / f'{arm}_{a.seed}.csv'
        w = train2(arm, net, [xtr,ytr,etr,base.prior_values(mtr,tables['FULL'])],
                   [xva,yva,eva,base.prior_values(mva,tables['FULL'])], tgt, meta, a.seed, a.epochs, a.wd)
        print('done', a.subset, a.fold, arm, n, 'trained' if w else 'cached', flush=True)

if __name__ == '__main__':
    main()