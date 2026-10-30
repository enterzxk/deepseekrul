"""R3 scale-normalised time experiment: cycle vs none vs oracle c/T (upper bound)."""
import argparse, json, sys, time, hashlib
import numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS, SensorMinMaxScaler
from candidates.model import CandidateModel
import run_candidate as rc

COLS = list(SELECTED_SENSOR_COLUMNS); CYC = len(COLS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['cycle','none','oracle'], required=True)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
    life = df.groupby('unit')['cycle'].max()
    units = np.sort(df.unit.unique()); rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(units)); nval = max(8, len(units)//5)
    val_units = set(units[perm[:nval]].tolist()); tr_units = set(units[perm[nval:]].tolist())
    tr_df = df[df.unit.isin(tr_units)].reset_index(drop=True)
    va_df = df[df.unit.isin(val_units)].reset_index(drop=True)
    sc = SensorMinMaxScaler(COLS).fit(tr_df[COLS])
    # per-engine training windows (stride 1) so we know the engine of every window
    Xs, Ys, Es, Ts = [], [], [], []
    for u in sorted(tr_units):
        g = tr_df[tr_df.unit == u].reset_index(drop=True)
        v = sc.transform(g[COLS])
        x, y, e = rc.build_windows(g, v, a.window, 1, 125.0)
        if len(x) == 0: continue
        Xs.append(x); Ys.append(y); Es.append(e); Ts.append(np.full(len(x), float(life[u])))
    Xtr = np.concatenate(Xs); Ytr = np.concatenate(Ys); Etr = np.concatenate(Es); Ttr = np.concatenate(Ts)
    # validation: one matched window per val engine
    va_df2 = va_df.copy().reset_index(drop=True)
    Xva, Yva, Eva, units_va = rc.build_windows_matched(va_df2, sc.transform(va_df2[COLS]), a.window, 125.0)
    Tva = np.array([float(life[u]) for u in units_va])
    raw_tr = np.minimum(Etr[:, CYC] * rc.CYCLE_SCALE, 1e9)
    raw_va = np.minimum(Eva[:, CYC] * rc.CYCLE_SCALE, 1e9)
    def apply(E, raw, T):
        if a.arm == 'none': E[:, CYC] = 0.0
        elif a.arm == 'oracle': E[:, CYC] = np.clip(raw / T, 0, 1)
        return E
    Etr = apply(Etr.copy(), raw_tr, Ttr); Eva = apply(Eva.copy(), raw_va, Tva)
    print(f'train windows={len(Xtr)} val={len(Xva)} arm={a.arm} feat_range=[{Etr[:,CYC].min():.3f},{Etr[:,CYC].max():.3f}]')
    model = CandidateModel(num_sensors=len(COLS), window_size=a.window, d_model=64, num_heads=4, num_layers=1,
                           dropout=0.15, use_baseline=True, use_ema_features=True, use_cycle=True,
                           use_graph=False, use_dual_head=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.MSELoss()
    t = lambda z: torch.tensor(z, device=dev)
    Xt, Yt, Et = t(Xtr), t(Ytr).unsqueeze(1), t(Etr)
    Xv, Yv, Ev = t(Xva), t(Yva).unsqueeze(1), t(Eva)
    best = (1e9, None, 0); hist = []; n = len(Xt); t0 = time.time()
    for ep in range(1, a.epochs+1):
        model.train(); idx = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, 64):
            b = idx[i:i+64]
            pred = model(Xt[b], extras=Et[b])['prediction']; delta = pred - Yt[b]
            late = torch.where(delta > 0, 1.8*delta.square(), delta.square())
            loss = lossf(pred, Yt[b]) + 0.5*late.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += float(loss)*len(b)
        model.eval()
        with torch.no_grad(): pv = model(Xv, extras=Ev)['prediction']
        rmse = float(torch.sqrt(torch.mean((pv-Yv)**2))); hist.append(dict(epoch=ep, val_rmse=rmse))
        if rmse < best[0]: best = (rmse, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): pv = model(Xv, extras=Ev)['prediction'].cpu().numpy().reshape(-1)
    pv = np.clip(pv, 0, 125.0); e = pv - Yva
    nz = np.where(e < 0, np.exp(-e/13)-1, np.exp(e/10)-1)
    q1, q2 = np.quantile(Tva, [1/3, 2/3])
    bands = {'short': Tva <= q1, 'mid': (Tva > q1) & (Tva <= q2), 'long': Tva > q2}
    per = {k: dict(n=int(m.sum()), rmse=float(np.sqrt((e[m]**2).mean())), bias=float(e[m].mean()), late=float((e[m]>0).mean()))
           for k, m in bands.items() if m.any()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'pred.npz', y_true=Yva, y_pred=pv, life=Tva, arm=a.arm, seed=a.seed)
    res = dict(arm=a.arm, seed=a.seed, best_val_rmse=best[0], best_epoch=best[2], epochs_ran=len(hist),
               n_train=int(len(Xtr)), n_val=int(len(Xva)), rmse=float(np.sqrt((e**2).mean())),
               mae=float(np.abs(e).mean()), nasa_mean=float(nz.mean()), late=float((e>0).mean()),
               severe_e10=float((e>10).mean()), by_lifespan=per, seconds=time.time()-t0,
               code_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())
    json.dump(res, open(out/'metrics.json','w'), indent=2)
    print(f'arm={a.arm} seed={a.seed} val={best[0]:.3f} rmse={res["rmse"]:.3f} late={res["late"]:.3f} ' +
          ' '.join(f'{k}:rmse={v["rmse"]:.2f},bias={v["bias"]:+.2f}' for k, v in per.items()))
if __name__ == '__main__': main()
