#!/bin/bash
cat > /home/yndx/zxk/bearing-rul-pi/scripts/dsh_r3_four_arms.py <<'PYEOF'
"""Four-arm time-feature experiment: cycle / none / oracle(T_true) / est(T_hat from PCA health index)."""
import argparse, json, sys, time, hashlib
import numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS, SensorMinMaxScaler
from candidates.model import CandidateModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import run_candidate as rc

COLS = list(SELECTED_SENSOR_COLUMNS); CYC = len(COLS); OPS = ['op_1','op_2','op_3']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['cycle','none','oracle','est'], required=True)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
    life = df.groupby('unit')['cycle'].max(); units = np.sort(df.unit.unique())
    rng = np.random.default_rng(a.seed); perm = rng.permutation(len(units)); nval = max(8, len(units)//5)
    val_units = set(units[perm[:nval]].tolist()); tr_units = set(units[perm[nval:]].tolist())
    tr_df = df[df.unit.isin(tr_units)].reset_index(drop=True)
    va_df = df[df.unit.isin(val_units)].reset_index(drop=True)
    sc = SensorMinMaxScaler(COLS).fit(tr_df[COLS])
    km = KMeans(n_clusters=6, n_init=10, random_state=2026).fit(df[OPS].to_numpy(float))
    df['cl'] = km.predict(df[OPS].to_numpy(float))
    tr_mask = df.unit.isin(tr_units)
    stats = {c: (g[COLS].mean().to_numpy(), g[COLS].std().to_numpy() + 1e-9) for c, g in df[tr_mask].groupby('cl')}
    def zc(frame):
        f = frame.reset_index(drop=True); out = np.empty((len(f), len(COLS)))
        for c, g in f.groupby('cl'):
            mu, sd = stats.get(c, (np.zeros(len(COLS)), np.ones(len(COLS))))
            out[g.index.to_numpy()] = (g[COLS].to_numpy(float) - mu) / sd
        return out
    pca = PCA(n_components=1, random_state=0).fit(zc(df[tr_mask]))
    w = pca.components_[0]
    L = float(np.mean([(zc(df[df.unit == u]) @ w)[-1] for u in sorted(tr_units)]))
    def that(frame, upto):
        h = (zc(frame) @ w)[:upto]; t = np.arange(len(h), dtype=float)
        sl = float(np.polyfit(t, h, 1)[0])
        return float(np.clip(upto + (L - h[-1])/sl, 1.2*upto, 400.0)) if sl > 1e-9 else np.nan
    pool = float(np.median([life[u] for u in tr_units]))
    Xs, Ys, Es, Ts, THs = [], [], [], [], []
    for u in sorted(tr_units):
        g = tr_df[tr_df.unit == u].reset_index(drop=True)
        x, y, e = rc.build_windows(g, sc.transform(g[COLS]), 50, 1, 125.0)
        if len(x) == 0: continue
        n = len(x); cs = g['cycle'].to_numpy()[49:]
        T = float(life[u]); ths = []
        for c in cs:
            if a.arm != 'est': ths.append(T); continue
            gt = df[df.unit == u]; rr = that(gt, int(c)); ths.append(rr if np.isfinite(rr) else pool)
        Xs.append(x); Ys.append(y); Es.append(e); Ts.append(np.full(n, T)); THs.append(np.array(ths, float))
    Xtr = np.concatenate(Xs); Ytr = np.concatenate(Ys); Etr = np.concatenate(Es); Ttr = np.concatenate(Ts); THtr = np.concatenate(THs)
    v = va_df.reset_index(drop=True)
    Xva, Yva, Eva, uva = rc.build_windows_matched(v, sc.transform(v[COLS]), 50, 125.0)
    Tva = np.array([float(life[u]) for u in uva]); THva = np.array([that(df[df.unit == u], int(life[u])) for u in uva])
    THva = np.where(np.isfinite(THva), THva, pool)
    raw_tr = Etr[:, CYC] * rc.CYCLE_SCALE; raw_va = Eva[:, CYC] * rc.CYCLE_SCALE
    if a.arm == 'none':
        Etr[:, CYC] = 0.0; Eva[:, CYC] = 0.0
    elif a.arm == 'oracle':
        Etr[:, CYC] = np.clip(raw_tr/Ttr, 0, 1); Eva[:, CYC] = np.clip(raw_va/Tva, 0, 1)
    elif a.arm == 'est':
        Etr[:, CYC] = np.clip(raw_tr/THtr, 0, 1); Eva[:, CYC] = np.clip(raw_va/THva, 0, 1)
    print(f'arm={a.arm} train={len(Xtr)} val={len(Xva)} feat=[{Etr[:,CYC].min():.3f},{Etr[:,CYC].max():.3f}]')
    model = CandidateModel(num_sensors=len(COLS), window_size=50, d_model=64, num_heads=4, num_layers=1,
                           dropout=0.15, use_baseline=True, use_ema_features=True, use_cycle=True,
                           use_graph=False, use_dual_head=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); lossf = nn.MSELoss()
    t_ = lambda z: torch.tensor(z, device=dev)
    Xt, Yt, Et = t_(Xtr), t_(Ytr).unsqueeze(1), t_(Etr); Xv, Yv, Ev = t_(Xva), t_(Yva).unsqueeze(1), t_(Eva)
    best = (1e9, None, 0); n = len(Xt); t0 = time.time()
    for ep in range(1, a.epochs+1):
        model.train(); idx = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, 64):
            b = idx[i:i+64]; pred = model(Xt[b], extras=Et[b])['prediction']; d = pred - Yt[b]
            late = torch.where(d > 0, 1.8*d.square(), d.square())
            loss = lossf(pred, Yt[b]) + 0.5*late.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += float(loss)*len(b)
        model.eval()
        with torch.no_grad(): pv = model(Xv, extras=Ev)['prediction']
        r = float(torch.sqrt(torch.mean((pv - Yv)**2)))
        if r < best[0]: best = (r, {k: v_.detach().clone() for k, v_ in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): pv = model(Xv, extras=Ev)['prediction'].cpu().numpy().reshape(-1)
    pv = np.clip(pv, 0, 125.0); e = pv - Yva
    nz = np.where(e < 0, np.exp(-e/13)-1, np.exp(e/10)-1)
    q1, q2 = np.quantile(Tva, [1/3, 2/3]); bands = {'short': Tva <= q1, 'mid': (Tva > q1) & (Tva <= q2), 'long': Tva > q2}
    per = {k: dict(n=int(m.sum()), rmse=float(np.sqrt((e[m]**2).mean())), bias=float(e[m].mean()), late=float((e[m]>0).mean())) for k, m in bands.items() if m.any()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'pred.npz', y_true=Yva, y_pred=pv, life=Tva, that=THva, arm=a.arm, seed=a.seed)
    json.dump(dict(arm=a.arm, seed=a.seed, best_val_rmse=best[0], best_epoch=best[2], epochs_ran=a.epochs,
                   n_train=int(len(Xtr)), n_val=int(len(Xva)), rmse=float(np.sqrt((e**2).mean())),
                   mae=float(np.abs(e).mean()), nasa_mean=float(nz.mean()), late=float((e>0).mean()),
                   severe_e10=float((e>10).mean()), by_lifespan=per, seconds=time.time()-t0,
                   code_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest()), open(out/'metrics.json','w'), indent=2)
    print(f'  val={best[0]:.3f} rmse={float(np.sqrt((e**2).mean())):.3f} late={float((e>0).mean()):.3f} ' +
          ' '.join(f'{k}:{v_["rmse"]:.1f}/{v_["bias"]:+.1f}' for k, v_ in per.items()))
if __name__ == '__main__': main()
PYEOF
cd /home/yndx/zxk/bearing-rul-pi && export LD_LIBRARY_PATH=/home/yndx/miniconda3/envs/a40_env/lib:$LD_LIBRARY_PATH && /home/yndx/miniconda3/envs/a40_env/bin/python scripts/dsh_r3_four_arms.py --arm est --seed 2027 --epochs 1 --out experiments/r3_probe/est_s2027 2>&1 | tail -4
