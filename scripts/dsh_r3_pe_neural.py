"""R3 neural time-arm experiment v2: exact window replication + raw cycle, arms cycle/zero/bounded."""
import argparse, json, sys, time, hashlib
import numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS, SensorMinMaxScaler
from candidates.model import CandidateModel
import run_candidate as rc

CYCLE_SCALE = rc.CYCLE_SCALE
print('CYCLE_SCALE =', CYCLE_SCALE)

def build_ex(frame, values, window, max_rul, cutoff_seed=2026):
    rng = np.random.default_rng(cutoff_seed)
    xs, ys, ex, raw_c, units = [], [], [], [], []
    for _, group in frame.groupby('unit', sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        length = seq.shape[0]
        if length < window + 5: continue
        end = int(rng.integers(40, max(41, int(0.95*length)) + 1)); end = min(max(end, window), length)
        early = seq[:min(5, length)]; baseline = early.mean(axis=0)
        ref_std = float(early.std(axis=0).mean()) if early.shape[0] > 1 else 0.0
        ref_count = min(1.0, early.shape[0]/5.0)
        window_vals = seq[end-window:end]
        deviation = float(np.abs(window_vals - baseline).mean())
        ema_state = seq[0].copy()
        for row in seq[:end]: ema_state = 0.5*ema_state + 0.5*row
        jitter = float(np.abs(seq[end-1] - ema_state).mean())
        mean_age = float(np.mean(np.arange(1, min(early.shape[0],5)+1)))
        evidence = [ref_count, 1.0, float(np.exp(-mean_age/30.0)), float(np.exp(-deviation)), 1.0/(1.0+ref_std), jitter]
        cycles = group['cycle'].to_numpy(dtype=np.int64)
        rul = rc.build_piecewise_linear_rul(group, max_rul=max_rul)
        xs.append(np.ascontiguousarray(window_vals)); ys.append(float(rul[end-1]))
        ex.append(np.concatenate([baseline, [min(cycles[end-1]/CYCLE_SCALE, 1.0)], evidence]))
        raw_c.append(float(cycles[end-1])); units.append(int(group['unit'].iloc[0]))
    return (np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32),
            np.stack(ex).astype(np.float32), np.asarray(raw_c, dtype=np.float32), units)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['cycle','zero','bounded'], required=True)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--max-rul', type=float, default=125.0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = load_cmapss_train(Path('data/CMAPSS'), 'FD002'); cols = list(SELECTED_SENSOR_COLUMNS)
    units = np.sort(df.unit.unique()); rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(units)); nval = max(8, len(units)//5)
    val_units = set(units[perm[:nval]].tolist()); tr_units = set(units[perm[nval:]].tolist())
    tr_df = df[df.unit.isin(tr_units)].copy().reset_index(drop=True)
    va_df = df[df.unit.isin(val_units)].copy().reset_index(drop=True)
    sc = SensorMinMaxScaler(cols).fit(tr_df[cols])
    tr_vals = sc.transform(tr_df[cols]); va_vals = sc.transform(va_df[cols])
    Xtr, ytr, Etr, Ctr, _ = build_ex(tr_df, tr_vals, a.window, a.max_rul)
    Xva, yva, Eva, Cva, _ = build_ex(va_df, va_vals, a.window, a.max_rul)
    # equivalence check against the official builder
    Xo, yo, Eo, _ = rc.build_windows_matched(tr_df, tr_vals, a.window, a.max_rul, cutoff_seed=2026)
    ok = np.allclose(Xtr, Xo) and np.allclose(ytr, yo) and np.allclose(Etr, Eo)
    print(f'replication matches official builder: {ok}')
    if not ok: raise SystemExit('window replication mismatch -> abort')
    if a.arm == 'zero':
        Etr[:, len(cols)] = 0.0; Eva[:, len(cols)] = 0.0
    elif a.arm == 'bounded':
        Etr[:, len(cols)] = (Ctr/(Ctr+50.0)).astype(np.float32)
        Eva[:, len(cols)] = (Cva/(Cva+50.0)).astype(np.float32)
    print(f'arm={a.arm} raw cycle range=[{Ctr.min():.0f},{Ctr.max():.0f}] feature range=[{Etr[:,len(cols)].min():.3f},{Etr[:,len(cols)].max():.3f}]')
    model = CandidateModel(num_sensors=len(cols), window_size=a.window, d_model=64, num_heads=4,
                           num_layers=1, dropout=0.15, use_baseline=True, use_ema_features=True,
                           use_cycle=True, use_graph=False, use_dual_head=False).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); lossf = nn.MSELoss()
    t = lambda x: torch.tensor(x, device=dev)
    Xtr_t, ytr_t, Etr_t = t(Xtr), t(ytr).unsqueeze(1), t(Etr)
    Xva_t, yva_t, Eva_t = t(Xva), t(yva).unsqueeze(1), t(Eva)
    n = len(Xtr_t); best = (1e9, None, 0); hist = []; t0 = time.time()
    for ep in range(1, a.epochs+1):
        model.train(); idx = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, a.batch):
            b = idx[i:i+a.batch]
            pred = model(Xtr_t[b], extras=Etr_t[b])['prediction']
            delta = pred - ytr_t[b]
            late = torch.where(delta > 0, 1.8*delta.square(), delta.square())
            loss = lossf(pred, ytr_t[b]) + 0.5*late.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss)*len(b)
        model.eval()
        with torch.no_grad(): pv = model(Xva_t, extras=Eva_t)['prediction']
        rmse = float(torch.sqrt(torch.mean((pv-yva_t)**2)))
        hist.append(dict(epoch=ep, train_loss=tot/n, val_rmse=rmse))
        if rmse < best[0]: best = (rmse, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): pv = model(Xva_t, extras=Eva_t)['prediction'].cpu().numpy().reshape(-1)
    pv = np.clip(pv, 0, a.max_rul)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'pred.npz', y_true=yva, y_pred=pv, raw_cycle=Cva, arm=a.arm, seed=a.seed)
    e = pv - yva; nz = np.where(e < 0, np.exp(-e/13)-1, np.exp(e/10)-1)
    json.dump(dict(arm=a.arm, seed=a.seed, best_val_rmse=best[0], best_epoch=best[2], epochs_ran=len(hist),
                   n_train=int(len(Xtr)), n_val=int(len(Xva)), rmse=float(np.sqrt((e**2).mean())),
                   mae=float(np.abs(e).mean()), nasa_mean=float(nz.mean()), late=float((e>0).mean()),
                   severe_e10=float((e>10).mean()), history=hist,
                   code_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(), seconds=time.time()-t0),
              open(out/'metrics.json','w'), indent=2)
    print(f'arm={a.arm} seed={a.seed} val_rmse={best[0]:.3f} rmse={np.sqrt((e**2).mean()):.3f} late={(e>0).mean():.3f} severe10={(e>10).mean():.3f}')
if __name__ == '__main__': main()
