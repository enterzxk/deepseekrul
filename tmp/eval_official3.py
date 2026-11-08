"""Official-test evaluator: condref windows with padding for short engines."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path('/home/user/rulb')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
from dataset import load_cmapss_train, load_cmapss_files, OperatingConditionScaler, SELECTED_SENSOR_COLUMNS
from candidates.model import CandidateModel
from sklearn.cluster import KMeans
import run_candidate as rc

def test_windows(frame, values, window, labels, pop_ref, tau=2.0, max_rul=125.0):
    xs, extras, units = [], [], []
    for uid, group in frame.groupby('unit', sort=True):
        pos = group.index.to_numpy(np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        clusters = np.asarray(labels[pos], dtype=np.int64)
        cycles = group['cycle'].to_numpy(np.int64)
        length = seq.shape[0]; end = length; t = end - 1
        m = int(clusters[t])
        early = np.where((cycles <= 30) & (clusters == m) & (np.arange(length) <= t))[0]
        sample_idx = early[-5:] if early.size > 5 else early
        n = int(sample_idx.size)
        if n > 0:
            xbar = seq[sample_idx].mean(axis=0)
            std_ref = float(seq[sample_idx].std(axis=0).mean()) if n > 1 else 0.0
            age = float(cycles[sample_idx].mean())
        else:
            xbar = np.zeros(seq.shape[1], np.float32); std_ref = 0.0; age = 30.0
        ref = (n/(n+tau))*xbar + (tau/(n+tau))*pop_ref[m]
        wv = seq[max(0, end-window):end]
        if wv.shape[0] < window:
            wv = np.concatenate([np.repeat(wv[:1], window-wv.shape[0], axis=0), wv], axis=0)
        deviation = float(np.abs(wv - ref).mean())
        ema = seq[0].copy()
        for row in seq[:end]: ema = 0.5*ema + 0.5*row
        jitter = float(np.abs(seq[t] - ema).mean())
        ev = [min(1.0, n/5.0), 1.0 if n >= 1 else 0.0, float(np.exp(-age/30.0)),
              float(np.exp(-deviation)), 1.0/(1.0+std_ref), jitter]
        xs.append(np.ascontiguousarray(wv))
        extras.append(np.concatenate([ref, [min(cycles[t]/250.0, 1.0)], ev]))
        units.append(int(uid))
    return np.stack(xs).astype(np.float32), np.stack(extras).astype(np.float32), units

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True); ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--window', type=int, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--variant', default='PE'); ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=64); ap.add_argument('--split-seed', type=int, default=7)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    raw = load_cmapss_train(ROOT/'data'/'CMAPSS', a.subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(a.split_seed + 20261020)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(units.size*0.2)))
    start = a.fold*n_val
    val_units = sorted(int(x) for x in shuffled[start:start+n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit','cycle']).reset_index(drop=True)
    cols = list(SELECTED_SENSOR_COLUMNS); op = ['op_1','op_2','op_3']
    scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=a.split_seed)
    tr_s = scaler.fit_transform(tr)
    op_tr = tr[op].to_numpy(np.float64)
    kmeans = KMeans(n_clusters=6, random_state=a.split_seed, n_init=10).fit(op_tr)
    labels_train = kmeans.labels_.astype(np.int64)
    pop_ref = np.zeros((6, tr_s.shape[1]), np.float32)
    early_mask = tr['cycle'].to_numpy(np.int64) <= 15
    for m in range(6):
        sel = early_mask & (labels_train == m)
        pop_ref[m] = tr_s[sel].mean(axis=0) if sel.any() else tr_s.mean(axis=0)
    _, test_frame, test_rul = load_cmapss_files(ROOT/'data'/'CMAPSS', a.subset)
    test_frame = test_frame.sort_values(['unit','cycle']).reset_index(drop=True)
    ts = scaler.transform(test_frame)
    labels_test = kmeans.predict(test_frame[op].to_numpy(np.float64)).astype(np.int64)
    x, extras, tunits = test_windows(test_frame, ts, a.window, labels_test, pop_ref)
    om = op_tr.mean(axis=0); os_ = op_tr.std(axis=0); os_ = np.where(os_ > 1e-8, os_, 1.0)
    term = []
    for u in tunits:
        g = test_frame.index[test_frame.unit == u]
        term.append((test_frame.loc[g[-1], op].to_numpy(np.float64) - om)/os_)
    cond = np.stack(term).astype(np.float32)
    y = np.minimum(np.asarray(test_rul, np.float64), 125.0)
    cfg = rc.CONFIGS[a.variant]
    model = CandidateModel(num_sensors=14, window_size=a.window, d_model=a.d_model,
        num_layers=a.num_layers, dropout=0.15, use_baseline=cfg['use_baseline'],
        use_ema_features=cfg['use_ema_features'], use_cycle=cfg['use_cycle'],
        use_pe=cfg.get('use_pe', False), use_graph=cfg['use_graph'], use_dual_head=cfg['use_dual_head'])
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu')); model.cuda().eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            out = model(torch.from_numpy(x[i:i+256]).cuda(), cond=torch.from_numpy(cond[i:i+256]).cuda(),
                        extras=torch.from_numpy(extras[i:i+256]).cuda())['prediction'].reshape(-1)
            preds.append(out.cpu().numpy())
    p = np.concatenate(preds)
    pd.DataFrame({'unit': tunits, 'y_true': y, 'y_pred': p}).to_csv(a.out, index=False)
    print('RMSE %.4f  n=%d  %s f%d w%d' % (float(np.sqrt(np.mean((p-y)**2))), len(p), a.subset, a.fold, a.window), flush=True)

if __name__ == '__main__':
    main()