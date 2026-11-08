"""Official-test evaluator (corrected): reuse run_candidate.build_windows_condref."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path('/home/user/rulb')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
from dataset import load_cmapss_train, load_cmapss_files, OperatingConditionScaler, SELECTED_SENSOR_COLUMNS
from candidates.model import CandidateModel
from sklearn.cluster import KMeans
import run_candidate as rc

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
    n_val = max(1, int(round(units.size * 0.2)))
    start = a.fold * n_val
    val_units = sorted(int(x) for x in shuffled[start:start+n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit','cycle']).reset_index(drop=True)
    cols = list(SELECTED_SENSOR_COLUMNS)
    scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=a.split_seed)
    tr_s = scaler.fit_transform(tr)
    op_cols = ['op_1','op_2','op_3']
    kmeans = KMeans(n_clusters=6, random_state=a.split_seed, n_init=10).fit(tr[op_cols].to_numpy(np.float64))
    labels_train = kmeans.labels_.astype(np.int64)
    pop_ref = np.zeros((6, tr_s.shape[1]), dtype=np.float32)
    early_mask = tr['cycle'].to_numpy(np.int64) <= 15
    for m in range(6):
        sel = early_mask & (labels_train == m)
        pop_ref[m] = tr_s[sel].mean(axis=0) if sel.any() else tr_s.mean(axis=0)
    _, test_frame, test_rul = load_cmapss_files(ROOT/'data'/'CMAPSS', a.subset)
    test_frame = test_frame.sort_values(['unit','cycle']).reset_index(drop=True)
    ts = scaler.transform(test_frame)
    labels_test = kmeans.predict(test_frame[op_cols].to_numpy(np.float64)).astype(np.int64)
    x, _, extras, tunits = rc.build_windows_condref(test_frame, ts, a.window, 125.0, labels_test, pop_ref, test_mode=True)
    cond = ((test_frame[op_cols].to_numpy(np.float64) - tr[op_cols].to_numpy(np.float64).mean(axis=0))
            / np.where(tr[op_cols].to_numpy(np.float64).std(axis=0) > 1e-8, tr[op_cols].to_numpy(np.float64).std(axis=0), 1.0)).astype(np.float32)
    # one window per engine, in sorted unit order
    last = {}
    for i,u in enumerate(tunits): last[u] = i
    order = sorted(last)
    keep = [last[u] for u in order]
    x = x[keep]; extras = extras[keep]
    # cond must be the terminal row of each engine
    term = []
    for u in order:
        g = test_frame.index[test_frame.unit == u]
        term.append(cond[g[-1]])
    cond = np.stack(term)
    y = np.minimum(np.asarray(test_rul, dtype=np.float64), 125.0)
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
    pd.DataFrame({'unit': order, 'y_true': y, 'y_pred': p}).to_csv(a.out, index=False)
    print('RMSE %.4f  n=%d  %s f%d w%d' % (float(np.sqrt(np.mean((p-y)**2))), len(p), a.subset, a.fold, a.window), flush=True)

if __name__ == '__main__':
    main()