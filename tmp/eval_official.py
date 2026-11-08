"""Official-test evaluator: rebuild the fold scaler, predict test_FDxxx, dump per-model predictions."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path('/home/user/rulb')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
from dataset import load_cmapss_train, load_cmapss_files, OperatingConditionScaler, SELECTED_SENSOR_COLUMNS
from candidates.model import CandidateModel
import run_candidate as rc

def fold_frames(subset, split_seed, fold):
    raw = load_cmapss_train(ROOT/'data'/'CMAPSS', subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(split_seed + 20261020)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(units.size * 0.2)))
    start = fold * n_val
    val_units = sorted(int(x) for x in shuffled[start:start+n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit','cycle']).reset_index(drop=True)
    return tr, train_units, val_units

def build_test(subset, window, scaler, op_mean, op_std):
    _, test_frame, test_rul = load_cmapss_files(ROOT/'data'/'CMAPSS', subset)
    test_frame = test_frame.sort_values(['unit','cycle']).reset_index(drop=True)
    cols = list(SELECTED_SENSOR_COLUMNS)
    sensors = scaler.transform(test_frame)
    cond = ((test_frame[['op_1','op_2','op_3']].to_numpy(np.float64) - op_mean)/op_std).astype(np.float32)
    xs, ys, exs, units = [], [], [], []
    for uid, rul in zip(sorted(int(x) for x in test_frame['unit'].unique()), test_rul):
        g = test_frame[test_frame.unit == uid]
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(sensors[pos], dtype=np.float32)
        # edge-padded last window, matching _last_window_with_edge_padding
        if seq.shape[0] >= window: w = seq[-window:]
        else: w = np.concatenate([np.repeat(seq[:1], window-seq.shape[0], axis=0), seq], axis=0)
        early = seq[:min(5, seq.shape[0])]
        baseline = early.mean(axis=0) if early.shape[0] > 0 else np.zeros(seq.shape[1], np.float32)
        cyc = float(g['cycle'].iloc[-1])
        ref_count = float(min(early.shape[0],5))/5.0
        ref_std = float(early.std(axis=0).mean()) if early.shape[0] > 1 else 0.0
        dev = float(np.abs(seq[-min(50,seq.shape[0]):] - baseline).mean())
        ema = seq[0].copy()
        for row in seq: ema = 0.5*ema + 0.5*row
        jitter = float(np.abs(seq[-1] - ema).mean())
        ev = [ref_count, 1.0 if early.shape[0]>=1 else 0.0, float(np.exp(-2.0/30.0)),
              float(np.exp(-dev)), 1.0/(1.0+ref_std), jitter]
        extras = np.concatenate([baseline, [min(cyc/250.0,1.0)], ev]).astype(np.float32)
        xs.append(w); ys.append(min(float(rul),125.0)); exs.append(extras); units.append(uid)
    return np.stack(xs), np.array(ys, np.float32), np.stack(exs), np.array(units), cond

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True); ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--window', type=int, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--variant', default='PE'); ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--d-model', type=int, default=64); ap.add_argument('--split-seed', type=int, default=7)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    tr, train_units, val_units = fold_frames(a.subset, a.split_seed, a.fold)
    cols = list(SELECTED_SENSOR_COLUMNS); op = ['op_1','op_2','op_3']
    op_mean = tr[op].to_numpy(np.float64).mean(axis=0); op_std = tr[op].to_numpy(np.float64).std(axis=0)
    op_std = np.where(op_std > 1e-8, op_std, 1.0)
    scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=a.split_seed)
    scaler.fit(tr)
    x, y, extras, units, cond = build_test(a.subset, a.window, scaler, op_mean, op_std)
    cfg = rc.CONFIGS[a.variant]
    model = CandidateModel(num_sensors=14, window_size=a.window, d_model=a.d_model,
        num_layers=a.num_layers, dropout=0.15, use_baseline=cfg['use_baseline'],
        use_ema_features=cfg['use_ema_features'], use_cycle=cfg['use_cycle'],
        use_pe=cfg.get('use_pe', False), use_graph=cfg['use_graph'], use_dual_head=cfg['use_dual_head'])
    sd = torch.load(a.ckpt, map_location='cpu')
    model.load_state_dict(sd); model.cuda().eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            xb = torch.from_numpy(x[i:i+256]).cuda()
            cb = torch.from_numpy(cond[i:i+256]).cuda()
            eb = torch.from_numpy(extras[i:i+256]).cuda()
            out = model(xb, cond=cb, extras=eb)['prediction'].reshape(-1)
            preds.append(out.cpu().numpy())
    p = np.concatenate(preds)
    df = pd.DataFrame({'unit': units, 'y_true': y, 'y_pred': p})
    df.to_csv(a.out, index=False)
    rmse = float(np.sqrt(np.mean((p-y)**2)))
    print('RMSE %.4f  n=%d  %s f%d w%d s%d' % (rmse, len(p), a.subset, a.fold, a.window, a.seed), flush=True)

if __name__ == '__main__':
    main()