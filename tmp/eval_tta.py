"""Official-test evaluator with multi-window test-time averaging (TTA)."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path('/home/user/rulb')
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from dataset import load_cmapss_train, load_cmapss_files, OperatingConditionScaler, SELECTED_SENSOR_COLUMNS
from candidates.dspt import build, MAX_SPAN
from sklearn.cluster import KMeans
CYCLE_SCALE = 250.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', required=True); ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--layers', type=int, default=2); ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--split-seed', type=int, default=7)
    ap.add_argument('--tta', type=int, default=1)
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    raw = load_cmapss_train(ROOT/'data'/'CMAPSS', a.subset)
    units = np.sort(raw['unit'].unique())
    rng = np.random.default_rng(a.split_seed + 20261020)
    sh = rng.permutation(units)
    n_val = max(1, int(round(units.size*0.2)))
    val_units = sorted(int(x) for x in sh[a.fold*n_val:(a.fold+1)*n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(['unit','cycle']).reset_index(drop=True)
    cols = list(SELECTED_SENSOR_COLUMNS); op = ['op_1','op_2','op_3']
    scaler = OperatingConditionScaler(cols, n_clusters=6, random_state=a.split_seed)
    tr_s = scaler.fit_transform(tr)
    km = KMeans(n_clusters=6, random_state=a.split_seed, n_init=10).fit(tr[op].to_numpy(np.float64))
    lt = km.labels_.astype(np.int64)
    pop = np.zeros((6, tr_s.shape[1]), np.float32)
    em = tr['cycle'].to_numpy(np.int64) <= 15
    for k in range(6):
        sel = em & (lt == k)
        pop[k] = tr_s[sel].mean(axis=0) if sel.any() else tr_s.mean(axis=0)
    _, tf, rul = load_cmapss_files(ROOT/'data'/'CMAPSS', a.subset)
    tf = tf.sort_values(['unit','cycle']).reset_index(drop=True)
    ts = scaler.transform(tf)
    lv = km.predict(tf[op].to_numpy(np.float64)).astype(np.int64)
    xs, bs, cs, ms, order = [], [], [], [], []
    for uid, g in tf.groupby('unit', sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(ts[pos], dtype=np.float32)
        cl = np.asarray(lv[pos], dtype=np.int64)
        cyc = g['cycle'].to_numpy(np.int64)
        T = len(seq) - 1
        for j in range(min(a.tta, T+1)):
            t = T - j
            m = min(t+1, MAX_SPAN)
            w = seq[t+1-m:t+1]; pad = MAX_SPAN - m
            x = np.zeros((MAX_SPAN, seq.shape[1]), np.float32); x[pad:] = w
            mk = np.zeros(MAX_SPAN, np.float32); mk[pad:] = 1.0
            k = int(cl[t])
            early = np.where((cyc <= 30) & (cl == k) & (np.arange(len(seq)) <= t))[0]
            si = early[-5:] if early.size > 5 else early
            n = int(si.size)
            xbar = seq[si].mean(axis=0) if n > 0 else np.zeros(seq.shape[1], np.float32)
            ref = (n/(n+2))*xbar + (2/(n+2))*pop[k]
            xs.append(x); ms.append(mk); bs.append(ref.astype(np.float32))
            cs.append(min(cyc[t]/CYCLE_SCALE, 1.0)); order.append(int(uid))
    X = np.stack(xs); MK = np.stack(ms); BS = np.stack(bs); CS = np.asarray(cs, np.float32)
    net = build(14, a.d_model, a.layers, 0.2)
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True)); net.cuda().eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            o = net(torch.from_numpy(X[i:i+256]).cuda(), mask=torch.from_numpy(MK[i:i+256]).cuda(),
                    baseline=torch.from_numpy(BS[i:i+256]).cuda(), cycle=torch.from_numpy(CS[i:i+256]).cuda())
            ps.append(o['prediction'].reshape(-1).cpu().numpy())
    p = np.concatenate(ps)
    df = pd.DataFrame({'unit': order, 'y_pred': p})
    agg = df.groupby('unit', sort=True).y_pred.mean()
    y = np.minimum(np.asarray(rul, np.float64), 125.0)
    pd.DataFrame({'unit': agg.index.to_numpy(), 'y_true': y, 'y_pred': agg.to_numpy()}).to_csv(a.out, index=False)
    print('TTA%d RMSE %.4f n=%d %s f%d' % (a.tta, float(np.sqrt(np.mean((agg.to_numpy()-y)**2))), len(agg), a.subset, a.fold), flush=True)

if __name__ == '__main__': main()