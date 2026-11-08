"""Variable-span observation: same 50 positions, but each window samples the last
`span` cycles (edge-replicated when the history is shorter). Tests whether a longer
observation span, at identical tensor size and compute, improves RUL accuracy."""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent; ROOT = PARENT.parent
sys.path[:0] = [str(PARENT), str(ROOT/'scripts'), str(ROOT/'src'), str(ROOT)]
import run_fold as base
import run_ida as ida
import run_full_fold3 as r3
CYCLE_SCALE = getattr(ida, 'CYCLE_SCALE', 250.0)

def _idx(lo, hi, n):
    avail = hi - lo
    if avail >= n:
        return np.linspace(lo, hi - 1, n).round().astype(int)
    return np.concatenate([np.full(n - avail, lo), np.arange(lo, hi)]).astype(int)

def build_windows_span_meta(frame, values, window, span, max_rul, cluster_labels, pop_ref, tau=2.0):
    xs, ys, extras, meta = [], [], [], []
    for _, group in frame.groupby('unit', sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        clusters = np.asarray(cluster_labels[pos], dtype=np.int64)
        cycles = group['cycle'].to_numpy(dtype=np.int64)
        rul = ida.build_piecewise_linear_rul(group, max_rul=max_rul)
        length = seq.shape[0]
        for end in range(window, length + 1):
            t = end - 1
            m = int(clusters[t])
            early = np.where((cycles <= 30) & (clusters == m) & (np.arange(length) <= t))[0]
            sample_idx = early[-5:] if early.size > 5 else early
            n = int(sample_idx.size)
            if n > 0:
                xbar = seq[sample_idx].mean(axis=0)
                std_ref = float(seq[sample_idx].std(axis=0).mean()) if n > 1 else 0.0
                age = float(cycles[sample_idx].mean())
            else:
                xbar = np.zeros(seq.shape[1], dtype=np.float32)
                std_ref = 0.0
                age = 30.0
            ref = (n / (n + tau)) * xbar + (tau / (n + tau)) * pop_ref[m]
            lo = max(0, end - span)
            window_vals = seq[_idx(lo, end, window)]
            deviation = float(np.abs(window_vals - ref).mean())
            ema_state = seq[0].copy()
            for row in seq[:end]:
                ema_state = 0.5 * ema_state + 0.5 * row
            jitter = float(np.abs(seq[t] - ema_state).mean())
            evidence = [min(1.0, n / 5.0), 1.0 if n >= 1 else 0.0,
                        float(np.exp(-age / 30.0)), float(np.exp(-deviation)),
                        1.0 / (1.0 + std_ref), jitter]
            xs.append(np.ascontiguousarray(window_vals))
            ys.append(float(rul[t]))
            extras.append(np.concatenate([ref, [min(cycles[t] / CYCLE_SCALE, 1.0)], evidence]))
            meta.append((int(group['unit'].iloc[0]), int(cycles[t])))
    return (np.stack(xs).astype(np.float32), np.asarray(ys, np.float32),
            np.stack(extras).astype(np.float32), np.asarray(meta, dtype=np.int64).reshape(-1, 2))

def cond_planes_span(frame, cond_vals, meta, window, span):
    out = np.zeros((len(meta), window, 3), dtype=np.float32)
    units = frame['unit'].to_numpy(); cycles = frame['cycle'].to_numpy(np.int64)
    uniq, first_idx = np.unique(units, return_index=True)
    starts = {int(u): int(fi) for u, fi in zip(uniq, first_idx)}
    for i, (u, end_cycle) in enumerate(meta):
        s = starts[int(u)]
        e = s + int(np.searchsorted(cycles[s:], end_cycle, side='right'))
        lo = max(s, e - span)
        chunk = cond_vals[lo:e]
        out[i] = chunk[_idx(0, len(chunk), window)]
    return out

def prep_fold_span(df, subset, fold, tr_u, va_u, window=50, span=100, max_rul=125.0):
    from sklearn.cluster import KMeans
    tr_df = df[df.unit.isin(tr_u)].sort_values(['unit','cycle']).reset_index(drop=True)
    va_df = df[df.unit.isin(va_u)].sort_values(['unit','cycle']).reset_index(drop=True)
    op = ['op_1','op_2','op_3']
    mean = tr_df[op].to_numpy(np.float64).mean(axis=0); std = tr_df[op].to_numpy(np.float64).std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    cond_t = ((tr_df[op].to_numpy(np.float64) - mean)/std).astype(np.float32)
    cond_v = ((va_df[op].to_numpy(np.float64) - mean)/std).astype(np.float32)
    scaler = ida.OperatingConditionScaler(ida.COLS, n_clusters=6, random_state=7+fold)
    tr_s, va_s = scaler.fit_transform(tr_df), scaler.transform(va_df)
    km = KMeans(n_clusters=6, random_state=7, n_init=10).fit(tr_df[op].to_numpy(np.float64))
    lab_tr = km.labels_.astype(np.int64); lab_va = km.predict(va_df[op].to_numpy(np.float64)).astype(np.int64)
    pop_ref = np.zeros((6, tr_s.shape[1]), dtype=np.float32)
    em = tr_df['cycle'].to_numpy(np.int64) <= 15
    for m in range(6):
        sel = em & (lab_tr == m)
        pop_ref[m] = tr_s[sel].mean(axis=0) if sel.any() else tr_s.mean(axis=0)
    xtr, ytr, etr, mtr = build_windows_span_meta(tr_df, tr_s, window, span, max_rul, lab_tr, pop_ref)
    xva, yva, eva, mva = build_windows_span_meta(va_df, va_s, window, span, max_rul, lab_va, pop_ref)
    ctr = cond_planes_span(tr_df, cond_t, mtr, window, span)
    cva = cond_planes_span(va_df, cond_v, mva, window, span)
    return (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset'); ap.add_argument('fold', type=int)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--span', type=int, default=100)
    ap.add_argument('--arms', default='M0,FULL')
    ap.add_argument('--epochs', type=int, default=80)
    a = ap.parse_args()
    dest = HERE / f'runs_hist_s{a.span}' / a.subset / str(a.fold); dest.mkdir(parents=True, exist_ok=True)
    folds, df = ida.outer_folds(a.subset)
    df = df.sort_values(['unit','cycle']).reset_index(drop=True)
    train, valid = folds[a.fold]
    tables = base.prior_tables(df, a.subset, train, valid, a.fold)
    tr, va = prep_fold_span(df, a.subset, a.fold, train, valid, window=a.window, span=a.span)
    xtr, ytr, etr, mtr = tr[:4]; xva, yva, eva, mva = va[:4]
    lives = df.groupby('unit').cycle.max()
    meta = pd.DataFrame(mva, columns=['unit','end_cycle'])
    meta['short'] = meta.unit.map(lives) <= np.quantile(lives.loc[sorted(train)], .25)
    meta['true_life'] = meta.unit.map(lives)
    for arm in [x for x in a.arms.split(',') if x]:
        net = r3.build_net(arm, a.window)
        ida.set_seed(a.seed); net = net.cuda()
        tx = [xtr, ytr, etr, base.prior_values(mtr, tables['FULL'])]
        vx = [xva, yva, eva, base.prior_values(mva, tables['FULL'])]
        target = dest / f'{arm}_{a.seed}.csv'
        wrote = r3.train_arm(arm, net, tx, vx, target, meta, a.seed, None, None, a.epochs)
        print('completed', a.subset, a.fold, arm, 'span', a.span, 'trained' if wrote else 'cached', flush=True)
    (dest / f'done_{a.seed}.json').write_text(json.dumps(dict(subset=a.subset, fold=a.fold, seed=a.seed, window=a.window, span=a.span, arms=a.arms)))

if __name__ == '__main__':
    main()