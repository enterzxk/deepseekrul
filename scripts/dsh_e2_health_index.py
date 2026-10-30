"""E2 diagnostic: does a monotone health index make the prefix lifespan estimator usable?

Frozen design: paper/E2_PREREGISTRATION.md (2026-10-29).
CPU only, no training, no test files.

Variants
  H0  cluster-z -> PCA1 (sign aligned on training rows)
  H1  H0 + per-engine EMA(alpha=0.3)
  H2  H1 + per-engine isotonic projection
  H3  not implemented in this run (see the amendment log in the preregistration)

Estimator (identical to R3 so that any change is attributable to h):
  T_hat = c + (L_s - h(c)) / slope(1..c), clipped to [1.2c, 400], fallback to the
  training median lifespan when the slope is non-positive.
"""
import argparse, json, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'src')
from dataset import load_cmapss_train, SELECTED_SENSOR_COLUMNS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

COLS = list(SELECTED_SENSOR_COLUMNS); OPS = ['op_1', 'op_2', 'op_3']
CUTS = [60, 100, 140]


def cluster_z(frame, stats, cl_col='cl'):
    f = frame.reset_index(drop=True)
    out = np.empty((len(f), len(COLS)))
    for c, g in f.groupby(cl_col):
        mu, sd = stats.get(c, (np.zeros(len(COLS)), np.ones(len(COLS))))
        out[g.index.to_numpy()] = (g[COLS].to_numpy(float) - mu) / sd
    return out


def ema(x, alpha=0.3):
    out = np.empty_like(x); s = x[0]
    for i, v in enumerate(x):
        s = alpha * v + (1 - alpha) * s; out[i] = s
    return out


def mono_score(x):
    d = np.diff(x)
    return float(np.mean(d >= 0))


def fit_variant(h, kind):
    if kind == 'H0':
        return h
    if kind == 'H1':
        return ema(h)
    if kind == 'H2':
        return IsotonicRegression(increasing=True).fit_transform(np.arange(len(h)), ema(h))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[2027, 2028, 2029])
    ap.add_argument('--out', default='results/dsh/e2/E2_health.json')
    a = ap.parse_args()
    df = load_cmapss_train(Path('data/CMAPSS'), 'FD002')
    life = df.groupby('unit')['cycle'].max()
    units = np.sort(df.unit.unique())
    rows, summary = [], {}

    for seed in a.seeds:
        rng = np.random.default_rng(seed); perm = rng.permutation(len(units))
        nval = max(8, len(units) // 5)
        val_units = set(units[perm[:nval]].tolist()); tr_units = set(units[perm[nval:]].tolist())
        tr_mask = df.unit.isin(tr_units)
        # unsupervised parts fitted on training engines only
        km = KMeans(n_clusters=6, n_init=10, random_state=2026).fit(df.loc[tr_mask, OPS].to_numpy(float))
        df['cl'] = km.predict(df[OPS].to_numpy(float))
        stats = {c: (g[COLS].mean().to_numpy(), g[COLS].std().to_numpy() + 1e-9)
                 for c, g in df[tr_mask].groupby('cl')}
        Ztr = cluster_z(df[tr_mask], stats)
        pca = PCA(n_components=1, random_state=0).fit(Ztr)
        w = pca.components_[0]
        h_tr = Ztr @ w
        cyc = df.loc[tr_mask, 'cycle'].to_numpy(float)
        if np.corrcoef(h_tr, cyc)[0, 1] < 0:
            w = -w; h_tr = -h_tr

        H = {k: {} for k in ['H0', 'H1', 'H2']}
        for u in units:
            g = df[df.unit == u]
            h = cluster_z(g, stats) @ w
            for k in H:
                H[k][u] = fit_variant(h, k)

        # degradation-rate class: the prefix slope at the same cutoff, with tertile
        # thresholds taken from training engines (deployable: no future information)
        pool = float(np.median([life[u] for u in tr_units]))
        for k in H:
            for c in CUTS:
                sl_tr = {}
                for u in tr_units:
                    h = H[k][u]
                    if len(h) <= c:
                        continue
                    sl_tr[u] = float(np.polyfit(np.arange(c, dtype=float), h[:c], 1)[0])
                q1c, q2c = np.quantile(list(sl_tr.values()), [1 / 3, 2 / 3])
                cls_tr = {u: ('slow' if v <= q1c else ('mid' if v <= q2c else 'fast')) for u, v in sl_tr.items()}
                L = {t: float(np.mean([H[k][u][-1] for u in cls_tr if cls_tr[u] == t]))
                     for t in ('slow', 'mid', 'fast') if any(v == t for v in cls_tr.values())}
                for u in sorted(val_units):
                    T = float(life[u]); h = H[k][u]
                    if T <= c + 5:
                        continue
                    seg = h[:c]; t = np.arange(c, dtype=float)
                    sl = float(np.polyfit(t, seg, 1)[0])
                    cls = 'slow' if sl <= q1c else ('mid' if sl <= q2c else 'fast')
                    if sl > 1e-9 and cls in L:
                        th = float(np.clip(c + (L[cls] - seg[-1]) / sl, 1.2 * c, 400.0)); fb = 0
                    else:
                        th = pool; fb = 1
                    rows.append(dict(seed=seed, variant=k, unit=int(u), lifeT=T, c=c, that=th, fb=fb,
                                     mono=mono_score(h[:c]),
                                     spear=float(spearmanr(h[:c], t).statistic)))
    r = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(str(Path(a.out).with_suffix('.csv')), index=False)
    for k, g in r.groupby('variant'):
        ok = g[g['fb'] == 0]
        summary[k] = dict(n=int(len(g)), fallback=float(g['fb'].mean()),
                          corr_all=float(np.corrcoef(g['that'], g['lifeT'])[0, 1]),
                          corr_no_fb=float(np.corrcoef(ok['that'], ok['lifeT'])[0, 1]) if len(ok) > 2 else None,
                          median_ratio=float(np.median(g['that'] / g['lifeT'])),
                          iqr=[float(np.percentile(g['that'] / g['lifeT'], 25)), float(np.percentile(g['that'] / g['lifeT'], 75))],
                          mono=float(g['mono'].mean()), spearman=float(np.nanmedian(g['spear'])))
    per_seed = {}
    for s, gs in r.groupby('seed'):
        per_seed[str(int(s))] = {}
        for k, g in gs.groupby('variant'):
            ok = g[g['fb'] == 0]
            per_seed[str(int(s))][k] = dict(
                corr=float(np.corrcoef(g['that'], g['lifeT'])[0, 1]),
                corr_no_fb=float(np.corrcoef(ok['that'], ok['lifeT'])[0, 1]) if len(ok) > 2 else None,
                median_ratio=float(np.median(g['that'] / g['lifeT'])),
                fallback=float(g['fb'].mean()))
    out = dict(prereg='paper/E2_PREREGISTRATION.md', cuts=CUTS, seeds=a.seeds,
               summary=summary, per_seed=per_seed, rows=int(len(r)))
    # per-seed detail kept in the CSV; the JSON carries the pooled summary
    json.dump(out, open(a.out, 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print('\nrows:', len(r), '->', a.out)


if __name__ == '__main__':
    main()
