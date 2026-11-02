"""Probe B3: where does the RUL error actually come from?

B1 and B2 say the multi-condition story is over on FD002/FD004: the condition is
a universal per-channel affine nuisance with an observable index, invertible in
closed form.  Probe B3 asks the next question, which decides what a method could
possibly be worth: once the nuisance is gone, what is left?

Steps, all on training engines with engine-level 5-fold so nothing is fitted on
the rows it is scored on:

1. Invert the condition map analytically, row by row.
2. Measure the intrinsic dimension of the aligned signal (PCA on rows, not on
   means: the mean-curve rank says nothing about the row-level problem).
3. Measure unit-to-unit variability of the degradation *rate* in the aligned
   one-dimensional coordinate.
4. Run a ladder of RUL estimators that differ only in how much per-engine
   information they are allowed, and report RMSE at cutoffs sampled from the
   official test RUL distribution:

   pop          inverse of the population health curve                 no adaptation
   pop+rate     population curve, per-engine slope from the prefix       one parameter
   oracle-curve the engine's own full curve                            upper bound
   oracle-noise perfect curve, but h observed with its own noise        noise floor

The gap between ``pop`` and ``oracle-curve`` is the total headroom any
adaptation method can win.  The gap between ``oracle-curve`` and ``oracle-noise``
is what no method can win, because the health reading itself is noisy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import data as D
import probe as B1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "bline"
GRID = np.arange(1, 200, 1.0)
FOLDS = 5
CUTOFFS = 400


def align_rows(x, lab, maps, mu, sd):
    """Apply the per-condition affine inverse row by row."""
    z = (x - mu) / sd
    out = np.empty_like(z)
    for c, (a, b) in maps.items():
        m = lab == c
        if m.any():
            out[m] = (z[m] - a) / b
    return out


def fit_maps(xs, rul, lab, unit, engines, grid):
    base = B1.pooled_curve(xs, rul, lab, unit, engines, grid)
    ok = np.isfinite(base).all(1)
    maps = {}
    for c in range(6):
        Cc, _ = B1.condition_curve(xs, rul, lab, unit, engines, c, grid)
        keep = ok & np.isfinite(Cc).all(1)
        a, b = B1.fit_diag(Cc[keep].T, base[keep].T)
        # a channel that does not move with the condition gets gain 1, not 0: a
        # near-zero gain would amplify row noise without bound on inversion
        b = np.where(np.abs(b) < 0.05, 1.0, b)
        maps[c] = (a, b)
    return maps, base


def monotone_fit(t, h):
    """Isotonic regression of h on t (increasing), returned on the input grid."""
    o = np.argsort(t)
    t, h = t[o], h[o]
    # PAVA
    w = np.ones_like(h)
    vals, wts = h.copy(), w.copy()
    idx = []
    for i in range(len(h)):
        vals_i, w_i = vals[i], wts[i]
        j = len(idx) - 1
        while j >= 0 and vals[idx[j]] > vals_i:
            # merge
            vj, wj = vals[idx[j]], wts[idx[j]]
            vals[idx[j]] = (vj * wj + vals_i * w_i) / (wj + w_i)
            wts[idx[j]] = wj + w_i
            vals_i, w_i = vals[idx[j]], wts[idx[j]]
            idx.pop()
            j -= 1
        idx.append(i)
        vals[i], wts[i] = vals_i, w_i
    fitted = np.empty_like(h)
    start = 0
    for i in idx:
        fitted[start:i + 1] = vals[i]
        start = i + 1
    out = np.empty_like(fitted)
    out[o] = fitted
    return out


def main() -> dict:
    res = {}
    for sub in ("FD002", "FD004"):
        tr = D.load(sub, "train")
        sc = D.setting_scale([tr])
        cen = D.fit_conditions([tr], k=6)
        lab = D.assign_conditions(tr, cen, sc)
        mu, sd = tr.x.mean(0), np.where(tr.x.std(0) > 1e-9, tr.x.std(0), 1.0)
        xs = (tr.x - mu) / sd
        engines = tr.engines()
        rng = np.random.default_rng(20261103)

        perm = rng.permutation(len(engines))
        folds = np.array_split(perm, FOLDS)
        preds = {k: [] for k in ("pop", "pop_rate", "oracle_curve", "oracle_noise")}
        truth = []
        pca_info = []

        for k in range(FOLDS):
            hold = engines[folds[k]]
            fit = np.setdiff1d(engines, hold)
            maps, base = fit_maps(xs, tr.rul, lab, tr.unit, fit, GRID)
            za = align_rows(tr.x, lab, maps, mu, sd)
            mfit = np.isin(tr.unit, fit)
            # PCA on aligned rows of the fitting engines only
            X = za[mfit]
            Xc = X - X.mean(0)
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            ev = s ** 2 / (s ** 2).sum()
            pca_info.append({"pc1": float(ev[0]), "pc2": float(ev[1]),
                             "pc1_3": float(ev[:3].sum()), "pc1_5": float(ev[:5].sum())})
            v1 = Vt[0]
            # orient the health axis so it grows as RUL shrinks
            if np.corrcoef(za[mfit] @ v1, -tr.rul[mfit])[0, 1] < 0:
                v1 = -v1
            h_all = za @ v1
            # population health curve: median h at each RUL on fitting engines
            grid_r, curve = [], []
            for r in GRID:
                m = mfit & (np.abs(tr.rul - r) < 0.51)
                if m.sum() >= 30:
                    grid_r.append(r)
                    curve.append(np.median(h_all[m]))
            grid_r, curve = np.array(grid_r), np.array(curve)
            if len(grid_r) < 20:
                continue
            keep = np.concatenate([[True], np.diff(curve) > 0])
            grid_r, curve = grid_r[keep], curve[keep]

            # per-engine monotone curve on fitting engines, for the oracle
            for eng in hold:
                idx = tr.index_of(eng)
                idx = idx[np.argsort(tr.cycle[idx])]
                t, h = tr.cycle[idx].astype(float), h_all[idx]
                r = tr.rul[idx]
                # cutoffs mimic the official test protocol: a prefix of at least
                # 30 cycles, evaluated inside the 125-cycle label plateau
                cand = np.flatnonzero((r <= 130) & (np.arange(len(r)) >= 30))
                if len(cand) == 0:
                    continue
                own = monotone_fit(t, h)
                noise_sd = float(np.std(h - own))
                for j in rng.choice(cand, size=min(8, len(cand)), replace=False):
                    j = int(j)
                    h_now = h[j]
                    t_now = t[j]
                    # pop: invert population curve
                    preds["pop"].append(float(np.interp(h_now, curve, grid_r)))
                    # pop+rate: population shape, per-engine slope over the prefix
                    pref = slice(max(0, j - 60), j + 1)
                    tt, hh = t[pref], h[pref]
                    A = np.column_stack([np.ones(len(tt)), tt])
                    coef, *_ = np.linalg.lstsq(A, hh, rcond=None)
                    slope = float(coef[1])                     # health per cycle
                    pop_slope = -float(np.polyfit(grid_r, curve, 1)[0])   # health per cycle
                    pop_rul = float(np.interp(h_now, curve, grid_r))
                    if abs(slope) > 1e-9 and pop_slope > 1e-9:
                        preds["pop_rate"].append(pop_rul / max(slope / pop_slope, 1e-3))
                    else:
                        preds["pop_rate"].append(np.nan)
                    # oracle curve: this engine's own monotone curve
                    preds["oracle_curve"].append(_invert_monotone(own, t, h_now))
                    # oracle noise: same but reading the *fitted* value, not the noisy one
                    preds["oracle_noise"].append(_invert_monotone(own, t, own[j]))
                    truth.append(float(r[j]))

        truth = np.array(truth)
        out = {"subset": sub, "n": int(len(truth)), "pca": pca_info}
        for key, v in preds.items():
            v = np.array(v, float)
            ok = np.isfinite(v)
            if ok.sum() < 10:
                out[key] = None
                continue
            e = v[ok] - truth[ok]
            out[key] = {"rmse": float(np.sqrt((e ** 2).mean())),
                        "mae": float(np.abs(e).mean()),
                        "bias": float(e.mean()), "n": int(ok.sum())}
        res[sub] = out
    return res


def _invert_monotone(curve_vals, t, h_now):
    """RUL at the crossing of a monotone fitted curve with h_now.

    ``curve_vals`` increases with cycle ``t``; h_now is a health level.  RUL is
    the number of cycles from the crossing point to the engine's failure, which
    for a training engine is ``t_end - t_cross``.
    """
    o = np.argsort(curve_vals)
    cv, tt = curve_vals[o], t[o]
    if h_now <= cv[0]:
        return float(t[-1] - tt[0])
    if h_now >= cv[-1]:
        return float(max(t[-1] - tt[-1], 0.0))
    t_cross = float(np.interp(h_now, cv, tt))
    return float(max(t[-1] - t_cross, 0.0))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = main()
    for sub, r in res.items():
        p = np.mean([[q["pc1"], q["pc2"], q["pc1_3"], q["pc1_5"]] for q in r["pca"]], axis=0)
        print(f"\n=== {sub}   cutoffs={r['n']}")
        print(f"  aligned PCA on rows: PC1 {p[0]:.3f}  PC2 {p[1]:.3f}  PC1-3 {p[2]:.3f}  PC1-5 {p[3]:.3f}")
        for key in ("pop", "pop_rate", "oracle_curve", "oracle_noise"):
            v = r[key]
            if v is None:
                print(f"  {key:13s} n/a")
            else:
                print(f"  {key:13s} RMSE {v['rmse']:6.2f}  MAE {v['mae']:6.2f}  bias {v['bias']:+6.2f}")
    (OUT / "probe_b3.json").write_text(json.dumps(res, indent=1))
    print(f"\nwritten {OUT / 'probe_b3.json'}")
