"""Why recalibrating an RUL model does not work at C-MAPSS scale.

Two questions, both answered from the frozen artefacts:

  1. Is the calibration set exchangeable with the deployment set?  Measured by
     two-sample tests on the predictions (label-free), on the errors, and on the
     PIT values of the frozen layer.

  2. Given a family of monotone corrections, is there any member that, fitted on
     dev, improves multi-level calibration on test?  Ten families are tried,
     spanning one to four parameters, fitted by three different objectives
     (Cramer-von Mises on the PIT, coverage matching at one level, and
     coverage matching at all four levels).  Each is compared against doing
     nothing and against the oracle that is fitted on the test set itself --
     the last of which separates "the family cannot represent the correction"
     from "the correction does not transfer".

The oracle matters.  If fitting on test cannot reach nominal either, the family
is too weak and the negative result is uninformative.  If it can, the negative
result is about transfer alone, which is the claim this script is built to test.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

K = 126
GRID = np.arange(K)
LEVELS = np.array([0.50, 0.80, 0.90, 0.95])
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ROOT = Path(__file__).resolve().parents[2]
KNOTS = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])


# ---------------------------------------------------------------- pmf helpers
def crps_l1(p, y):
    cdf = np.cumsum(p, axis=1)
    ind = (y[:, None] <= GRID[None, :]).astype(float)
    return float(np.abs(cdf - ind).sum(1).mean())


def pit_of(p, y):
    cdf = np.cumsum(p, axis=1)
    return cdf[np.arange(len(y)), np.asarray(y, int)]


def intervals(p, level):
    cdf = np.cumsum(p, axis=1)
    a = (1.0 - level) / 2.0
    lo = np.array([np.searchsorted(cdf[i], a) for i in range(len(p))])
    hi = np.array([min(np.searchsorted(cdf[i], 1.0 - a), K - 1) for i in range(len(p))])
    return lo, hi


def coverage_profile(p, y):
    y = np.asarray(y, int)
    out = []
    for lv in LEVELS:
        lo, hi = intervals(p, lv)
        out.append(float(((lo <= y) & (y <= hi)).mean()))
    return np.array(out)


def coverage_profile_randomized(p, y, n_rep=60, seed=7):
    """Coverage with a randomised PIT -- the definition that is correct for a pmf on a grid.

    The integer-endpoint definition used in the literature always includes the bin
    containing each quantile, so it is inflated by that bin's mass.  Randomising the
    position inside the bin removes the inflation, and the two definitions are equal
    only when the outcome is continuous.
    """
    y = np.asarray(y, int)
    cdf = np.cumsum(p, axis=1)
    prev = np.concatenate([np.zeros((len(p), 1)), cdf[:, :-1]], axis=1)
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(LEVELS))
    for _ in range(n_rep):
        us = prev[np.arange(len(y)), y] + rng.random(len(y)) * p[np.arange(len(y)), y]
        for j, lv in enumerate(LEVELS):
            a = (1.0 - lv) / 2.0
            acc[j] += float(((us >= a) & (us <= 1.0 - a)).mean())
    return acc / n_rep


def claimed_mass(p, y, level=0.90):
    """Probability the model itself puts inside the interval it reports at `level`."""
    lo, hi = intervals(p, level)
    return float(np.mean([p[i, lo[i]:hi[i] + 1].sum() for i in range(len(p))]))


def mce(p, y):
    """Mean absolute gap between realized coverage and the four nominal levels."""
    return float(np.mean(np.abs(coverage_profile(p, y) - LEVELS)))


# ------------------------------------------------------------- map families
def _interp_map(u, knots, vals):
    return np.interp(u, knots, vals)


def apply_map(p, knots, vals):
    cdf = np.cumsum(p, axis=1)
    F = np.concatenate(
        [np.zeros((len(p), 1)), _interp_map(cdf.reshape(-1), knots, vals).reshape(cdf.shape)],
        axis=1,
    )
    F = np.maximum.accumulate(F, axis=1)
    F = np.clip(F, 0.0, 1.0)
    F[:, -1] = 1.0
    pn = np.diff(F, axis=1)
    pn = np.clip(pn, 1e-12, None)
    return pn / pn.sum(1, keepdims=True)


def power_map(u, kappa):
    """Two-sided power map: kappa < 1 puts more mass in both tails.

    kappa may be a scalar or an array broadcastable to u.
    """
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    k = np.asarray(kappa, float)
    return np.where(u < 0.5, 0.5 * np.power(2 * u, k), 1.0 - 0.5 * np.power(2 * (1 - u), k))


def knot_map_from_theta(theta):
    """Six increments -> monotone map on KNOTS with endpoints pinned at 0 and 1."""
    g = np.clip(np.cumsum(np.abs(theta)), 1e-9, None)
    g = g / g[-1]
    return np.concatenate([[0.0], g])


def pit_cvm(u, weights=None):
    u = np.clip(np.asarray(u, float), 1e-9, 1 - 1e-9)
    if weights is None:
        return float(stats.cramervonmises(u, "uniform").statistic)
    o = np.argsort(u, kind="mergesort")
    us, ws = u[o], np.asarray(weights, float)[o]
    ws = ws / ws.sum()
    F = np.cumsum(ws)
    return float(np.sum(ws * (F - us) ** 2))


# --------------------------------------------------------------- corrections
def fit_scale(p, y, objective, weights=None):
    best, arg = np.inf, 1.0
    for s in np.arange(0.60, 3.51, 0.02):
        q = _rescale(p, s)
        v = objective(pit_of(q, y), weights) if weights is not None else objective(pit_of(q, y))
        if v < best:
            best, arg = v, s
    return arg


def _rescale(p, s):
    """Widen a pmf about its own mean by s, on the probability scale."""
    cdf = np.cumsum(p, axis=1)
    mean = (p * GRID[None, :]).sum(1)
    idx = np.clip(mean[:, None] + s * (GRID[None, :] - mean[:, None]), 0, K - 1)
    F = np.array([np.interp(idx[i], GRID, cdf[i]) for i in range(len(p))])
    F = np.maximum.accumulate(F, axis=1)
    F[:, -1] = 1.0
    pn = np.diff(np.concatenate([np.zeros((len(p), 1)), F], axis=1), axis=1)
    pn = np.clip(pn, 1e-12, None)
    return pn / pn.sum(1, keepdims=True)


def fit_power(p, y, z=None, objective=None):
    if z is None:
        ks = np.arange(0.30, 2.51, 0.02)
        f = lambda k: objective(power_map(pit_of(p, y), k))
        return float(min(ks, key=f))

    def obj(th):
        k = np.exp(th[0] + th[1] * z)
        return objective(power_map(pit_of(p, y), k))

    r = minimize(obj, np.zeros(2), method="Nelder-Mead",
                 options=dict(xatol=1e-5, fatol=1e-10, maxiter=4000))
    return r.x


def apply_power(p, param, z=None):
    cdf = np.cumsum(p, axis=1)
    if np.ndim(param) == 0:
        vals = power_map(cdf, float(param))
    else:
        vals = power_map(cdf, np.exp(param[0] + param[1] * z)[:, None])
    F = np.concatenate([np.zeros((len(p), 1)), vals], axis=1)
    F = np.maximum.accumulate(F, axis=1)
    F[:, -1] = 1.0
    pn = np.diff(F, axis=1)
    pn = np.clip(pn, 1e-12, None)
    return pn / pn.sum(1, keepdims=True)


def fit_knots(p, y, n_param=6, restarts=4):
    def obj(th):
        return float(np.sum((coverage_profile(apply_map(p, KNOTS, knot_map_from_theta(th)), y) - LEVELS) ** 2))

    best = None
    rng = np.random.default_rng(0)
    for i in range(restarts):
        x0 = np.full(n_param, 1.0 / n_param) + (0 if i == 0 else rng.normal(0, 0.05, n_param))
        r = minimize(obj, x0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return knot_map_from_theta(best.x)


def fit_ecdf_map(u):
    """Nonparametric PIT remap: T = ECDF of the calibration PITs.

    By construction T(u_i) is uniform on the calibration sample, so this is the
    strongest member of *any* monotone family and serves as the ceiling. Ties
    (the atom at RUL = 125 lands every capped unit on u = 1) get their average
    rank, which keeps the map monotone.
    """
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    order = np.argsort(u, kind="mergesort")
    ranks = np.empty(len(u), float)
    ranks[order] = np.arange(1, len(u) + 1) / (len(u) + 1.0)
    # average ranks within tied u so the map is a function, not a relation
    uniq, inv = np.unique(u, return_inverse=True)
    avg = np.array([ranks[inv == i].mean() for i in range(len(uniq))])
    avg = np.maximum.accumulate(avg)
    xs = np.concatenate([[0.0], uniq, [1.0]])
    gs = np.concatenate([[0.0], avg, [1.0]])
    return xs, np.maximum.accumulate(gs)


def apply_ecdf(p, xs, gs):
    cdf = np.cumsum(p, axis=1)
    F = np.concatenate([np.zeros((len(p), 1)), np.interp(cdf, xs, gs)], axis=1)
    F = np.maximum.accumulate(F, axis=1)
    F = np.clip(F, 0.0, 1.0)
    F[:, -1] = 1.0
    pn = np.diff(F, axis=1)
    pn = np.clip(pn, 1e-12, None)
    return pn / pn.sum(1, keepdims=True)


def density_ratio(z_dev, z_test):
    """Logistic dev(0)/test(1) on [1, z, z^2]; returns importance weights on dev."""
    z = np.r_[z_dev, z_test]
    X = np.column_stack([np.ones(len(z)), z, z ** 2])
    t = np.r_[np.zeros(len(z_dev)), np.ones(len(z_test))]

    def nll(b):
        s = X @ b
        return float(np.mean(np.logaddexp(0.0, s) - t * s) + 1e-4 * np.sum(b[1:] ** 2))

    b = minimize(nll, np.zeros(3), method="Nelder-Mead").x
    s = b[0] + b[1] * z_dev + b[2] * z_dev ** 2
    w = np.exp(s - s.max())
    return w / w.mean()


# ------------------------------------------------------------------ protocol
def load_subset(subset):
    files = sorted(glob.glob(str(ROOT / "experiments/cal/dev_preds" / subset / "A_s*.csv")))
    rows = []
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append((float(r["y_true"]), float(r["y_pred"])))
    yd = np.array([r[0] for r in rows]).astype(int)
    yhd = np.array([r[1] for r in rows])
    d = np.load(ROOT / "experiments/cal/audit" / f"{subset}_A.npz")
    return yd, yhd, d["y"].astype(int), d["yhat"], float(d["scale_cens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments/cal/audit/protocol_audit.json"))
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "cal"))
    from npforward import forward, pmf_censored  # noqa: E402

    w = json.loads(Path(ROOT / "experiments/cal/calibrators/calibrator_A.json").read_text())
    weights = {
        "w0": np.asarray(w["weights"]["net.0.weight"], float),
        "b0": np.asarray(w["weights"]["net.0.bias"], float),
        "w2": np.asarray(w["weights"]["net.2.weight"], float),
        "b2": np.asarray(w["weights"]["net.2.bias"], float),
    }
    scale = float(w["scale"]["value"])

    exchange, attempts = {}, {}
    for s in SUBSETS:
        yd, yhd, yt, yht, _ = load_subset(s)
        m1, s1 = forward(weights, yhd, 125.0)
        m2, s2 = forward(weights, yht, 125.0)
        p_dev = pmf_censored(m1, s1 * scale, K)
        p_te = pmf_censored(m2, s2 * scale, K)

        zd, zt = yhd / 125.0, yht / 125.0
        exchange[s] = dict(
            n_dev=int(len(yd)), n_test=int(len(yt)),
            mean_rul_dev=float(yd.mean()), mean_rul_test=float(yt.mean()),
            capped_dev=float((yd >= 125).mean()), capped_test=float((yt >= 125).mean()),
            ks_pred_p=float(stats.ks_2samp(yhd, yht).pvalue),
            ks_rul_p=float(stats.ks_2samp(yd, yt).pvalue),
            ks_abserr_p=float(stats.ks_2samp(np.abs(yd - yhd), np.abs(yt - yht)).pvalue),
            cvm_dev=pit_cvm(pit_of(p_dev, yd)),
            cvm_test=pit_cvm(pit_of(p_te, yt)),
            mce_dev=mce(p_dev, yd), mce_test=mce(p_te, yt),
        )
        exchange[s]["cvm_ratio"] = exchange[s]["cvm_dev"] / exchange[s]["cvm_test"]
        exchange[s]["integer_cov"] = [float(v) for v in coverage_profile(p_te, yt)]
        exchange[s]["randomized_cov"] = [float(v) for v in coverage_profile_randomized(p_te, yt)]
        exchange[s]["claimed_mass90"] = claimed_mass(p_te, yt)
        exchange[s]["integer_minus_randomized"] = [
            float(a - b) for a, b in
            zip(exchange[s]["integer_cov"], exchange[s]["randomized_cov"])]

        wts = density_ratio(zd, zt)
        prof = {}
        prof["identity"] = p_te

        sc_cvm = fit_scale(p_dev, yd, pit_cvm)
        prof["scale_cvm"] = _rescale(p_te, sc_cvm)

        sc_w = fit_scale(p_dev, yd, pit_cvm, weights=wts)
        prof["scale_cvm_weighted"] = _rescale(p_te, sc_w)

        kp = fit_power(p_dev, yd, objective=pit_cvm)
        prof["power_cvm"] = apply_power(p_te, kp)

        kp_w = fit_power(p_dev, yd, objective=lambda u: pit_cvm(u, wts))
        prof["power_cvm_weighted"] = apply_power(p_te, kp_w)

        kc = fit_power(p_dev, yd, z=zd, objective=pit_cvm)
        prof["power_cond_cvm"] = apply_power(p_te, kc, zt)

        def cover_obj(u, _lv=0.90):
            q = np.clip(u, 1e-9, 1 - 1e-9)
            return float(((q > 0.05) & (q < 0.95)).mean() - 0.90) ** 2

        sc_cov = fit_scale(p_dev, yd, lambda u: cover_obj(u))
        prof["scale_coverage"] = _rescale(p_te, sc_cov)

        kn = fit_knots(p_dev, yd)
        prof["knots_multilevel"] = apply_map(p_te, KNOTS, kn)

        # oracle: the same families fitted on the deployment set itself
        prof["ORACLE_knots_on_test"] = apply_map(p_te, KNOTS, fit_knots(p_te, yt))
        prof["ORACLE_power_on_test"] = apply_power(p_te, fit_power(p_te, yt, objective=pit_cvm))
        prof["ORACLE_ecdf_on_test"] = apply_ecdf(p_te, *fit_ecdf_map(pit_of(p_te, yt)))
        prof["ecdf_dev_fitted"] = apply_ecdf(p_te, *fit_ecdf_map(pit_of(p_dev, yd)))

        attempts[s] = {
            k: dict(mce=mce(v, yt), cov90=float(coverage_profile(v, yt)[2]),
                    crps=crps_l1(v, yt), cvm=pit_cvm(pit_of(v, yt)))
            for k, v in prof.items()
        }

    names = list(attempts[SUBSETS[0]])
    summary = {}
    for n in names:
        summary[n] = dict(
            mce=float(np.mean([attempts[s][n]["mce"] for s in SUBSETS])),
            cov90=float(np.mean([attempts[s][n]["cov90"] for s in SUBSETS])),
            crps=float(np.mean([attempts[s][n]["crps"] for s in SUBSETS])),
            cvm=float(np.mean([attempts[s][n]["cvm"] for s in SUBSETS])),
        )

    out = dict(exchangeability=exchange, attempts=attempts, summary=summary,
               levels=list(LEVELS))
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"{'subset':6s} {'n_dev':>5s} {'n_te':>5s} {'RUL dev':>8s} {'RUL te':>7s} "
          f"{'KS(pred)':>9s} {'KS(err)':>8s} {'CvM dev':>8s} {'CvM te':>7s} {'ratio':>6s}")
    for s in SUBSETS:
        e = exchange[s]
        print(f"{s:6s} {e['n_dev']:5d} {e['n_test']:5d} {e['mean_rul_dev']:8.1f} "
              f"{e['mean_rul_test']:7.1f} {e['ks_pred_p']:9.4f} {e['ks_abserr_p']:8.4f} "
              f"{e['cvm_dev']:8.3f} {e['cvm_test']:7.3f} {e['cvm_ratio']:6.1f}")
    print()
    print(f"{'subset':6s} {'claimed90':>9s} {'int50':>6s} {'rnd50':>6s} {'infl':>6s} "
          f"{'int90':>6s} {'rnd90':>6s} {'infl':>6s} {'overconf':>9s}")
    for s in SUBSETS:
        e = exchange[s]
        print(f"{s:6s} {e['claimed_mass90']:9.3f} {e['integer_cov'][0]:6.3f} "
              f"{e['randomized_cov'][0]:6.3f} {e['integer_minus_randomized'][0]:+6.3f} "
              f"{e['integer_cov'][2]:6.3f} {e['randomized_cov'][2]:6.3f} "
              f"{e['integer_minus_randomized'][2]:+6.3f} "
              f"{e['claimed_mass90'] - e['randomized_cov'][2]:+9.3f}")
    print()
    print(f"{'correction':24s} {'mean|err|':>10s} {'cov@90':>7s} {'CRPS':>7s} {'CvM':>7s}")
    for n in names:
        m = summary[n]
        tag = "  <- ceiling" if n.startswith("ORACLE") else ""
        print(f"{n:24s} {m['mce']:10.4f} {m['cov90']:7.3f} {m['crps']:7.3f} {m['cvm']:7.3f}{tag}")
    base = summary["identity"]
    fitted = [n for n in names if not n.startswith("ORACLE") and n != "identity"]
    better = [n for n in fitted if summary[n]["mce"] < base["mce"] - 1e-6]
    print(f"\n{len(better)}/{len(fitted)} dev-fitted corrections beat doing nothing "
          f"on test multi-level error")


if __name__ == "__main__":
    main()
