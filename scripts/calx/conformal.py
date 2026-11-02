"""Cross-fitted conformal intervals for RUL, evaluated against the frozen layer.

Protocol: paper/CVPLUS_PROTOCOL.md (frozen 2026-11-02, amendment 1 on R4).

Four rows, same test engines, same target:
  base   frozen 66-parameter censored layer, width tuned to cross-fitted dev PICP90
  cv+    constant-width cross-conformal over the 5 fold models (Barber 2021)
  ncv+   same, score normalised by the frozen layer's own scale head
  cw-cp  ncv+ with covariate-shift weights on the out-of-fold residuals

Coverage is measured twice: randomized PIT (the correct rule for a pmf on an integer
grid) and the integer-endpoint convention the literature reports. The gap between them is
reported, because it is 0.014-0.024 at the 90% level and that is larger than most of the
method differences anyone has claimed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import ndtr

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))
from npforward import forward as np_forward, load_weights  # noqa: E402

MAX_RUL = 125.0
K = int(MAX_RUL) + 1
LEVELS = (0.90, 0.50)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


# ------------------------------------------------------------------ io
def read_csv(path):
    us, ys, ps = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            us.append(int(r.get("unit_id", r.get("unit"))))
            ys.append(float(r["y_true"]))
            ps.append(float(r["y_pred"]))
    o = np.argsort(us)
    return np.array(us)[o], np.array(ys)[o], np.array(ps)[o]


def load_set(root):
    out = {}
    for f in sorted(root.glob("A_s*_f*.csv")):
        seed, fold = f.stem.split("_")[1], f.stem.split("_")[2]
        out[(int(seed[1:]), int(fold[1:]))] = read_csv(f)
    return out


# ------------------------------------------------------------------ distributions
def pmf_censored(mu, sigma):
    Fm = ndtr((np.arange(1, K)[None, :] - np.asarray(mu)[:, None]) / np.asarray(sigma)[:, None])
    p = np.zeros((len(np.atleast_1d(mu)), K))
    p[:, 0] = Fm[:, 0]
    p[:, 1:K - 1] = np.diff(Fm, axis=1)
    p[:, K - 1] = 1.0 - Fm[:, -1]
    p = np.clip(p, 0.0, None)
    return p / p.sum(1, keepdims=True)


def interval_from_pmf(p, level):
    cdf = np.cumsum(p, axis=1)
    a = (1.0 - level) / 2.0
    lo = np.array([int(np.searchsorted(cdf[i], a)) for i in range(len(p))])
    hi = np.array([min(int(np.searchsorted(cdf[i], 1.0 - a)), K - 1) for i in range(len(p))])
    return lo, hi


def cov_randomized_pmf(p, y, level, nrep=200, seed=7):
    """PIT inside the bin, uniform: the definition that does not reward coarse grids."""
    y = np.asarray(y, int)
    cdf = np.cumsum(p, axis=1)
    prev = np.concatenate([np.zeros((len(p), 1)), cdf[:, :-1]], axis=1)
    a = (1.0 - level) / 2.0
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    acc = 0.0
    for _ in range(nrep):
        u = prev[idx, y] + rng.random(len(y)) * p[idx, y]
        acc += float(((u >= a) & (u <= 1.0 - a)).mean())
    return acc / nrep


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def winkler(lo, hi, y, level):
    a = 1.0 - level
    return float(np.mean((hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)))


# ------------------------------------------------------------------ conformal
def wquantile(v, w, q):
    """Smallest v with cumulative weight fraction >= q."""
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    j = int(np.searchsorted(c, q * c[-1] - 1e-12))
    return float(v[min(j, len(v) - 1)])


def conformal_intervals(mu, r_abs, scale, alpha, wts=None):
    """Cross-conformal endpoints for every test point.

    mu     (n_test, K)   fold-model predictions
    r_abs  (n_cal,)      |residual| of calibration engines, in the units of `scale`
    scale  (n_test,)     per-point scale multiplying r_abs (1.0 for the plain score)
    wts    (n_cal,)      optional calibration weights
    """
    n, Kf = mu.shape
    lo = np.empty(n)
    hi = np.empty(n)
    w = np.ones(len(r_abs)) if wts is None else np.asarray(wts, float)
    w = np.repeat(w, Kf)                      # each residual pairs with every fold model
    for i in range(n):
        s = scale[i]
        vlo = (mu[i][:, None] - r_abs[None, :] * s).ravel()
        vhi = (mu[i][:, None] + r_abs[None, :] * s).ravel()
        lo[i] = wquantile(vlo, w, alpha)
        hi[i] = wquantile(vhi, w, 1.0 - alpha)
    return lo, hi


def normalised_scores(r_signed, yhat, sig):
    return np.abs(r_signed) / sig


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments" / "calx" / "conformal.json"))
    ap.add_argument("--seeds", default="2027,2028,2029")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    cal = json.loads((ROOT / "experiments" / "cal" / "calibrators" / "calibrator_A.json").read_text())
    wts0 = load_weights(ROOT / "experiments" / "cal" / "calibrators" / "calibrator_A.json")
    sig_fn = lambda v: np_forward(wts0, np.asarray(v, float), MAX_RUL)[1]

    report = {"protocol": "paper/CVPLUS_PROTOCOL.md", "levels": list(LEVELS), "subsets": {}}

    for subset in SUBSETS:
        dev = load_set(ROOT / "experiments" / "cal" / "dev_preds" / subset)
        tst = load_set(ROOT / "experiments" / "calx" / "fold_test" / subset)
        base = np.load(ROOT / "experiments" / "cal" / "audit" / f"{subset}_A.npz")
        b_unit, b_y, b_p = base["unit"], base["y"].astype(float), base["p_cens"]

        units = tst[(seeds[0], 0)][0]
        assert np.array_equal(units, b_unit), f"{subset}: fold_test unit order != frozen layer"

        # ---- integrity: the 15 fold-model predictions must average to the deployed number
        stack = np.stack([tst[(s, f)][2] for s in seeds for f in range(5)])
        e21 = read_csv(ROOT / "results" / "dsh" / "e21" / "preds" / f"{subset}_A.csv")
        assert np.array_equal(e21[0], units)
        dev_gap = float(np.max(np.abs(stack.mean(0) - e21[2])))

        row = {"n_test": int(len(units)), "n_dev": int(sum(len(v[0]) for v in dev.values())),
               "integrity_max_gap_vs_deployed": dev_gap, "methods": {}}

        # ---- M0 base: frozen layer, randomized and integer coverage
        m0 = {}
        for lv in LEVELS:
            lo, hi = interval_from_pmf(b_p, lv)
            cov_int = float(np.mean((lo <= b_y) & (b_y <= hi)))
            m0[f"{lv:.2f}"] = dict(
                cov=round(cov_randomized_pmf(b_p, b_y, lv), 4),
                cov_int=round(cov_int, 4),
                width=float(np.mean(hi - lo)),
                claimed_mass=float(np.mean([b_p[i, lo[i]:hi[i] + 1].sum() for i in range(len(b_p))])),
                winkler=winkler(lo.astype(float), hi.astype(float), b_y, lv))
        row["methods"]["base"] = m0

        # ---- ORACLE90: the multiplicative width that makes test randomized coverage 0.90
        target = 0.90
        f = lambda s: cov_randomized_pmf(pmf_censored(base["mu"], base["sigma"] * s), b_y, target) - target
        a_, b_ = 0.2, 8.0
        for _ in range(40):
            m = 0.5 * (a_ + b_)
            if f(m) < 0:
                a_ = m
            else:
                b_ = m
        sc = 0.5 * (a_ + b_)
        p_or = pmf_censored(base["mu"], base["sigma"] * sc)
        lo_o, hi_o = interval_from_pmf(p_or, 0.90)
        base_lo, base_hi = interval_from_pmf(b_p, 0.90)
        row["oracle90"] = {"scale": float(sc), "width": float(np.mean(hi_o - lo_o)),
                           "width_ratio_vs_base": float(np.mean(hi_o - lo_o) / np.mean(base_hi - base_lo)),
                           "cov": cov_randomized_pmf(p_or, b_y, 0.90)}

        # ---- per-seed conformal rows
        per = {m: {f"{lv:.2f}": [] for lv in LEVELS} for m in ("cv+", "ncv+", "cw-cp")}
        ess_log = []
        for s in seeds:
            d_units, d_y, d_hat, d_fold = [], [], [], []
            for f in range(5):
                u, yv, pv = dev[(s, f)]
                d_units.append(u); d_y.append(yv); d_hat.append(pv); d_fold.append(np.full(len(u), f))
            d_y = np.concatenate(d_y); d_hat = np.concatenate(d_hat); d_fold = np.concatenate(d_fold)
            r = d_y - d_hat
            sig_d = sig_fn(d_hat)

            mu = np.stack([tst[(s, f)][2] for f in range(5)], axis=1)   # (n_test, 5)
            sig_t = sig_fn(mu.mean(1))

            # covariate-shift weights, 1-D in the ensemble prediction
            kde_t = stats.gaussian_kde(mu.mean(1), bw_method="silverman")
            kde_d = stats.gaussian_kde(d_hat, bw_method="silverman")
            w = np.clip(kde_t(d_hat) / np.maximum(kde_d(d_hat), 1e-12), *np.percentile(
                kde_t(d_hat) / np.maximum(kde_d(d_hat), 1e-12), [1, 99]))
            ess_log.append(float(w.sum() ** 2 / (w ** 2).sum()))

            for lv in LEVELS:
                al = (1.0 - lv) / 2.0
                for name, r_use, scale_use, w_use in (
                        ("cv+", np.abs(r), np.ones(len(units)), None),
                        ("ncv+", np.abs(r) / sig_d, sig_t, None),
                        ("cw-cp", np.abs(r) / sig_d, sig_t, w)):
                    lo, hi = conformal_intervals(mu, r_use, scale_use, al, w_use)
                    n = len(units)
                    k = int(((lo <= b_y) & (b_y <= hi)).sum())
                    per[name][f"{lv:.2f}"].append(dict(
                        cov=k / n, lo_w=wilson(k, n)[0], hi_w=wilson(k, n)[1],
                        width=float(np.mean(hi - lo)),
                        winkler=winkler(lo, hi, b_y, lv)))

        for name in ("cv+", "ncv+", "cw-cp"):
            row["methods"][name] = {
                lv: dict(cov=round(float(np.mean([x["cov"] for x in v])), 4),
                         cov_sd=round(float(np.std([x["cov"] for x in v])), 4),
                         width=round(float(np.mean([x["width"] for x in v])), 2),
                         winkler=round(float(np.mean([x["winkler"] for x in v])), 2),
                         per_seed=[round(x["cov"], 3) for x in v])
                for lv, v in per[name].items()}
        row["weights_ess"] = [round(e, 1) for e in ess_log]
        row["n_dev_per_seed"] = int(len(d_y))
        report["subsets"][subset] = row
        print(f"{subset} done  base90={row['methods']['base']['0.90']['cov']:.3f} "
              f"ncv+90={row['methods']['ncv+']['0.90']['cov']:.3f} "
              f"cw90={row['methods']['cw-cp']['0.90']['cov']:.3f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
