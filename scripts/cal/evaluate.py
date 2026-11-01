"""Head-to-head evaluation of the calibrated survival layer against standard UQ baselines.

Every method is fitted ONLY on out-of-fold development predictions (never on test
labels), under one of three fitting protocols:

  per_subset  fit on the target fleet's own development engines
  loso        fit on the other fleets' development engines (zero target-fleet data)
  pooled      fit once on every development engine

Reported per subset: RMSE of the predictive mean, CRPS, 90% coverage / width /
Winkler score, each with an engine-level bootstrap interval, plus coverage by true
RUL band (the fleet-stratified convention this project reports).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)  # tiny tensors: more threads only adds contention

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
MAX_RUL = 125
K = MAX_RUL + 1
BINS = np.arange(K)

sys.path.insert(0, str(ROOT / "scripts" / "cal"))
from calibrate import (CalibNet, choose_scale, fit, load_dev, load_test, merge,  # noqa: E402
                       pmf_scaled)


# ---------------------------------------------------------------- distributions
def dist_metrics(p, y, yhat):
    """p: (n, K) pmf over 0..125 (last bin = '>=125')."""
    cdf = np.cumsum(p, axis=1)
    crps = np.abs(cdf - (BINS[None, :] >= y[:, None])).sum(1)
    mean = (p * BINS[None, :]).sum(1)
    out = dict(rmse=float(np.sqrt(np.mean((mean - y) ** 2))),
               crps=float(crps.mean()))
    per = dict(sq=(mean - y) ** 2, crps=crps)
    for level in (0.5, 0.8, 0.9):
        a = (1 - level) / 2
        lo = np.array([np.searchsorted(cdf[i], a) for i in range(len(y))])
        hi = np.array([min(np.searchsorted(cdf[i], 1 - a), K - 1) for i in range(len(y))])
        cover = ((lo <= y) & (y <= hi)).astype(float)
        width = (hi - lo).astype(float)
        wink = width + (2 / (1 - level)) * np.maximum(lo - y, 0) + (2 / (1 - level)) * np.maximum(y - hi, 0)
        tag = int(level * 100)
        out[f"cover{tag}"] = float(cover.mean())
        out[f"width{tag}"] = float(width.mean())
        out[f"winkler{tag}"] = float(wink.mean())
        per[f"cover{tag}"] = cover
        if tag == 90:
            per["lo"], per["hi"] = lo, hi
    out["_per_sample"] = per
    return out


def interval_metrics(lo, hi, y):
    cover = ((lo <= y) & (y <= hi)).astype(float)
    width = (hi - lo).astype(float)
    wink = width + (2 / 0.10) * np.maximum(lo - y, 0) + (2 / 0.10) * np.maximum(y - hi, 0)
    # CRPS of the degenerate interval is not defined; only interval scores are reported.
    return dict(cover90=float(cover.mean()), width90=float(width.mean()),
                winkler90=float(wink.mean()),
                _per_sample=dict(cover=cover, width=width, wink=wink, lo=lo, hi=hi))


# ---------------------------------------------------------------- baselines
def homo_dist(dev, yhat):
    sigma = float(np.std(dev["y"] - dev["yhat"])) + 1e-6
    z = (BINS[None, :] - yhat[:, None]) / sigma
    lp = -0.5 * 6.0 * np.log1p(z ** 2 / 5.0)
    p = np.exp(lp - lp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def split_conformal(dev, yhat, alpha=0.10):
    """Constant-width, distribution-free 90% interval centred on the point forecast."""
    r = np.abs(dev["y"] - dev["yhat"])
    n = len(r)
    q = np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    lo = np.clip(yhat - q, 0, MAX_RUL)
    hi = np.clip(yhat + q, 0, MAX_RUL)
    return lo, hi


def _pinball(theta, x, target, tau):
    pred = theta[0] + theta[1] * x
    d = target - pred
    return torch.mean(torch.maximum(tau * d, (tau - 1) * d))


def cqr(dev, yhat, alpha=0.10, steps=2000, lr=0.03, n_knots=6, seed=0):
    """Conformalised quantile regression on a monotone piecewise-linear basis in yhat.

    Split-CQR: the quantile model is fitted on half the development engines and the
    conformity scores come from the other half. Scoring on the fitting sample would
    make the intervals optimistically narrow and destroy the coverage guarantee.

    A plain linear basis is too rigid when the label is bounded at 125: the fitted
    5% quantile then runs to the floor and the intervals explode. Knots let the
    quantiles bend near the cap, which is the fair version of this baseline.
    """
    knots = np.linspace(0.1, 0.9, n_knots)

    def feats(u):
        u = np.asarray(u, dtype=np.float32) / MAX_RUL
        return np.stack([np.ones_like(u), u] + [np.maximum(u - k, 0) for k in knots], axis=1)

    n = len(dev["y"])
    perm = np.random.default_rng(seed).permutation(n)
    fit_idx, cal_idx = perm[: n // 2], perm[n // 2:]

    x = torch.tensor(feats(dev["yhat"][fit_idx]), dtype=torch.float32)
    y = torch.tensor(dev["y"][fit_idx], dtype=torch.float32)
    qs = []
    for tau in (alpha / 2, 1 - alpha / 2):
        th = torch.zeros(x.shape[1], requires_grad=True)
        opt = torch.optim.Adam([th], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = x @ th
            d = y - pred
            loss = torch.mean(torch.maximum(tau * d, (tau - 1) * d)) + 1e-4 * th[2:].pow(2).sum()
            loss.backward()
            opt.step()
        qs.append(th.detach())

    xc = torch.tensor(feats(dev["yhat"][cal_idx]), dtype=torch.float32)
    lo_c = (xc @ qs[0]).numpy()
    hi_c = (xc @ qs[1]).numpy()
    yc = dev["y"][cal_idx]
    score = np.maximum(lo_c - yc, yc - hi_c)
    m = len(score)
    q = float(np.quantile(score, min(1.0, np.ceil((m + 1) * (1 - alpha)) / m)))

    xt = torch.tensor(feats(yhat), dtype=torch.float32)
    lo_hat = (xt @ qs[0]).numpy()
    hi_hat = (xt @ qs[1]).numpy()
    return np.clip(lo_hat - q, 0, MAX_RUL), np.clip(hi_hat + q, 0, MAX_RUL)


# ---------------------------------------------------------------- reporting
def bootstrap(per_sample, n_boot=2000, seed=0):
    """Engine-level bootstrap. Each metric has its own reducer: RMSE needs the square
    root of the resampled mean squared error, not the mean squared error."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(per_sample.values())))
    idx = rng.integers(0, n, size=(n_boot, n))
    reducers = {"sq": lambda v: np.sqrt(np.mean(v)), "crps": np.mean}
    for t in (50, 80, 90):
        reducers[f"cover{t}"] = np.mean
    out = {}
    for k, v in per_sample.items():
        if k not in reducers:
            continue
        v = np.asarray(v, dtype=np.float64)
        if k == "sq":
            vals = np.sqrt((v[idx] ** 2).mean(axis=1))
        else:
            vals = v[idx].mean(axis=1)
        out[k] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
    return out


def band_cover(lo, hi, y, edges=(0, 25, 50, 75, 100, 126)):
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        if m.sum() < 5:
            continue
        rows.append(dict(band=f"[{a},{b})", n=int(m.sum()),
                         cover90=float(np.mean((lo[m] <= y[m]) & (y[m] <= hi[m]))),
                         width90=float(np.mean(hi[m] - lo[m])),
                         bias=float(np.mean((lo[m] + hi[m]) / 2 - y[m]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(ROOT))
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--mode", default="pooled", choices=["per_subset", "loso", "pooled"])
    ap.add_argument("--arm", default="A")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cal" / "eval"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    stage = Path(a.stage)
    subsets = a.subsets.split(",")
    devs = {s: load_dev(stage, s, a.arm) for s in subsets}
    pooled_dev = merge(list(devs.values()))

    report = {}
    for s in subsets:
        dev, te = devs[s], load_test(stage, s, a.arm)
        fit_dev = (dev if a.mode == "per_subset"
                   else (merge([devs[o] for o in subsets if o != s]) if a.mode == "loso" else pooled_dev))
        y, yhat = te["y"], te["yhat"]

        net = fit(fit_dev)
        scale, gap = choose_scale(fit_dev)
        p = pmf_scaled(net, yhat, scale=scale)
        ours = dist_metrics(p, y, yhat)

        homo = dist_metrics(homo_dist(fit_dev, yhat), y, yhat)
        sc_lo, sc_hi = split_conformal(fit_dev, yhat)
        sc = interval_metrics(sc_lo, sc_hi, y)
        cq_lo, cq_hi = cqr(fit_dev, yhat)
        cq = interval_metrics(cq_lo, cq_hi, y)

        def pack(d):
            ci = bootstrap(d["_per_sample"], a.n_boot)
            keys = {"rmse": "sq", "crps": "crps"}
            for t in (50, 80, 90):
                keys[f"cover{t}"] = f"cover{t}"
            return {k: dict(value=d[k], ci=ci.get(keys.get(k, ""), [None, None]))
                    for k in d if not k.startswith("_")}

        # Randomised PIT. Capped labels (y = 125) are an atom, not a point: giving them
        # the CDF value at the cap piles every one of them into the top histogram bin
        # and manufactures a spurious U shape. Draw uniformly inside the atom instead.
        cdf = np.cumsum(p, axis=1)
        rng_pit = np.random.default_rng(0)
        pit = np.empty(len(y))
        for i in range(len(y)):
            if y[i] >= MAX_RUL:
                lo_p = cdf[i, K - 2] if K > 1 else 0.0
                pit[i] = rng_pit.uniform(lo_p, 1.0)
            else:
                lo_p = cdf[i, int(y[i]) - 1] if y[i] > 0 else 0.0
                pit[i] = rng_pit.uniform(lo_p, cdf[i, int(y[i])])
        hist, _ = np.histogram(pit, bins=10, range=(0, 1))

        report[s] = dict(
            n=int(len(y)), mode=a.mode, arm=a.arm, n_fit=int(len(fit_dev["y"])), scale=scale, scale_gap=gap,
            ours=pack(ours), homoscedastic=pack(homo),
            split_conformal=pack(sc), cqr=pack(cq),
            bands=band_cover(ours["_per_sample"]["lo"], ours["_per_sample"]["hi"], y),
            bands_split=band_cover(sc_lo, sc_hi, y),
            pit_hist=(hist / max(hist.sum(), 1)).tolist(),
        )

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    hdr = f"{'subset':7s} {'method':16s} {'RMSE':>7s} {'CRPS':>7s} {'PICP90':>7s} {'MPIW90':>7s} {'Wink90':>7s}"
    print(hdr); print("-" * len(hdr))
    for s, r in report.items():
        print(f"        fitted sigma scale {r['scale']:.2f} (dev coverage gap {r['scale_gap']:.3f})")
        for name in ("ours", "homoscedastic", "split_conformal", "cqr"):
            m = r[name]
            def f(k):
                return f"{m[k]['value']:7.2f}" if k in m else f"{'-':>7s}"
            def c(k):
                if k not in m:
                    return ""
                lo, hi = m[k]["ci"]
                return f" [{lo:.2f},{hi:.2f}]" if lo is not None else ""
            print(f"{s:7s} {name:16s} {f('rmse')}{c('rmse')} {f('crps')} {f('cover90')} {f('width90')} {f('winkler90')}")
        print()


if __name__ == "__main__":
    main()
