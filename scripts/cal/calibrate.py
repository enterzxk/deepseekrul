"""Post-hoc calibrated survival layer for the A-arm (hybrid route).

Claim under test: a window regressor's point prediction can be turned into a calibrated
RUL distribution WITHOUT retraining or changing the point estimate. The layer is fitted
on out-of-fold development predictions only (each engine is predicted by models that
never saw it), cross-fitted by engine, then applied once to the official E21 test
predictions.

Distribution: discretised Student-t over RUL bins 0..124 plus a >=125 tail, with
location and scale as smooth functions of the backbone's point prediction. Fitted by
right-censored NLL (labels at the 125 cap are censored, not point masses).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
MAX_RUL = 125
K = MAX_RUL + 1  # bins 0..124 + tail


class CalibNet(nn.Module):
    def __init__(self, hidden: int = 16, nu: float = 5.0):
        super().__init__()
        self.nu = nu
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.bins = nn.Parameter(torch.arange(0, K, dtype=torch.float32), requires_grad=False)

    def params(self, yhat):
        x = (yhat / MAX_RUL).unsqueeze(-1)
        out = self.net(x)
        mu = yhat + 20.0 * out[..., 0]
        log_sigma = out[..., 1].clamp(-4.0, 3.0)
        sigma = F.softplus(log_sigma) * 20.0 + 1.0
        return mu, sigma

    def log_prob(self, yhat, y):
        mu, sigma = self.params(yhat)
        z = (self.bins.unsqueeze(0) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
        logp = -0.5 * (self.nu + 1.0) * torch.log1p(z ** 2 / self.nu)
        logp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)
        idx = y.clamp(0, K - 1).long()
        point = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        tail = logp[..., K - 1]
        return torch.where(y >= MAX_RUL, tail, point)


def fit(dev, epochs=400, lr=0.05, seed=0):
    torch.manual_seed(seed)
    net = CalibNet()
    yhat = torch.tensor(dev["yhat"], dtype=torch.float32)
    y = torch.tensor(dev["y"], dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -net.log_prob(yhat, y).mean() + 1e-3 * sum(p.pow(2).sum() for p in net.parameters())
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def pmf(net, yhat):
    mu, sigma = net.params(torch.tensor(yhat, dtype=torch.float32))
    z = (net.bins.unsqueeze(0) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    logp = -0.5 * (net.nu + 1.0) * torch.log1p(z ** 2 / net.nu)
    return torch.softmax(logp, dim=-1).numpy()


@torch.no_grad()
def pmf_preserve_mean(net, yhat, scale=1.0, iters=40):
    """Predictive pmf re-tilted so that E[Y] equals the backbone point forecast exactly.

    The tilt p_b ~ p_b * exp(lambda * b) is the minimum-KL distribution with the
    requested mean, so the shape learned on development data is preserved as far as
    the mean constraint allows. Calibration is then strictly additive: RMSE is the
    backbone's RMSE to machine precision, and any accuracy change is attributable to
    the backbone alone.
    """
    yt = torch.tensor(np.asarray(yhat, dtype=np.float32))
    mu, sigma = net.params(yt)
    sigma = sigma * scale
    bins = net.bins
    z = (bins.unsqueeze(0) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    logp = -0.5 * (net.nu + 1.0) * torch.log1p(z ** 2 / net.nu)
    lp = logp - torch.logsumexp(logp, dim=-1, keepdim=True)

    def mean_at(lam):
        w = torch.log_softmax(lp + lam * bins.unsqueeze(0), dim=-1).exp()
        return (w * bins).sum(-1)

    lo = torch.full((len(yt), 1), -25.0)
    hi = torch.full((len(yt), 1), 25.0)
    target = yt.unsqueeze(-1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = mean_at(mid).unsqueeze(-1)
        too_low = m < target
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    lam = 0.5 * (lo + hi)
    return torch.log_softmax(lp + lam * bins.unsqueeze(0), dim=-1).exp().numpy()


@torch.no_grad()
def pmf_scaled(net, yhat, scale=1.0):
    """Predictive pmf with the learned location/scale, optionally widened by `scale`."""
    yt = torch.tensor(np.asarray(yhat, dtype=np.float32))
    mu, sigma = net.params(yt)
    sigma = sigma * scale
    bins = net.bins
    z = (bins.unsqueeze(0) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    logp = -0.5 * (net.nu + 1.0) * torch.log1p(z ** 2 / net.nu)
    return torch.softmax(logp, dim=-1).numpy()


def coverage_of(p, y):
    cdf = np.cumsum(p, axis=1)
    lo = np.array([np.searchsorted(cdf[i], 0.05) for i in range(len(y))])
    hi = np.array([min(np.searchsorted(cdf[i], 0.95), K - 1) for i in range(len(y))])
    return float(np.mean((lo <= y) & (y <= hi)))


def choose_scale(dev, target=0.90, n_splits=5, seed=0, grid=None):
    """Pick the sigma multiplier that makes CROSS-FITTED development coverage hit target.

    In-sample coverage is optimistic: the layer was fitted on those very engines. The
    scale is therefore calibrated on held-out folds only, which makes the nominal
    level a property of the construction rather than a hope.
    """
    if grid is None:
        grid = np.linspace(0.7, 2.5, 37)
    rng = np.random.default_rng(seed)
    n = len(dev["y"])
    order = rng.permutation(n)
    folds = np.array_split(order, n_splits)
    idx = np.arange(n)
    nets = []
    for f in folds:
        mask = np.ones(len(idx), bool); mask[f] = False
        nets.append((fit({k: dev[k][mask] for k in ("yhat", "y")}),
                     f, dev["y"][f], dev["yhat"][f]))
    # Coverage is NOT monotone in the scale: past a point the pmf approaches a uniform
    # distribution over 0..125, whose central 90% excludes the top of the grid, so every
    # label sitting exactly at the cap falls outside. Taking the global argmin of the gap
    # then lands on that pathological branch. Take the SMALLEST scale that attains the
    # nominal level instead: it is both the sharpest and the stable choice.
    covs = []
    for s in grid:
        ps = np.zeros((len(idx), K), dtype=np.float32)
        for net, f, _, yh in nets:
            ps[f] = pmf_scaled(net, yh, scale=s)
        covs.append((float(s), coverage_of(ps, dev["y"])))
    for s, c in covs:
        if c >= target:
            return s, abs(c - target)
    s, c = min(covs, key=lambda t: abs(t[1] - target))
    return s, abs(c - target)


def metrics(p, y, yhat):
    cdf = np.cumsum(p, axis=1)
    bins = np.arange(K)
    crps = np.abs(cdf - (bins[None, :] >= y[:, None])).sum(1).mean()
    mean = (p * bins[None, :]).sum(1)
    rmse = float(np.sqrt(np.mean((mean - y) ** 2)))
    rmse_point = float(np.sqrt(np.mean((yhat - y) ** 2)))
    out = {"n": int(len(y)), "rmse_dist_mean": rmse, "rmse_backbone": rmse_point,
           "crps": float(crps), "bias_dist": float(np.mean(mean - y))}
    for level in (0.5, 0.8, 0.9):
        lo = np.array([np.searchsorted(cdf[i], (1 - level) / 2) for i in range(len(y))])
        hi = np.array([np.searchsorted(cdf[i], 1 - (1 - level) / 2) for i in range(len(y))])
        hi = np.minimum(hi, K - 1)
        cover = float(np.mean((lo <= y) & (y <= hi)))
        width = float(np.mean(hi - lo))
        alpha = 1 - level
        wink = float(np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)))
        out[f"cover{int(level*100)}"] = cover
        out[f"width{int(level*100)}"] = width
        out[f"winkler{int(level*100)}"] = wink
    return out


def load_dev(stage: Path, subset: str, arm: str = "A"):
    rows = []
    direct = ROOT / "experiments" / "cal" / "dev_preds" / subset
    files = sorted(direct.glob(f"{arm}_s*.csv")) if direct.is_dir() else []
    if files:
        import csv
        for f in files:
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    rows.append((int(r["unit_id"]), float(r["y_true"]), float(r["y_pred"])))
    for f in ([] if files else glob.glob(str(stage / "experiments" / "**" / "matched_cutoff_predictions.csv"), recursive=True)):
        if f"/A_s" not in f.replace("\\", "/"):
            continue
        if subset not in f:
            continue
        import csv
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["unit_id"]), float(r["y_true"]), float(r["y_pred"])))
    if not rows:
        raise FileNotFoundError(f"no dev predictions for {subset} under {stage}")
    # average the seeds that predict the same engine -> a small ensemble, closer to test
    from collections import defaultdict
    agg = defaultdict(list)
    truth = {}
    for u, t, p in rows:
        agg[u].append(p)
        truth[u] = t
    units = sorted(agg)
    return {"unit": np.array(units), "yhat": np.array([np.mean(agg[u]) for u in units], dtype=np.float32),
            "y": np.array([min(truth[u], MAX_RUL) for u in units], dtype=np.float32)}


def load_test(stage: Path, subset: str, arm: str = "A"):
    import csv
    f = stage / "results" / "dsh" / "e21" / "preds" / f"{subset}_{arm}.csv"
    us, ps, ys = [], [], []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            us.append(int(r["unit"]))
            ps.append(float(r["y_pred"]))
            ys.append(min(float(r["y_true"]), MAX_RUL))
    return {"unit": np.array(us), "yhat": np.array(ps, dtype=np.float32), "y": np.array(ys, dtype=np.float32)}


def cross_fit(dev, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    units = dev["unit"]
    order = rng.permutation(len(units))
    folds = np.array_split(order, n_splits)
    idx = np.arange(len(units))
    p_all = np.zeros((len(units), K), dtype=np.float32)
    for f in folds:
        mask = np.ones(len(units), bool); mask[f] = False
        net = fit({k: dev[k][mask] for k in ("yhat", "y")})
        p_all[f] = pmf(net, dev["yhat"][f])
    return p_all


def merge(ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ("yhat", "y")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(ROOT / "_calstage"))
    ap.add_argument("--subsets", default="FD001")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cal"))
    ap.add_argument("--mode", default="per_subset", choices=["per_subset", "loso", "pooled"],
                    help="per_subset: fit on the target subset's own dev engines. "
                         "loso: fit on the OTHER subsets' dev engines only (zero target leakage). "
                         "pooled: fit once on every dev engine.")
    a = ap.parse_args()
    stage = Path(a.stage)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    subsets = a.subsets.split(",")
    devs = {s: load_dev(stage, s) for s in subsets}
    pooled_dev = merge(list(devs.values()))

    report = {}
    for subset in subsets:
        dev = devs[subset]
        te = load_test(stage, subset)
        p_cv = cross_fit(dev)
        cv = metrics(p_cv, dev["y"], dev["yhat"])
        if a.mode == "loso":
            others = [devs[s] for s in subsets if s != subset] or [dev]
            fit_dev = merge(others)
        elif a.mode == "pooled":
            fit_dev = pooled_dev
        else:
            fit_dev = dev
        net = fit(fit_dev)
        p_te = pmf(net, te["yhat"])
        test = metrics(p_te, te["y"], te["yhat"])
        # homoscedastic baseline fitted on the same residuals the layer was fitted on
        sigma = float(np.std(fit_dev["y"] - fit_dev["yhat"])) + 1e-6
        mu = te["yhat"]
        z = (np.arange(K)[None, :] - mu[:, None]) / sigma
        lp = -0.5 * 6.0 * np.log1p(z ** 2 / 5.0)
        p0 = np.exp(lp - lp.max(1, keepdims=True)); p0 /= p0.sum(1, keepdims=True)
        base = metrics(p0, te["y"], te["yhat"])
        report[subset] = {"dev": cv, "test": test, "test_homoscedastic": base,
                          "sigma_dev": sigma, "n_fit": int(len(fit_dev["y"])),
                          "n_dev": int(len(dev["y"])), "n_test": int(len(te["y"])), "mode": a.mode}
        print(subset, json.dumps({"dev_cv": cv, "test": test, "test_base": base}, indent=1))

    (out / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", out / "calibration_report.json")


if __name__ == "__main__":
    main()
