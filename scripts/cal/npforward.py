"""Numpy forward pass for the frozen 66-parameter layer.

The frozen calibrator JSONs hold the weights; `layer.py` holds the forward pass in
torch. Torch is not installed on the analysis machine, and the audit that follows
consumes only *interval endpoints*, not gradients. So we re-implement the two layers in
numpy -- 1->16 tanh ->2 -- and check agreement against the torch report before using it
for anything.

`sigma` applies the clamp to the pre-activation and then softplus, exactly as
`layer.LocationScale.params` does.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
BASE = 125.0


def softplus(x):
    return np.logaddexp(0.0, x)


def load_weights(path):
    d = json.loads(Path(path).read_text())
    return {
        "w0": np.asarray(d["weights"]["net.0.weight"], float),
        "b0": np.asarray(d["weights"]["net.0.bias"], float),
        "w2": np.asarray(d["weights"]["net.2.weight"], float),
        "b2": np.asarray(d["weights"]["net.2.bias"], float),
        "scale": float(d.get("scale", {}).get("value", 1.0)),
        "meta": d,
    }


def forward(w, yhat, max_rul):
    """(mu, sigma) with the same dimensional rescaling as `layer.params`."""
    r = max_rul / BASE
    x = np.asarray(yhat, float) / max_rul
    h = np.tanh(x[:, None] * w["w0"][:, 0][None, :] + w["b0"][None, :])
    out = h @ w["w2"].T + w["b2"][None, :]
    mu = np.asarray(yhat, float) + 20.0 * r * out[:, 0]
    sigma = softplus(np.clip(out[:, 1], -4.0, 3.0)) * 20.0 * r + 1.0 * r
    return mu, sigma


def pmf_censored(mu, sigma, K):
    Fm = ndtr((np.arange(1, K)[None, :] - np.asarray(mu)[:, None]) / np.asarray(sigma)[:, None])
    p = np.zeros((len(mu), K), float)
    p[:, 0] = Fm[:, 0]
    p[:, 1:K - 1] = np.diff(Fm, axis=1)
    p[:, K - 1] = 1.0 - Fm[:, -1]
    p = np.clip(p, 0.0, None)
    return p / p.sum(1, keepdims=True)


def interval(p, level):
    cdf = np.cumsum(p, axis=1)
    a = (1.0 - level) / 2.0
    n, K = p.shape
    lo = np.empty(n, int)
    hi = np.empty(n, int)
    for i in range(n):
        lo[i] = int(np.searchsorted(cdf[i], a))
        hi[i] = min(int(np.searchsorted(cdf[i], 1.0 - a)), K - 1)
    return lo, hi


def load_test(stage, subset, arm, cap=125.0):
    f = Path(stage) / "results" / "dsh" / "e21" / "preds" / f"{subset}_{arm}.csv"
    us, ps, ys = [], [], []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            us.append(int(r["unit"]))
            ps.append(float(r["y_pred"]))
            ys.append(min(float(r["y_true"]), cap))
    return {"unit": np.array(us), "yhat": np.array(ps), "y": np.array(ys)}


def load_dev(subset, arm, cap=125.0):
    """Seed/fold average, matching `calibrate.load_dev`."""
    d = ROOT / "experiments" / "cal" / "dev_preds" / subset
    agg, truth = {}, {}
    for f in sorted(d.glob(f"{arm}_s*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                u = int(r["unit_id"])
                agg.setdefault(u, []).append(float(r["y_pred"]))
                truth[u] = min(float(r["y_true"]), cap)
    units = sorted(agg)
    return {"unit": np.array(units),
            "yhat": np.array([np.mean(agg[u]) for u in units]),
            "y": np.array([truth[u] for u in units])}
