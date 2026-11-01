"""Is the top bin carrying TAIL MASS or merely DENSITY?

The layer claims a right-censored likelihood: a label at the 125 cap should contribute
P(R >= 125) = S(125). The implementation builds log-probabilities as a Student-t density
evaluated on the integer grid and then normalises with a softmax. Normalising a density
over a finite grid is *truncation*, not censoring, so the top bin ends up holding the
density at 125 rather than the surviving mass beyond it.

For a t(0,20,nu=5) centred at 113 the two differ by more than an order of magnitude. If
so, every capped label is penalised by roughly that ratio, which pushes predictions down
exactly where the cap lives -- and would explain the -10 cycle bias in the [100,126) band
without invoking any measurement story.

This script measures both quantities on the frozen calibrator and reports the ratio.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))

from calibrate import K, MAX_RUL  # noqa: E402


def t_cdf(z, nu):
    """Student-t CDF via the incomplete beta identity, vectorised."""
    from scipy.special import betainc
    x = nu / (nu + z ** 2)
    tail = 0.5 * betainc(nu / 2.0, 0.5, x)
    return np.where(z > 0, 1.0 - tail, tail)


def scale_params(rec, yhat, scale=None):
    w = rec["weights"]
    x = (np.asarray(yhat, dtype=np.float64) / MAX_RUL)[:, None]
    h = np.tanh(x @ np.array(w["net.0.weight"]).T + np.array(w["net.0.bias"]))
    o = h @ np.array(w["net.2.weight"]).T + np.array(w["net.2.bias"])
    mu = np.asarray(yhat, dtype=np.float64) + 20.0 * o[:, 0]
    sigma = np.log1p(np.exp(np.clip(o[:, 1], -4.0, 3.0))) * 20.0 + 1.0
    s = rec["scale"]["value"] if scale is None else scale
    return mu, sigma * s


def pmf_density(rec, yhat):
    """What the code currently does: softmax of the density over the integer grid."""
    mu, sigma = scale_params(rec, yhat)
    nu = rec["architecture"]["nu"]
    z = (np.arange(K)[None, :] - mu[:, None]) / sigma[:, None]
    lp = -0.5 * (nu + 1.0) * np.log1p(z ** 2 / nu)
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def pmf_censored(rec, yhat):
    """What censoring means: bin b<125 holds [b,b+1), bin 125 holds [125, inf).

    p[0]   = F(1)                 (everything below 1 lands in bin 0)
    p[b]   = F(b+1) - F(b),  b=1..123
    p[124] = F(125) - F(124)
    p[125] = 1 - F(125)           <- surviving mass, not a density
    """
    mu, sigma = scale_params(rec, yhat)
    nu = rec["architecture"]["nu"]
    edges = np.arange(1, K, dtype=np.float64)           # 1..125
    F = t_cdf((edges[None, :] - mu[:, None]) / sigma[:, None], nu)
    p = np.zeros((len(mu), K), dtype=np.float64)
    p[:, 0] = F[:, 0]
    p[:, 1:K - 1] = np.diff(F, axis=1)
    p[:, K - 1] = 1.0 - F[:, -1]
    p = np.clip(p, 0.0, None)
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--stage", default=str(ROOT))
    a = ap.parse_args()

    rec = json.loads((ROOT / "experiments" / "cal" / "calibrators" /
                      f"calibrator_{a.arm}.json").read_text())
    print(f"arm {a.arm}  scale {rec['scale']['value']:.2f}  nu {rec['architecture']['nu']}")
    print("sub    mean p[125]  proper S(125)   ratio    mean(now)  mean(cens)  truth_mean")
    for s in a.subsets.split(","):
        te = load(a.stage, s, a.arm)
        yh, y = te["yhat"], te["y"]
        pd_ = pmf_density(rec, yh)
        pc = pmf_censored(rec, yh)
        bins = np.arange(K)
        m_d = (pd_ * bins).sum(1)
        m_c = (pc * bins).sum(1)
        mu, sigma = scale_params(rec, yh)
        proper = 1.0 - t_cdf((MAX_RUL - mu) / sigma, rec["architecture"]["nu"])
        print(f"{s}  {pd_[:, -1].mean():11.4f}  {proper.mean():13.4f}  "
              f"{proper.mean() / max(pd_[:, -1].mean(), 1e-9):6.1f}x  "
              f"{m_d.mean():9.2f}  {m_c.mean():10.2f}  {y.mean():10.2f}")
        # bias inside the band where the atom dominates
        m = (y >= 100) & (y <= 125)
        print(f"       [100,126) n={int(m.sum()):3d}  bias now {m_d[m].mean() - y[m].mean():+.2f}"
              f"   bias censored {m_c[m].mean() - y[m].mean():+.2f}")


def load(stage, subset, arm):
    import csv
    f = Path(stage) / "results" / "dsh" / "e21" / "preds" / f"{subset}_{arm}.csv"
    us, ps, ys = [], [], []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            us.append(int(r["unit"]))
            ps.append(float(r["y_pred"]))
            ys.append(min(float(r["y_true"]), MAX_RUL))
    return {"unit": np.array(us), "yhat": np.array(ps), "y": np.array(ys)}


if __name__ == "__main__":
    main()
