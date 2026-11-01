"""Can any map of yhat remove the band bias? Trace the exact trade-off instead of asserting.

The negative claim in the report is easy to attack: "you did not find the right map". This
script removes that attack by making the whole achievable set computable.

Consider the one-parameter family g_t(yhat) = yhat + t * (mean(y) - yhat), which shrinks the
prediction toward the marginal mean. t = 0 is the untouched backbone, t = 1 is a constant
predictor with zero resolution. For each t we report three things on test:

  rmse        sqrt(mean((g_t(yhat) - y)^2)), the resolution the map gives up
  max_abs_b   the worst | mean(g_t(yhat) - y | y in band) | over the five RUL bands
  mean_abs_b  the average |band bias|

If some map of yhat could fix the band bias at no cost, max_abs_b would reach ~0 at a t whose
rmse is still near the t = 0 value. It does not: max_abs_b falls only by inflating rmse, and
the minimum sits at several cycles, not at zero. That is the quantitative form of the claim
that the bias is carried by the information in yhat, not by the calibrator.

Isotonic regression is also run as the unconstrained-monotone reference, and a per-band
oracle shift -- which is allowed to look at the test bands directly and is therefore not
achievable by any real method -- as an upper bound on what band conditioning would buy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer import MAX_RUL  # noqa: E402
from calibrate import load_dev, load_test  # noqa: E402

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
BANDS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ARMS = ["A", "BE", "PE"]


def band_of(y, a, b):
    return (y >= a) & (y <= b) if b == 126 else (y >= a) & (y < b)


def pava(y):
    y = np.asarray(y, dtype=np.float64)
    val, wt, cnt = [], [], []
    for yi in y:
        val.append(yi); wt.append(1.0); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v = (val[-2] * wt[-2] + val[-1] * wt[-1]) / (wt[-2] + wt[-1])
            w = wt[-2] + wt[-1]
            c = cnt[-2] + cnt[-1]
            val[-2:] = [v]; wt[-2:] = [w]; cnt[-2:] = [c]
    return np.repeat(val, cnt)


def metrics(g, y, t=None):
    out = {"rmse": float(np.sqrt(np.mean((g - y) ** 2)))}
    bs = []
    for a, b in BANDS:
        m = band_of(y, a, b)
        if m.sum():
            bs.append(float(np.mean(g[m] - y[m])))
    out["max_abs_b"] = float(max(abs(v) for v in bs))
    out["mean_abs_b"] = float(np.mean(np.abs(bs)))
    out["biases"] = [round(v, 2) for v in bs]
    return out


def main():
    report = {}
    for arm in ARMS:
        devs = {s: load_dev(ROOT, s, arm) for s in SUBSETS}
        pooled = {k: np.concatenate([d[k] for d in devs.values()])
                  for k in ("unit", "yhat", "y")}
        ybar = float(np.mean(pooled["y"]))
        # isotonic reference, fitted on dev
        o = np.argsort(pooled["yhat"], kind="mergesort")
        gy = pava(pooled["y"][o])
        ux, idx = np.unique(pooled["yhat"][o], return_index=True)
        gx = gy[idx]

        print(f"\n=== arm {arm}  mean(y_dev)={ybar:.1f} ===")
        print(f"{'t':>5s} {'rmse':>7s} {'max|b|':>7s} {'mean|b|':>8s}  band biases")
        rows = []
        for t in np.linspace(0.0, 1.0, 11):
            gs, ys = [], []
            for s in SUBSETS:
                te = load_test(ROOT, s, arm)
                yh = np.clip(te["yhat"], 0, MAX_RUL)
                gs.append(yh + t * (ybar - yh))
                ys.append(te["y"])
            m = metrics(np.concatenate(gs), np.concatenate(ys))
            rows.append({"t": float(t), **m})
            print(f"{t:5.2f} {m['rmse']:7.2f} {m['max_abs_b']:7.2f} "
                  f"{m['mean_abs_b']:8.2f}  {m['biases']}")

        best = min(rows, key=lambda r: r["max_abs_b"])
        base = rows[0]
        print(f"  -> bias-minimising t={best['t']:.2f}: max|b| {base['max_abs_b']:.2f} -> "
              f"{best['max_abs_b']:.2f}, rmse {base['rmse']:.2f} -> {best['rmse']:.2f} "
              f"(+{100 * (best['rmse'] / base['rmse'] - 1):.1f}%)")

        # isotonic reference
        gs, ys = [], []
        for s in SUBSETS:
            te = load_test(ROOT, s, arm)
            yh = np.clip(te["yhat"], 0, MAX_RUL)
            j = np.clip(np.searchsorted(ux, yh, side="right") - 1, 0, len(gx) - 1)
            gs.append(gx[j]); ys.append(te["y"])
        iso = metrics(np.concatenate(gs), np.concatenate(ys))
        print(f"  -> isotonic (best monotone for MSE): rmse {iso['rmse']:.2f} "
              f"max|b| {iso['max_abs_b']:.2f}  biases {iso['biases']}")

        # oracle per-band shift: not achievable, an upper bound only
        gs, ys = [], []
        for s in SUBSETS:
            te = load_test(ROOT, s, arm)
            yh = np.clip(te["yhat"], 0, MAX_RUL).astype(np.float64)
            y = te["y"]
            for a, b in BANDS:
                m = band_of(y, a, b)
                if m.sum():
                    yh[m] -= float(np.mean(yh[m] - y[m]))
            gs.append(yh); ys.append(y)
        orc = metrics(np.concatenate(gs), np.concatenate(ys))
        print(f"  -> per-band oracle (not achievable): rmse {orc['rmse']:.2f} "
              f"max|b| {orc['max_abs_b']:.2f}")

        report[arm] = {"ybar": ybar, "curve": rows, "best": best, "isotonic": iso,
                       "oracle": orc}

    out = ROOT / "experiments" / "cal" / "bias_tradeoff.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
