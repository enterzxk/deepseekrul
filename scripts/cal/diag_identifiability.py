"""How much of the conditional bias is removable by ANY function of the prediction?

The v2 experiments show that a 66-parameter location-scale layer fails to remove the
band-conditional bias, and a 4-parameter affine form does no worse. The tempting reading is
"the layer is too weak". This script tests the opposite: that no g(yhat), however flexible,
can remove it, because the bias is conditional on the TRUE state while g only sees the
PREDICTION, and the two are not in one-to-one correspondence.

Three quantities per (arm, subset):

  spread_ratio   sd(y | yhat in bin) / sd(y)     how much label variation survives
                                                 conditioning on the prediction
  yhat_iqr_bin   the width of a yhat bin that still contains both a y=80 and a y=125
                                                 engine -- the direct obstruction
  bias_g         band bias of g*(yhat), where g* is the least-squares-optimal monotone
                 map E[y | yhat] estimated by isotonic regression on dev. This is the best
                 any prediction-only recalibrator can do for squared error, so whatever
                 band bias survives is irreducible, not a capacity failure.
"""

from __future__ import annotations

import argparse
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


def pava(y, w=None):
    """Pool-adjacent-violators: the monotone least-squares fit, i.e. E[y | order] isotonic."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    val, wt, cnt = [], [], []
    for yi, wi in zip(y, w):
        val.append(yi); wt.append(wi); cnt.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v = (val[-2] * wt[-2] + val[-1] * wt[-1]) / (wt[-2] + wt[-1])
            wsum = wt[-2] + wt[-1]
            c = cnt[-2] + cnt[-1]
            val[-2:] = [v]; wt[-2:] = [wsum]; cnt[-2:] = [c]
    return np.repeat(val, cnt)


def fit_isotonic(x, y):
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o]
    g = pava(ys)
    # collapse to a step function: for each distinct x, the fitted value
    ux, idx = np.unique(xs, return_index=True)
    return ux, g[idx]


def apply_isotonic(par, x):
    ux, gy = par
    j = np.clip(np.searchsorted(ux, x, side="right") - 1, 0, len(gy) - 1)
    return gy[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cal" / "identifiability.json"))
    args = ap.parse_args()

    report = {}
    for arm in ARMS:
        devs = {s: load_dev(ROOT, s, arm) for s in SUBSETS}
        pooled = {k: np.concatenate([d[k] for d in devs.values()])
                  for k in ("unit", "yhat", "y")}
        par = fit_isotonic(pooled["yhat"], pooled["y"])

        print(f"\n=== arm {arm} ===")
        print(f"{'subset':7s} {'band':10s} {'n':>4s} {'bb':>7s} {'g*':>7s} "
              f"{'resid':>7s} {'yhat_lo':>8s} {'yhat_hi':>8s} {'pcap':>5s}")
        rows = []
        for s in SUBSETS:
            te = load_test(ROOT, s, arm)
            y, yh = te["y"], np.clip(te["yhat"], 0, MAX_RUL)
            g = np.clip(apply_isotonic(par, yh), 0, MAX_RUL)
            for a, b in BANDS:
                m = (y >= a) & (y <= b) if b == 126 else (y >= a) & (y < b)
                if m.sum() == 0:
                    continue
                bb = float(np.mean(yh[m] - y[m]))
                gb = float(np.mean(g[m] - y[m]))
                rows.append({"subset": s, "band": f"[{a},{b})", "n": int(m.sum()),
                             "bias_bb": bb, "bias_g": gb, "resid": gb - bb,
                             "yhat_lo": float(yh[m].min()), "yhat_hi": float(yh[m].max()),
                             "p_capped": float((y[m] >= MAX_RUL).mean())})
                print(f"{s:7s} {f'[{a},{b})':10s} {int(m.sum()):4d} {bb:7.2f} {gb:7.2f} "
                      f"{gb - bb:+7.2f} {yh[m].min():8.1f} {yh[m].max():8.1f} "
                      f"{(y[m] >= MAX_RUL).mean():5.2f}")

        # ---- how much label spread survives conditioning on a narrow yhat bin ----------
        edges = np.arange(0, 126, 10)
        spreads = []
        all_y = np.concatenate([load_test(ROOT, s, arm)["y"] for s in SUBSETS])
        all_h = np.concatenate([np.clip(load_test(ROOT, s, arm)["yhat"], 0, MAX_RUL)
                                for s in SUBSETS])
        sd_all = float(np.std(all_y))
        for a, b in zip(edges[:-1], edges[1:]):
            m = (all_h >= a) & (all_h < b)
            if m.sum() >= 10:
                spreads.append({"bin": f"[{a},{b})", "n": int(m.sum()),
                                "sd_y": float(np.std(all_y[m])),
                                "ratio": float(np.std(all_y[m]) / sd_all),
                                "share_capped": float((all_y[m] >= MAX_RUL).mean())})
        report[arm] = {"bands": rows, "spread": spreads, "sd_y_all": sd_all}
        print(f"  sd(y) overall = {sd_all:.2f};  label spread left after conditioning on yhat:")
        for r in spreads:
            print(f"    yhat {r['bin']:9s} n={r['n']:4d}  sd(y)={r['sd_y']:6.2f} "
                  f"ratio={r['ratio']:.2f}  share_at_cap={r['share_capped']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
