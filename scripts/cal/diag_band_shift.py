"""Why can a dev-fitted recalibrator not fix the test-set conditional bias?

The v2 comparison leaves one number unexplained: in the true-RUL band [75,100) every
backbone on every fleet over-predicts by 6-11 cycles, and the calibrated mean does not
remove it. Two explanations are possible and they have opposite consequences.

H1  the backbone is biased in that band and the layer has the capacity to fix it, so the
    failure is an objective/capacity failure of the layer;
H2  the band bias is *different* on dev than on test, so no function fitted on dev --
    however expressive -- can remove it on test without seeing test labels.

Only H2 would make the defect a property of the data protocol rather than of the method.
This script settles it by measuring the same band statistic on both samples.

Reported per (arm, subset), per true-RUL band:
  n_dev / n_test        sample sizes
  bb_dev / bb_test      mean(clip(yhat) - y)  -- backbone, before any calibration
  shift                 bb_test - bb_dev, the part no dev-fitted map can anticipate
  cal_all_dev           mean(mu_cens - y) on the pooled dev sample the layer was fitted to
  cal_test              mean(mu_cens - y) on the test band
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layer import MAX_RUL, choose_scale, fit_layer, gen_pmf  # noqa: E402
from calibrate import load_dev, load_test  # noqa: E402

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
BANDS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ARMS = ["A", "BE", "PE"]
GRID = np.arange(126, dtype=np.float64)


def band_of(y, a, b):
    return (y >= a) & (y <= b) if b == 126 else (y >= a) & (y < b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cal" / "band_shift.json"))
    args = ap.parse_args()

    report = {}
    for arm in ARMS:
        devs = {s: load_dev(ROOT, s, arm) for s in SUBSETS}
        pooled = {k: np.concatenate([d[k] for d in devs.values()])
                  for k in ("unit", "yhat", "y")}
        net = fit_layer(pooled, kind="mlp", mode="censored")
        scale, _ = choose_scale(lambda d: fit_layer(d, kind="mlp", mode="censored"), pooled)
        mu_dev = gen_pmf(net, pooled["yhat"], scale=scale) @ GRID

        rows = []
        for s in SUBSETS:
            te = load_test(ROOT, s, arm)
            mu_te = gen_pmf(net, te["yhat"], scale=scale) @ GRID
            dv = devs[s]
            for a, b in BANDS:
                md, mt = band_of(dv["y"], a, b), band_of(te["y"], a, b)
                if md.sum() == 0 or mt.sum() == 0:
                    continue
                bb_d = float(np.mean(np.clip(dv["yhat"][md], 0, MAX_RUL) - dv["y"][md]))
                bb_t = float(np.mean(np.clip(te["yhat"][mt], 0, MAX_RUL) - te["y"][mt]))
                rows.append({
                    "subset": s, "band": f"[{a},{b})",
                    "n_dev": int(md.sum()), "n_test": int(mt.sum()),
                    "bb_dev": bb_d, "bb_test": bb_t, "shift": bb_t - bb_d,
                    "cal_test": float(np.mean(mu_te[mt] - te["y"][mt])),
                })
        for r in rows:
            a, b = [int(v) for v in r["band"].strip("[]()").split(",")]
            m = band_of(pooled["y"], a, b)
            r["cal_dev"] = float(np.mean(mu_dev[m] - pooled["y"][m])) if m.sum() else None
            r["n_dev_all"] = int(m.sum())
        report[arm] = {"scale": float(scale), "rows": rows}

        print(f"\n=== arm {arm}  (scale={scale:.2f}) ===")
        print(f"{'subset':7s} {'band':10s} {'n_dev':>5s} {'n_test':>6s} "
              f"{'bb_dev':>7s} {'bb_test':>8s} {'shift':>7s} "
              f"{'cal_dev':>8s} {'cal_test':>9s} {'resid':>7s}")
        for r in rows:
            resid = r["cal_test"] - (r["cal_dev"] if r["cal_dev"] is not None else np.nan)
            print(f"{r['subset']:7s} {r['band']:10s} {r['n_dev']:5d} {r['n_test']:6d} "
                  f"{r['bb_dev']:7.2f} {r['bb_test']:8.2f} {r['shift']:+7.2f} "
                  f"{(r['cal_dev'] if r['cal_dev'] is not None else np.nan):8.2f} "
                  f"{r['cal_test']:9.2f} {resid:+7.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
