"""Recompute the headline metrics from committed artefacts only.

Loads experiments/cal/calibrators/calibrator_<arm>.json plus the official test
predictions, rebuilds the predictive pmf by hand, and checks the result against
experiments/cal/eval_<arm>/eval_report.json.

Nothing here touches a checkpoint, a GPU, or a random seed. If this passes, the reported
numbers are a function of the committed files alone -- which is the whole point of
freezing the 66 weights instead of describing the fitting procedure.
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

from calibrate import K, MAX_RUL, load_test  # noqa: E402


def pmf_from_record(rec, yhat):
    w = rec["weights"]
    x = (np.asarray(yhat, dtype=np.float32) / MAX_RUL)[:, None]
    h = np.tanh(x @ np.array(w["net.0.weight"]).T + np.array(w["net.0.bias"]))
    o = h @ np.array(w["net.2.weight"]).T + np.array(w["net.2.bias"])
    mu = yhat + 20.0 * o[:, 0]
    log_sigma = np.clip(o[:, 1], -4.0, 3.0)
    sigma = np.log1p(np.exp(log_sigma)) * 20.0 + 1.0
    sigma = sigma * rec["scale"]["value"]
    bins = np.arange(K)[None, :]
    z = (bins - mu[:, None]) / sigma[:, None]
    lp = -0.5 * (rec["architecture"]["nu"] + 1.0) * np.log1p(z ** 2 / rec["architecture"]["nu"])
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def score(p, y):
    cdf = np.cumsum(p, axis=1)
    bins = np.arange(K)
    crps = float(np.abs(cdf - (bins[None, :] >= y[:, None])).sum(1).mean())
    mean = (p * bins[None, :]).sum(1)
    lo = np.array([np.searchsorted(cdf[i], 0.05) for i in range(len(y))])
    hi = np.minimum(np.array([np.searchsorted(cdf[i], 0.95) for i in range(len(y))]), K - 1)
    cover = float(np.mean((lo <= y) & (y <= hi)))
    width = float(np.mean(hi - lo))
    return dict(crps=crps, rmse=float(np.sqrt(np.mean((mean - y) ** 2))),
                cover90=cover, width90=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(ROOT))
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--arm", default="A")
    a = ap.parse_args()

    rec = json.loads((ROOT / "experiments" / "cal" / "calibrators" /
                      f"calibrator_{a.arm}.json").read_text())
    ref = json.loads((ROOT / "experiments" / "cal" / f"eval_{a.arm}" /
                      "eval_report.json").read_text())

    worst = 0.0
    print(f"arm {a.arm}  scale {rec['scale']['value']:.2f}  "
          f"dev_sha {rec['fit']['dev_sha256'][:12]}  n_fit {rec['fit']['n_rows']}")
    print("sub    crps frozen / report     cover90 frozen / report   width90 frozen / report")
    for s in a.subsets.split(","):
        te = load_test(Path(a.stage), s, a.arm)
        m = score(pmf_from_record(rec, te["yhat"]), te["y"])
        r = ref[s]["ours"]
        for k in ("crps", "cover90", "width90"):
            worst = max(worst, abs(m[k] - r[k]["value"]))
        print(f"{s}  {m['crps']:8.3f} / {r['crps']['value']:8.3f}   "
              f"{m['cover90']:6.3f} / {r['cover90']['value']:6.3f}   "
              f"{m['width90']:7.2f} / {r['width90']['value']:7.2f}")
    print(f"\nmax |frozen - report| over crps/cover90/width90 = {worst:.2e}")
    print("PASS" if worst < 5e-3 else "FAIL: the frozen record does not reproduce the report")


if __name__ == "__main__":
    main()
