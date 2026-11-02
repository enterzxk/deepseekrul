"""Two objects called "the calibration layer" live in this repository. They are not the same.

  experiments/cal/audit/{subset}_A.npz     layer.LocationScale  (Gaussian pmf, t-free)
  experiments/cal/calibrators/calibrator_A.json   calibrate.CalibNet (t5 density)

Both are 66-parameter location-scale heads with a censored likelihood, both were fitted on
the same 709 out-of-fold rows, and both are described in the write-ups as "the layer".
Scored on the same 653 test engines at each one's own dev-tuned scale they differ by
0.07-0.09 in coverage. This script puts them side by side so the write-up can name which
one every number came from.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
from npforward import forward, load_weights, pmf_censored  # noqa: E402
from protocol_audit import coverage_profile, coverage_profile_randomized  # noqa: E402

K = 126
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def main():
    w = load_weights(ROOT / "experiments" / "cal" / "calibrators" / "calibrator_A.json")
    json_scale = float(json.loads(
        (ROOT / "experiments" / "cal" / "calibrators" / "calibrator_A.json").read_text()
    )["scale"]["value"])

    out = {}
    for s in SUBSETS:
        d = np.load(ROOT / "experiments" / "cal" / "audit" / f"{s}_A.npz")
        y = d["y"].astype(int)
        mu, sd = forward(w, d["yhat"], 125.0)

        rec = {"n": int(len(y)), "layers": {}}
        for name, p, sc in (
                ("deployed_LocationScale", d["p_cens"], float(d["scale_cens"])),
                ("json_CalibNet", pmf_censored(mu, sd * json_scale, K), json_scale)):
            rnd = coverage_profile_randomized(p, y, n_rep=200)
            itg = coverage_profile(p, y)
            rec["layers"][name] = dict(
                scale=round(sc, 3),
                cov90_rand=round(float(rnd[2]), 4), cov90_int=round(float(itg[2]), 4),
                cov50_rand=round(float(rnd[0]), 4), cov50_int=round(float(itg[0]), 4),
                claim90=round(float(np.mean([
                    p[i, int(np.searchsorted(np.cumsum(p[i]), 0.05)):
                         min(int(np.searchsorted(np.cumsum(p[i]), 0.95)), K - 1) + 1].sum()
                    for i in range(len(p))])), 4))
        a, b = rec["layers"]["deployed_LocationScale"], rec["layers"]["json_CalibNet"]
        rec["delta_90"] = round(a["cov90_rand"] - b["cov90_rand"], 4)
        out[s] = rec
        print(f"{s} deployed {a['cov90_rand']:.4f} / json {b['cov90_rand']:.4f} "
              f"delta {rec['delta_90']:+.4f}", flush=True)

    (ROOT / "experiments" / "calx" / "provenance.json").write_text(json.dumps(out, indent=1))
    print("wrote experiments/calx/provenance.json")


if __name__ == "__main__":
    main()
