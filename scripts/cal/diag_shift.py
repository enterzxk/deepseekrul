"""Diagnose the dev -> test shift that breaks FD003 coverage.

Everything here is descriptive: it reads the out-of-fold dev predictions and the
official test predictions, and reports where the calibrated layer loses coverage.
No parameter is tuned on test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
MAX_RUL = 125
K = MAX_RUL + 1

sys.path.insert(0, str(ROOT / "scripts" / "cal"))
from calibrate import CalibNet, cross_fit, fit, load_dev, load_test, metrics, pmf  # noqa: E402


def band_table(p, y, yhat, edges=(0, 25, 50, 75, 100, 125)):
    cdf = np.cumsum(p, axis=1)
    rows = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = (y >= lo_e) & (y < hi_e) if hi_e < 125 else (y >= lo_e) & (y <= hi_e)
        if m.sum() < 5:
            continue
        lo = np.array([np.searchsorted(cdf[i], 0.05) for i in np.where(m)[0]])
        hi = np.array([min(np.searchsorted(cdf[i], 0.95), K - 1) for i in np.where(m)[0]])
        rows.append(dict(band=f"[{lo_e},{hi_e})", n=int(m.sum()),
                         rmse=float(np.sqrt(np.mean((yhat[m] - y[m]) ** 2))),
                         bias=float(np.mean(yhat[m] - y[m])),
                         resid_sd=float(np.std(y[m] - yhat[m])),
                         cover90=float(np.mean((lo <= y[m]) & (y[m] <= hi))),
                         width90=float(np.mean(hi - lo))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(ROOT))
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    a = ap.parse_args()
    stage = Path(a.stage)

    out = {}
    for subset in a.subsets.split(","):
        dev = load_dev(stage, subset)
        te = load_test(stage, subset)
        net = fit(dev)
        p_te = pmf(net, te["yhat"])
        p_cv = cross_fit(dev)

        # 1) how different are the two prediction populations?
        shift = dict(
            dev_yhat_mean=float(dev["yhat"].mean()), test_yhat_mean=float(te["yhat"].mean()),
            dev_yhat_sd=float(dev["yhat"].std()), test_yhat_sd=float(te["yhat"].std()),
            dev_resid_sd=float(np.std(dev["y"] - dev["yhat"])),
            test_resid_sd=float(np.std(te["y"] - te["yhat"])),
            dev_bias=float(np.mean(dev["y"] - dev["yhat"])),
            test_bias=float(np.mean(te["y"] - te["yhat"])),
        )
        shift["resid_sd_ratio"] = shift["test_resid_sd"] / max(shift["dev_resid_sd"], 1e-6)

        # 2) where coverage is lost on test
        bands = band_table(p_te, te["y"], te["yhat"])

        # 3) what a dev-only scale rule would have done: inflate sigma by the
        #    dev-estimated std of the standardised residual
        cdf_dev = np.cumsum(p_cv, axis=1)
        # PIT of the true label under the cross-fitted predictive CDF
        pit = np.array([cdf_dev[i, int(min(dev["y"][i], K - 1))] for i in range(len(dev["y"]))])
        out[subset] = dict(shift=shift, bands=bands,
                           dev_pit=dict(q10=float(np.quantile(pit, 0.10)),
                                        q50=float(np.quantile(pit, 0.50)),
                                        q90=float(np.quantile(pit, 0.90)),
                                        below05=float(np.mean(pit < 0.05)),
                                        above95=float(np.mean(pit > 0.95))),
                           dev=metrics(p_cv, dev["y"], dev["yhat"]),
                           test=metrics(p_te, te["y"], te["yhat"]))

    print(json.dumps(out, indent=1))
    (ROOT / "experiments" / "cal" / "shift_report.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
