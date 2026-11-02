"""Dump per-sample predictive parameters from the frozen layers, torch vs numpy.

Two reasons this runs on the GPU box rather than locally:
  1. `layer.py` is torch, and the analysis machine has no torch.
  2. It doubles as the cross-implementation check: the same weights are pushed through
     `layer.LocationScale` and through `npforward.forward`, and the max abs difference
     is reported. If they disagree the analysis is built on sand, so the script refuses
     to write output in that case.

Output: one npz per (subset, arm) with unit / y / yhat / mu / sigma, plus the 80/50
interval endpoints, so downstream work needs no further model evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer as L                      # noqa: E402
from npforward import forward as np_forward, load_weights  # noqa: E402

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
ARMS = ["A", "BE", "PE"]


def read_preds(path, cap):
    import csv
    us, ps, ys = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            us.append(int(r["unit"]))
            ps.append(float(r["y_pred"]))
            ys.append(min(float(r["y_true"]), cap))
    return np.array(us), np.array(ys, float), np.array(ps, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="dir holding results/dsh/e21/preds")
    ap.add_argument("--cal", required=True, help="dir holding calibrator_*.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=125.0)
    args = ap.parse_args()

    L.set_grid(int(args.cap))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for arm in ARMS:
        cal = json.loads((Path(args.cal) / f"calibrator_{arm}.json").read_text())
        net = L.LocationScale(kind=cal["architecture"].get("kind", "mlp"),
                             hidden=cal["architecture"]["hidden"]).double()
        net.load_state_dict({k: torch.tensor(v) for k, v in cal["weights"].items()})
        net.eval()
        wnp = load_weights(Path(args.cal) / f"calibrator_{arm}.json")
        scale = float(cal["scale"]["value"])

        for s in SUBSETS:
            f = Path(args.stage) / "results" / "dsh" / "e21" / "preds" / f"{s}_{arm}.csv"
            if not f.exists():
                continue
            unit, y, yhat = read_preds(f, args.cap)
            with torch.no_grad():
                mu_t, sg_t = net.params(torch.tensor(yhat, dtype=torch.float64))
            mu_t, sg_t = mu_t.numpy(), sg_t.numpy()
            mu_n, sg_n = np_forward(wnp, yhat, args.cap)
            dmu = float(np.abs(mu_t - mu_n).max())
            dsg = float(np.abs(sg_t - sg_n).max())

            p = L.pmf_np(mu_t, sg_t * scale, "censored")
            lo90, hi90 = L.interval(p, 0.90)
            lo80, hi80 = L.interval(p, 0.80)
            lo50, hi50 = L.interval(p, 0.50)
            np.savez_compressed(
                out / f"{s}_{arm}.npz",
                unit=unit, y=y, yhat=yhat, mu=mu_t, sigma=sg_t, scale=scale,
                lo90=lo90, hi90=hi90, lo80=lo80, hi80=hi80, lo50=lo50, hi50=hi50)
            report[f"{s}_{arm}"] = dict(n=len(y), mu_dmax=dmu, sigma_dmax=dsg,
                                        cover90=float(np.mean((lo90 <= y) & (y <= hi90))),
                                        cover80=float(np.mean((lo80 <= y) & (y <= hi80))),
                                        cover50=float(np.mean((lo50 <= y) & (y <= hi50))),
                                        scale=scale)

    worst = max(max(v["mu_dmax"], v["sigma_dmax"]) for v in report.values())
    (out / "check.json").write_text(json.dumps(
        {"worst_abs_diff": worst, "per_cell": report}, indent=1))
    print(json.dumps({"worst_abs_diff": worst}, indent=1))
    for k, v in sorted(report.items()):
        print(f"{k:12s} n={v['n']:4d} c90={v['cover90']:.4f} c80={v['cover80']:.4f} "
              f"c50={v['cover50']:.4f} |dmu|={v['mu_dmax']:.2e} |dsg|={v['sigma_dmax']:.2e}")
    if worst > 1e-8:
        raise SystemExit(f"torch/numpy mismatch {worst:.3e} -- refusing to emit endpoints")


if __name__ == "__main__":
    main()
