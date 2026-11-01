"""Freeze the fitted calibration layer into a self-contained record.

A 66-parameter layer is fully determined by its weights plus the scale chosen on
cross-fitted development data. Persisting them means the official-test metrics can be
recomputed from committed artefacts alone (dev predictions + this JSON) without touching
a checkpoint, a GPU, or the fitting procedure's random seed.

Two things this record deliberately pins down:

* `scale` and `target` -- so a later run cannot silently re-tune the nominal level.
* `dev_sha256` -- a digest of the exact development rows the weights saw. If the dev
  predictions are regenerated with different preprocessing, this digest changes and the
  record is visibly stale rather than quietly wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))

from calibrate import K, MAX_RUL, choose_scale, fit, load_dev  # noqa: E402


def digest(dev):
    h = hashlib.sha256()
    for u, y, yh in zip(dev["unit"], dev["y"], dev["yhat"]):
        h.update(f"{int(u)}:{float(y):.6f}:{float(yh):.6f}\n".encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(ROOT))
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cal" / "calibrators"))
    a = ap.parse_args()

    subsets = a.subsets.split(",")
    devs = {s: load_dev(Path(a.stage), s, a.arm) for s in subsets}
    pooled = {k: np.concatenate([d[k] for d in devs.values()]) for k in ("unit", "yhat", "y")}

    net = fit(pooled)
    scale, gap = choose_scale(pooled)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "arm": a.arm,
        "subsets": subsets,
        "protocol": "pooled",
        "architecture": {
            "hidden": 16, "activation": "tanh", "nu": net.nu,
            "mu": "yhat + 20*out0",
            "sigma": "softplus(clamp(out1,-4,3))*20 + 1",
            "bins": f"0..{K - 1} (last bin is the >= {int(MAX_RUL)} tail)",
        },
        "fit": {
            "objective": "right-censored NLL, ridge 1e-3",
            "epochs": 400, "lr": 0.05, "optimizer": "Adam", "torch_seed": 0,
            "n_rows": int(len(pooled["y"])),
            "n_by_subset": {s: int(len(d["y"])) for s, d in devs.items()},
            "dev_sha256": digest(pooled),
        },
        "scale": {"value": float(scale), "target_cover90": 0.90,
                  "grid": [0.7, 2.5, 37], "rule": "smallest scale reaching target",
                  "dev_crossfit_gap": float(gap)},
        "weights": {k: v.detach().cpu().numpy().tolist() for k, v in net.state_dict().items()
                    if k != "bins"},
        "n_params": int(sum(p.numel() for p in net.parameters() if p.requires_grad)),
    }
    f = out / f"calibrator_{a.arm}.json"
    f.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("saved", f, "params", rec["n_params"], "scale %.2f" % scale, "gap %.3f" % gap)


if __name__ == "__main__":
    main()
