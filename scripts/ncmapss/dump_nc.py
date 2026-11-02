"""Dump the N-CMAPSS pooled-protocol pmf and its (mu, sigma) for the coverage audit.

Same contract as `scripts/cal/dump_endpoints.py` for C-MAPSS: the auditor needs, per test
sample, the predictive pmf and the two parameters behind it. The parameters are what let
the power simulation tilt the generator without touching the model under audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cal"))

import cal_nc as C           # noqa: E402
import layer as L            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(C.ROOT / "experiments/ncmapss/preds/preds.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    data = C.load_preds(Path(a.preds))
    fleets = sorted(data)
    print(f"{len(fleets)} fleets")
    dev = C.merge([data[f]["dev"] for f in fleets])
    dist, scales, times, s0 = C.fit_all(dev)
    nets = dist.nets
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in fleets:
        te = data[f]["test"]
        yh = te["yhat"]
        p = L.gen_pmf(nets["cens"], yh, scale=scales["cens"])
        with torch.no_grad():
            mu, sg = nets["cens"].params(torch.tensor(yh, dtype=torch.float64))
        mu, sg = mu.numpy(), sg.numpy()
        np.savez_compressed(
            out / f"{f}.npz", fleet=f, unit=te["unit"], y=te["y"], yhat=yh,
            mu=mu, sigma=sg, p_cens=p, scale_cens=scales["cens"])
        e = L.cover(p, te["y"], 0.90)
        print(f"{f:8s} n={len(te['y']):5d} units={len(np.unique(te['unit'])):3d} "
              f"c90={e:.4f}")
    (out / "meta.json").write_text(json.dumps(
        {"scales": scales, "s0": s0, "cap": L.MAX_RUL}, indent=1))


if __name__ == "__main__":
    main()
