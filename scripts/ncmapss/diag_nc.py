"""Is the N-CMAPSS band bias removable by any monotone map of yhat?

H2 in paper/NCMAPSS_PROTOCOL.md. The question does not involve the calibration layer at
all: it asks whether the function class { g : yhat -> R, g monotone } contains some member
whose per-band mean bias is small. The MSE-optimal member is isotonic regression, and it
can be fitted directly on the test sample, which removes any dev-to-test shift from the
comparison. If even that map leaves band bias, no map in the class removes it.

Run locally -- this needs numpy only, no torch, no model.

    python scripts/ncmapss/diag_nc.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]


def pava(y, w=None):
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    v, ww = [], []
    for yi, wi in zip(y, w):
        v.append(yi); ww.append(wi)
        while len(v) > 1 and v[-2] > v[-1]:
            v2, w2 = v.pop(), ww.pop()
            v1, w1 = v.pop(), ww.pop()
            v.append((v1 * w1 + v2 * w2) / (w1 + w2)); ww.append(w1 + w2)
    out, i = np.empty_like(y), 0
    for vi, wi in zip(v, ww):
        n = int(round(wi)); out[i:i + n] = vi; i += n
    return np.maximum.accumulate(out)


def iso_on(x, y, xq):
    """Fit isotonic y~x on (x, y), evaluate at xq. The MSE-optimal monotone map."""
    o = np.argsort(x, kind="mergesort")
    g = pava(y[o])
    ux, idx = np.unique(x[o], return_index=True)
    gx = g[idx]
    j = np.clip(np.searchsorted(ux, xq, side="right") - 1, 0, len(gx) - 1)
    return gx[j]


def band_report(pred, y, name):
    out = {}
    for a, b in BANDS:
        m = (y >= a) & (y < b)
        if m.sum() == 0:
            continue
        out[f"[{a},{b})"] = {"n": int(m.sum()),
                             "bias": round(float((pred[m] - y[m]).mean()), 2),
                             "rmse": round(float(np.sqrt(((pred[m] - y[m]) ** 2).mean())), 2)}
    key = max(out.items(), key=lambda kv: abs(kv[1]["bias"]))
    return {"name": name, "bands": out,
            "worst_band": key[0], "worst_abs_bias": abs(key[1]["bias"]),
            "rmse": round(float(np.sqrt(((pred - y) ** 2).mean())), 2)}


def main():
    rows = list(csv.DictReader(open(ROOT / "experiments/ncmapss/preds/preds.csv")))
    for split in ("test",):
        sel = [r for r in rows if r["split"] == split]
        fleets = sorted({r["fleet"] for r in sel})
        report = {}

        # ---- pooled across fleets ------------------------------------------------
        y = np.array([float(r["y"]) for r in sel])
        yh = np.array([float(r["yhat"]) for r in sel])
        print(f"=== pooled test, n={len(y)}, fleets={len(fleets)} ===")
        variants = [
            band_report(yh, y, "raw backbone"),
            band_report(iso_on(yh, y, yh), y, "isotonic on test (shift-free)"),
            band_report(np.full_like(y, y.mean()), y, "constant (ybar)"),
        ]
        for v in variants:
            print(f"  {v['name']:34s} worst |bias| {v['worst_abs_bias']:6.2f} "
                  f"at {v['worst_band']:10s} RMSE {v['rmse']:6.2f}  "
                  f"{ {k: b['bias'] for k, b in v['bands'].items()} }")

        # shrinkage family: how band bias moves as predictions are pulled to the mean
        print("\n  shrinkage toward the mean (alpha=1 is the raw backbone):")
        print(f"  {'alpha':>6s}{'worst |bias|':>14s}{'RMSE':>8s}   band biases")
        curve = []
        for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            p = y.mean() + alpha * (yh - y.mean())
            v = band_report(p, y, f"alpha={alpha}")
            b = {k: d["bias"] for k, d in v["bands"].items()}
            curve.append({"alpha": alpha, "worst_abs_bias": v["worst_abs_bias"],
                          "rmse": v["rmse"], "bands": b})
            print(f"  {alpha:6.1f}{v['worst_abs_bias']:14.2f}{v['rmse']:8.2f}   {b}")
        report["pooled_curve"] = curve
        report["pooled"] = {v["name"]: v for v in variants}

        # ---- per fleet -----------------------------------------------------------
        print(f"\n=== per fleet (test) ===")
        print(f"{'fleet':7s}{'n':>5s}{'raw worst':>11s}{'iso-test worst':>16s}"
              f"{'raw RMSE':>10s}{'iso RMSE':>10s}")
        per = {}
        for f in fleets:
            s2 = [r for r in sel if r["fleet"] == f]
            yf = np.array([float(r["y"]) for r in s2])
            pf = np.array([float(r["yhat"]) for r in s2])
            r0 = band_report(pf, yf, "raw")            if len(yf) else None
            r1 = band_report(iso_on(pf, yf, pf), yf, "iso") if len(yf) else None
            per[f] = {"raw": r0, "iso_on_test": r1,
                      "y_mean": float(yf.mean()), "yhat_mean": float(pf.mean())}
            print(f"{f:7s}{len(yf):5d}{r0['worst_abs_bias']:11.2f}"
                  f"{r1['worst_abs_bias']:16.2f}{r0['rmse']:10.2f}{r1['rmse']:10.2f}")
        report["per_fleet"] = per

        out = ROOT / "experiments/ncmapss/diag_nc.json"
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
