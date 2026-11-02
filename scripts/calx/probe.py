"""Exploratory probe: does the layer's *location* head fix the conformal over-coverage?

Labelled exploratory under paper/CVPLUS_PROTOCOL.md section 6 -- it is not part of the
pre-registered four rows. The pre-registered rows over-cover by 0.06-0.08 at 90% and
0.25-0.34 at 50%, and the obvious suspect is the centring: the conformal rows centre on
the backbone's raw prediction while the frozen layer shifts it by up to 20 RUL units.
This script separates centre from scale.

For each subset and seed, five variants of the same cross-fitted construction:
  raw        centre = fold prediction                 score = |r|
  raw/sig    centre = fold prediction                 score = |r| / sigma
  mu         centre = layer location                  score = |r|
  mu/sig     centre = layer location                  score = |r| / sigma
  mu(oracle) centre = layer location, width scaled on TEST so coverage = 0.90 (headroom)
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts" / "cal"))
from npforward import forward as np_forward, load_weights  # noqa: E402

MAX_RUL, K = 125.0, 126
LEVELS = (0.90, 0.50)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEEDS = (2027, 2028, 2029)


def read_csv(p):
    us, ys, ps = [], [], []
    for r in csv.DictReader(open(p)):
        us.append(int(r.get("unit_id", r.get("unit"))))
        ys.append(float(r["y_true"]))
        ps.append(float(r["y_pred"]))
    o = np.argsort(us)
    return np.array(us)[o], np.array(ys)[o], np.array(ps)[o]


def load_set(root):
    out = {}
    for f in sorted(root.glob("A_s*_f*.csv")):
        a, b = f.stem.split("_")[1], f.stem.split("_")[2]
        out[(int(a[1:]), int(b[1:]))] = read_csv(f)
    return out


def wq(v, q):
    return float(np.quantile(v, q, method="lower"))


def main():
    w = load_weights(ROOT / "experiments" / "cal" / "calibrators" / "calibrator_A.json")
    loc = lambda v: np_forward(w, np.asarray(v, float), MAX_RUL)

    rep = {}
    for s in SUBSETS:
        dev = load_set(ROOT / "experiments" / "cal" / "dev_preds" / s)
        tst = load_set(ROOT / "experiments" / "calx" / "fold_test" / s)
        d = np.load(ROOT / "experiments" / "cal" / "audit" / f"{s}_A.npz")
        y = d["y"].astype(float)
        rows = {}
        for seed in SEEDS:
            dy, dh, df = [], [], []
            for f in range(5):
                u, yy, pp = dev[(seed, f)]
                dy.append(yy); dh.append(pp); df.append(np.full(len(u), f))
            dy, dh = np.concatenate(dy), np.concatenate(dh)
            mu_d, sg_d = loc(dh)
            mu_t, sg_t = loc(np.mean([tst[(seed, f)][2] for f in range(5)], axis=0))
            mu_k = np.stack([loc(tst[(seed, f)][2])[0] for f in range(5)], axis=1)
            raw_k = np.stack([tst[(seed, f)][2] for f in range(5)], axis=1)

            for name, r, scale_t in (
                    ("raw", dy - dh, np.ones(len(y))),
                    ("raw/sig", (dy - dh) / sg_d, sg_t),
                    ("mu", dy - mu_d, np.ones(len(y))),
                    ("mu/sig", (dy - mu_d) / sg_d, sg_t)):
                cen = mu_k if name.startswith("mu") else raw_k
                for lv in LEVELS:
                    al = (1 - lv) / 2
                    lo = np.array([wq(cen[i][:, None] - np.abs(r)[None, :] * scale_t[i], al)
                                   for i in range(len(y))])
                    hi = np.array([wq(cen[i][:, None] + np.abs(r)[None, :] * scale_t[i], 1 - al)
                                   for i in range(len(y))])
                    rows.setdefault(f"{name}@{lv:.2f}", []).append(
                        (float(np.mean((lo <= y) & (y <= hi))), float(np.mean(hi - lo))))
            # headroom: shrink the mu-centred interval on test until 0.90
            base_lo = np.array([wq(mu_k[i][:, None] - np.abs(dy - mu_d)[None, :], 0.05) for i in range(len(y))])
            base_hi = np.array([wq(mu_k[i][:, None] + np.abs(dy - mu_d)[None, :], 0.95) for i in range(len(y))])
            f = lambda k: float(np.mean((mu_t - k * (mu_t - base_lo) <= y) & (y <= mu_t + k * (base_hi - mu_t)))) - 0.90
            a_, b_ = 0.05, 2.0
            for _ in range(60):
                m = 0.5 * (a_ + b_)
                a_, b_ = (m, b_) if f(m) < 0 else (a_, m)
            k = 0.5 * (a_ + b_)
            rows.setdefault("mu@0.90", []).append((float(np.mean((mu_t - k * (mu_t - base_lo) <= y)
                                                                 & (y <= mu_t + k * (base_hi - mu_t)))),
                                                  float(np.mean(k * (base_hi - base_lo)))))
        rep[s] = {k: [round(float(np.mean([t[0] for t in v])), 3), round(float(np.mean([t[1] for t in v])), 2)]
                  for k, v in rows.items()}
        print(s, {k: v for k, v in rep[s].items()}, flush=True)

    out = ROOT / "experiments" / "calx" / "probe.json"
    out.write_text(json.dumps(rep, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
