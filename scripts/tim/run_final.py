"""The main table, rebuilt on one protocol.

Every construction here deploys the same 15-model mean and calibrates on one score per
engine; the only things that vary are the calibration object (`cv+` vs `matched`), the
score sign, the tail allocation and the variance normalisation.  No seed loop: the seeds
enter through the calibration sample, and the deployment is a single fixed object, so a
row of this table is one interval method rather than an average over three of them.

Two tail variants are separated that the earlier version merged:

  free        gamma chosen on the whole calibration sample, by empirical width
  free-split  gamma chosen on a random half of the calibration ENGINES and the interval
              built on the other half, so gamma is independent of the order statistics
              it is applied to and the finite-sample rank bound applies to the final
              interval as stated
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core as C
import protocol as P

OUT = C.ROOT / "experiments" / "tim"
LEVELS = tuple(float(x) for x in os.environ.get("RUL_LEVELS", "0.90,0.50").split(","))
TAG = os.environ.get("RUL_PRED_TAG", "A")


def out_name():
    base = "final" if TAG == "A" else f"final_{TAG}"
    if LEVELS == (0.90, 0.50):
        return OUT / f"{base}.json"
    lv = "_".join(f"{x:.2f}".replace("0.", "") for x in LEVELS)
    return OUT / f"{base}_lv{lv}.json"


CONSTRUCTIONS = (
    "layer-central", "layer-shortest",
    "cv+ sym", "cv+ equal tails",
    "matched sym", "matched equal tails", "matched free", "matched free-split",
    "matched free, no normalisation",
)


def run_subset(subset):
    Pd, y, tst = P.deployed(subset)
    n = len(y)
    rows = {}

    # ---- parametric layer, both interval types, integer-endpoint coverage for each
    root = Path(os.environ.get("RUL_PRED_ROOT", C.ROOT / "experiments"))
    layer = np.load(root / "cal" / "audit" / f"{subset}_{TAG}.npz")
    for lv in LEVELS:
        lo_i, hi_i = C.interval_from_pmf(layer["p_cens"], lv)
        lo_s, hi_s = C.shortest_from_pmf(layer["p_cens"], lv)
        for name, (lo, hi) in (("layer-central", (lo_i, hi_i)),
                               ("layer-shortest", (lo_s, hi_s))):
            lo = lo.astype(float); hi = hi.astype(float)
            rows[f"{name}@{lv:.2f}"] = dict(
                cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                winkler=C.winkler(lo, hi, y, lv),
                cov_mass=C.interval_mass(layer["p_cens"], lo.astype(int), hi.astype(int)),
                n=n)
        rows[f"layer-pit@{lv:.2f}"] = dict(
            cov=C.randomized_pit_cov(layer["p_cens"], y.astype(int), lv), n=n)

    # ---- conformal arms
    arms = {m: P.arm(subset, mode=m, normalise=True) for m in ("cv+", "matched")}
    arms["matched-raw"] = P.arm(subset, mode="matched", normalise=False)

    # one fixed half-split of the calibration ENGINES, shared by every fleet so the
    # split is not re-drawn per level (which would be another selection step)
    u = arms["matched"]["unit"]
    rng = np.random.default_rng(11)
    pick = rng.permutation(np.unique(u))
    half = set(pick[: len(pick) // 2].tolist())
    in_a = np.array([x in half for x in u])

    for lv in LEVELS:
        alpha = 1.0 - lv
        for tag, key, sym in (("cv+ sym", "cv+", True),
                              ("cv+ equal tails", "cv+", False),
                              ("matched sym", "matched", True),
                              ("matched equal tails", "matched", False)):
            a = arms[key]
            lo, hi = P.interval(Pd, a["z"], lv, symmetric=sym, sigma=a["sigma"])
            rows[f"{tag}@{lv:.2f}"] = dict(cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                                           winkler=C.winkler(lo, hi, y, lv), n=n)

        a = arms["matched"]
        g = P.pick_gamma(a["z"], alpha)
        lo, hi = P.interval(Pd, a["z"], lv, gamma=g, sigma=a["sigma"])
        rows[f"matched free@{lv:.2f}"] = dict(cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                                              winkler=C.winkler(lo, hi, y, lv), gamma=g, n=n)

        # gamma from half the engines, interval from the other half
        z_sel, z_fit = a["z"][in_a], a["z"][~in_a]
        g2 = P.pick_gamma(z_sel, alpha)
        lo, hi = C.split_conformal(Pd, z_fit, z_fit, g2, alpha - g2, scale_test=a["sigma"](Pd))
        rows[f"matched free-split@{lv:.2f}"] = dict(cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
                                                    winkler=C.winkler(lo, hi, y, lv),
                                                    gamma=g2, n_cal=int((~in_a).sum()), n=n)

        ra = arms["matched-raw"]
        gr = P.pick_gamma(ra["z"], alpha)
        lo, hi = P.interval(Pd, ra["z"], lv, gamma=gr)
        rows[f"matched free, no normalisation@{lv:.2f}"] = dict(
            cov=C.coverage(lo, hi, y), width=C.width(lo, hi),
            winkler=C.winkler(lo, hi, y, lv), gamma=gr, n=n)

    rows["_deploy"] = dict(n=n, rmse=float(np.sqrt(np.mean((y - Pd) ** 2))),
                           n_cal_cv=len(arms["cv+"]["r"]), n_cal_matched=len(arms["matched"]["r"]))
    return rows


def main():
    rep = {"levels": list(LEVELS), "protocol": "protocol.py", "subsets": {}}
    for s in C.SUBSETS:
        rep["subsets"][s] = run_subset(s)
        d = rep["subsets"][s]["_deploy"]
        print(f"== {s} n={d['n']} rmse {d['rmse']:.2f} cal(cv+) {d['n_cal_cv']} cal(matched) {d['n_cal_matched']}")
        for k, v in rep["subsets"][s].items():
            if k.startswith("_"):
                continue
            w = f"{v['width']:6.1f}" if v.get("width") else "  -   "
            print(f"   {k:38s} cov {v['cov']:.3f}  w {w}  W {v.get('winkler', float('nan')):5.1f}")
    dst = out_name()
    dst.write_text(json.dumps(rep, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
