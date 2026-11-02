"""How stable is a band table? Resample the engines and watch the conclusions move.

A coverage table is a point estimate of five proportions. The question a reader actually
has is "which band is worst, and is it bad?" -- a question about the *argmin*, which is
far less stable than any single proportion. This script reports, per cell:

  argmin_agree   fraction of bootstrap replicates whose worst band is the sample's worst
  argmin_share   the full distribution over which band comes out worst
  picp_sd        bootstrap sd of the worst band's coverage
  rank_flip      fraction of replicates in which the band ordering changes at all

and then the same for the four C-MAPSS fleets pooled, where n is 5x larger. If pooling
stabilises the answer, the instability is a sample-size fact, not a property of the model.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import cover_indicator  # noqa: E402

BANDS_125 = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
BANDS_100 = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]


def band_picp(y, e, bands):
    out = []
    for lo, hi in bands:
        m = (y >= lo) & (y <= hi) if hi > y.max() else (y >= lo) & (y < hi)
        out.append(float(e[m].mean()) if m.any() else np.nan)
    return np.array(out)


def stability(p, y, bands, n_boot=2000, seed=0):
    e, _, _ = cover_indicator(p, y, 0.90)
    base = band_picp(y, e, bands)
    worst = int(np.nanargmin(base))
    rng = np.random.default_rng(seed)
    n = len(y)
    picks, ranks, vals = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bp = band_picp(y[idx], e[idx], bands)
        if np.isnan(bp).all():
            continue
        picks.append(int(np.nanargmin(bp)))
        ranks.append(tuple(np.argsort(np.argsort(bp))))
        vals.append(bp[worst])
    picks = np.array(picks)
    vals = np.array(vals)
    share = np.bincount(picks, minlength=len(bands)) / len(picks)
    same_rank = np.mean([r == ranks[0] for r in ranks])
    lo, hi = np.percentile(vals, [5, 95])
    return dict(base=[float(v) for v in base], worst_band=worst,
                argmin_agree=float(share[worst]), argmin_share=[float(v) for v in share],
                rank_flip=float(1.0 - same_rank), n_boot=int(len(picks)),
                worst_sd=float(vals.std()), worst_ci90=[float(lo), float(hi)],
                worst_ci90_width=float(hi - lo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="experiments/cal/audit/FD*_A.npz")
    ap.add_argument("--bands", default="125")
    ap.add_argument("--out", default="experiments/cal/audit/stability.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    bands = BANDS_125 if a.bands == "125" else BANDS_100
    files = sorted(glob.glob(a.glob))
    res = {}
    print(f"{'cell':12s} {'n':>5s} {'worst':>10s} {'picp':>5s} {'sd':>5s} "
          f"{'ci90w':>6s} {'agree':>6s} {'flip':>6s}")
    for f in files:
        d = np.load(f)
        key = Path(f).stem
        r = stability(d["p_cens"], d["y"], bands, a.n_boot)
        r["n"] = int(len(d["y"]))
        res[key] = r
        print(f"{key:12s} {r['n']:5d} {bands[r['worst_band']]!s:>10s} "
              f"{r['base'][r['worst_band']]:5.2f} {r['worst_sd']:5.3f} "
              f"{r['worst_ci90_width']:6.3f} {r['argmin_agree']:6.2f} {r['rank_flip']:6.2f}")

    # pooled over fleets: same band edges, 5x the engines
    ps = [np.load(f)["p_cens"] for f in files]
    ys = [np.load(f)["y"] for f in files]
    if all(pp.shape[1] == ps[0].shape[1] for pp in ps):
        pcat, ycat = np.concatenate(ps), np.concatenate(ys)
        r = stability(pcat, ycat, bands, a.n_boot)
        r["n"] = int(len(ycat))
        res["POOLED"] = r
        print(f"{'POOLED':12s} {r['n']:5d} {bands[r['worst_band']]!s:>10s} "
              f"{r['base'][r['worst_band']]:5.2f} {r['worst_sd']:5.3f} "
              f"{r['worst_ci90_width']:6.3f} {r['argmin_agree']:6.2f} {r['rank_flip']:6.2f}")

    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
