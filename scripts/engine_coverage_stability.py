"""Is per-engine coverage failure a stable property of the engine?

D018 asks whether calibration-only information can identify the engines whose
intervals will fail.  That question presupposes the failing set is a property of
the engine.  Here we test the presupposition with the frozen artefacts only: the
same (subset, fold) has four training seeds that share one engine split, so any
disagreement across seeds about *which* engines fail is pure training randomness
and bounds what any calibration-only predictor could ever achieve.

Outputs (stdout + optional JSON):
  - mean pairwise Spearman correlation of per-engine coverage across seeds
  - overlap of the bottom-quartile engine set across seeds (Jaccard)
  - variance decomposition of per-engine coverage: engine / seed / interaction
  - the same for the *binary* event "engine coverage < 0.80"

No training, no new data, no model selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]
FOLDS = list(range(5))


def engine_split(units, frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    u = np.unique(units).copy()
    rng.shuffle(u)
    n = max(1, int(round(u.size * frac)))
    return np.isin(units, u[:n]), np.isin(units, u[n:])


def cqr_interval(y, lo, hi, cal, tst, delta=0.1):
    s = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])
    q = float(np.quantile(s, min(1.0, (1 - delta) * (1 + 1 / s.size)), method="higher"))
    return lo[tst] - q, hi[tst] + q


def per_engine_coverage(root: Path, subset: str, fold: int, seed: int) -> pd.Series:
    d = root / "experiments/punet_matrix" / subset / f"f{fold}_s{seed}"
    c = np.genfromtxt(d / "preds_cqr.csv", delimiter=",", names=True)
    y, lo, hi = c["y_true"], c["y_lo"], c["y_hi"]
    cal, tst = engine_split(c["unit"])
    L, H = cqr_interval(y, lo, hi, cal, tst)
    covered = (y[tst] >= L) & (y[tst] <= H)
    return pd.Series(covered.astype(float)).groupby(pd.Series(c["unit"][tst])).mean()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def jaccard_bottom(x: np.ndarray, y: np.ndarray, q=0.25) -> float:
    k = max(1, int(round(len(x) * q)))
    sx = set(np.argsort(x)[:k])
    sy = set(np.argsort(y)[:k])
    return len(sx & sy) / len(sx | sy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    rows, pair_rows, var_rows = [], [], []
    for sub in SUBSETS:
        for fold in FOLDS:
            cov = {}
            for seed in SEEDS:
                p = root / "experiments/punet_matrix" / sub / f"f{fold}_s{seed}" / "preds_cqr.csv"
                if p.exists():
                    cov[seed] = per_engine_coverage(root, sub, fold, seed)
            if len(cov) < 2:
                continue
            seeds = sorted(cov)
            idx = cov[seeds[0]].index
            M = np.vstack([cov[s].reindex(idx).to_numpy() for s in seeds])  # seeds x engines
            for a, b in itertools.combinations(range(len(seeds)), 2):
                pair_rows.append(dict(subset=sub, fold=fold, s1=seeds[a], s2=seeds[b],
                                      spearman=spearman(M[a], M[b]),
                                      jaccard_bottom=jaccard_bottom(M[a], M[b])))
            # one-way random-effects ICC(1,1): engines are groups, seeds are
            # replicates.  ICC = share of per-engine-coverage variance that is
            # attributable to engine identity rather than training seed.
            k = M.shape[0]
            n = M.shape[1]
            grand = M.mean()
            eng = M.mean(axis=0)          # per-engine mean over seeds
            ss_eng = k * ((eng - grand) ** 2).sum()
            ss_within = ((M - eng[None, :]) ** 2).sum()
            msb = ss_eng / max(n - 1, 1)
            msw = ss_within / max(n * (k - 1), 1)
            icc = (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) > 0 else float("nan")
            var_rows.append(dict(subset=sub, fold=fold, n_engines=n,
                                 icc_engine=float(icc),
                                 seed_sd=float(M.std(axis=0).mean()),
                                 mean_cov=float(grand),
                                 min_engine_mean=float(eng.min())))
            rows.append(dict(subset=sub, fold=fold, **{f"s{s}": float(cov[s].min()) for s in seeds}))

    pr = pd.DataFrame(pair_rows)
    vr = pd.DataFrame(var_rows)
    print("=== pairwise seed agreement of per-engine coverage ===")
    g = pr.groupby("subset").agg(n_pairs=("spearman", "size"),
                                 mean_rho=("spearman", "mean"),
                                 min_rho=("spearman", "min"),
                                 max_rho=("spearman", "max"),
                                 mean_jaccard_bottom=("jaccard_bottom", "mean"))
    print(g.round(3).to_string())
    print(f"\npooled: mean rho {pr.spearman.mean():+.3f} | median {pr.spearman.median():+.3f} | "
          f"share of pairs with rho>0 {100*(pr.spearman>0).mean():.0f}% | "
          f"mean bottom-quartile Jaccard {pr.jaccard_bottom.mean():.3f}")
    print("\n=== variance decomposition of per-engine coverage (within subset x fold) ===")
    print("ICC(1,1) = share of per-engine-coverage variance attributable to engine identity")
    print(vr.groupby("subset")[["icc_engine", "seed_sd"]].mean().round(3).to_string())
    print("\npooled: mean ICC %.3f | mean within-engine seed SD %.4f" % (
        vr.icc_engine.mean(), vr.seed_sd.mean()))
    print("\n=== per (subset, fold): engine count, mean coverage, worst engine (seed-averaged) ===")
    print(vr[["subset", "fold", "n_engines", "mean_cov", "min_engine_mean"]].round(3).to_string(index=False))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"pairs": pr.to_dict("records"), "variance": vr.to_dict("records")}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
