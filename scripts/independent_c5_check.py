"""Independent recomputation of C5 (per-engine coverage reproducibility).

Deliberately does NOT import engine_coverage_stability.py, paper_evidence.py or
freeze_audit.py: the protocol is re-implemented from its written definition
(docs/BASELINE_FREEZE.md 3.4 + CLAIM_LEDGER C5), using a different code path
(pandas merge/groupby rather than the numpy loops in the original).

Checks
  ICC(1,1) of per-engine coverage with engines as groups and seeds as replicates
  mean pairwise Spearman across seeds (within subset x fold)
  mean Jaccard overlap of the bottom-quartile engine set across seeds
  within-engine standard deviation across seeds

Usage: python scripts/independent_c5_check.py
Exit 0 iff the recomputed values match the recorded ones within tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [42, 7, 2024, 13]
RECORDED = {"icc": 0.780, "rho": 0.762, "jaccard": 0.599, "seed_sd": 0.032}
TOL = {"icc": 0.005, "rho": 0.005, "jaccard": 0.005, "seed_sd": 0.002}


def split_engines(units: np.ndarray, frac: float = 0.3, seed: int = 42):
    """Engine-disjoint calibration/test split, re-implemented from the definition."""
    uniq = sorted(set(units.tolist()))
    order = np.random.default_rng(seed).permutation(len(uniq))
    n_cal = max(1, int(round(len(uniq) * frac)))
    calib = {uniq[i] for i in order[:n_cal]}
    test = {uniq[i] for i in order[n_cal:]}
    return calib, test


def cell_coverage(root: Path, subset: str, fold: int, seed: int) -> pd.Series | None:
    p = root / "experiments/punet_matrix" / subset / f"f{fold}_s{seed}" / "preds_cqr.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    calib, test = split_engines(df["unit"].to_numpy())
    cal = df[df.unit.isin(calib)]
    tst = df[df.unit.isin(test)].copy()
    scores = np.maximum(cal["y_lo"] - cal["y_true"], cal["y_true"] - cal["y_hi"])
    n = len(scores)
    level = min(1.0, 0.9 * (1.0 + 1.0 / n))
    q = float(np.quantile(scores.to_numpy(), level, method="higher"))
    tst["covered"] = ((tst.y_true >= tst.y_lo - q) & (tst.y_true <= tst.y_hi + q)).astype(float)
    return tst.groupby("unit")["covered"].mean()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else np.nan


def main() -> int:
    root = Path(".")
    iccs, rhos, jacs, sds, sds_sample = [], [], [], [], []
    for sub in SUBSETS:
        for fold in range(5):
            cols = {}
            for seed in SEEDS:
                s = cell_coverage(root, sub, fold, seed)
                if s is not None:
                    cols[seed] = s
            if len(cols) < 2:
                continue
            mat = pd.DataFrame(cols)                     # index=engine, columns=seed
            k = mat.shape[1]
            n = mat.shape[0]
            grand = mat.to_numpy().mean()
            eng_means = mat.mean(axis=1).to_numpy()
            ssb = k * ((eng_means - grand) ** 2).sum()
            ssw = ((mat.to_numpy() - eng_means[:, None]) ** 2).sum()
            msb = ssb / (n - 1)
            msw = ssw / (n * (k - 1))
            iccs.append((msb - msw) / (msb + (k - 1) * msw))
            # Two conventions differ by sqrt(k/(k-1)); the recorded value uses the
            # population SD (ddof=0, numpy default), so we compute both and compare
            # against ddof=0 while reporting ddof=1 for transparency.
            sds.append(float(mat.std(axis=1, ddof=0).mean()))
            sds_sample.append(float(mat.std(axis=1, ddof=1).mean()))

            seeds = list(mat.columns)
            for a in range(len(seeds)):
                for b in range(a + 1, len(seeds)):
                    rhos.append(spearman(mat[seeds[a]].to_numpy(), mat[seeds[b]].to_numpy()))
                    va, vb = mat[seeds[a]].to_numpy(), mat[seeds[b]].to_numpy()
                    q = max(1, int(round(n * 0.25)))
                    sa = set(np.argsort(va)[:q])
                    sb = set(np.argsort(vb)[:q])
                    jacs.append(len(sa & sb) / len(sa | sb))

    got = dict(icc=float(np.mean(iccs)), rho=float(np.nanmean(rhos)),
               jaccard=float(np.mean(jacs)), seed_sd=float(np.mean(sds)))
    ok = True
    print(f"{'statistic':<10} {'recorded':>9} {'recomputed':>11} {'tol':>6}  result")
    for k, v in got.items():
        passed = abs(v - RECORDED[k]) <= TOL[k]
        ok &= passed
        print(f"{k:<10} {RECORDED[k]:>9.3f} {v:>11.3f} {TOL[k]:>6.3f}  {'PASS' if passed else 'FAIL'}")
    print(f"\ncells: {len(iccs)} subset x fold units; seeds: {len(SEEDS)}")
    print(f"within-engine SD: population (ddof=0, recorded convention) = {np.mean(sds):.4f}; "
          f"sample (ddof=1) = {np.mean(sds_sample):.4f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
