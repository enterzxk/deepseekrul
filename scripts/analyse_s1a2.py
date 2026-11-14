"""S1-A second half: paired engine-level analysis and the frozen three-tier rule.

Everything here follows ``docs/paper_b_next/S1A2_PUBLIC_PREREG.md`` (v1.2,
``24f02911``):

* engine-level RMSE is recomputed from each cell's ``validation_predictions.csv``;
  ``metrics.json::val_engine_rmse`` is pooled and never read;
* V0 comes from the d7bd9c17 result tree, never re-run;
* the combined effect weights the four subsets equally, not by engine count;
* 95% CI from a stratified engine bootstrap (10,000 draws, RNG 20261114), seeds
  averaged first; Bonferroni 98.33% reported alongside;
* a method "passes" when its combined point estimate is negative, at least 3 of
  4 subsets are negative, and its combined 95% CI excludes 0;
* the three tiers are ``p >= 3`` / ``p < 3 and m >= 3`` / ``m < 3``.

Usage
-----
    python scripts/analyse_s1a2.py --new-roots /home/yndx/cycle_runs,/home/user/cycle_runs \
        --v0-root /home/yndx/paper_b_public_baselines_20261111/paper_b/experiments/\
open_source_reproduction_20261111/runs/unified --json report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("aidgn", "two_phase", "dual_mixer", "ffve", "fcstgnn")
SHAM = ("aidgn", "two_phase", "dual_mixer")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEEDS = (101, 202, 303)
BONFERRONI = 100.0 * (1.0 - 0.05 / 3.0)


def engine_rmse(path):
    frame = pd.read_csv(path)
    squared = (frame["y_pred"].to_numpy(float) - frame["y_true"].to_numpy(float)) ** 2
    work = pd.DataFrame({"unit": frame["unit"].to_numpy(), "se": squared})
    return work.groupby("unit")["se"].mean().pow(0.5).sort_index()


def find_cell(roots, arm, method, subset, seed):
    """V0's tree has no arm level, so an empty ``arm`` means no extra segment."""
    for root in roots:
        base = Path(root) / arm if arm else Path(root)
        path = base / method / subset / f"s{seed}" / "validation_predictions.csv"
        if path.exists():
            return path
    return None


def load(roots, arm, method, subset, seed):
    path = find_cell(roots, arm, method, subset, seed)
    return None if path is None else engine_rmse(path)


def stack(roots, arm, method, subset):
    """(n_seeds, n_engines) matrix plus the seeds actually present."""
    series = [(seed, load(roots, arm, method, subset, seed)) for seed in SEEDS]
    usable = [(seed, value) for seed, value in series if value is not None]
    if not usable:
        return None, []
    index = usable[0][1].index
    for seed, value in usable[1:]:
        if not value.index.equals(index):
            raise SystemExit(f"engine sets differ: {arm}/{method}/{subset}/s{seed}")
    return np.vstack([value.to_numpy() for _, value in usable]), [s for s, _ in usable]


def paired(left_roots, left_arm, right_roots, right_arm, method, subset):
    left, left_seeds = stack(left_roots, left_arm, method, subset)
    right, right_seeds = stack(right_roots, right_arm, method, subset)
    if left is None or right is None:
        return None, []
    shared = [s for s in left_seeds if s in right_seeds]
    if not shared:
        return None, []
    left, _ = stack(left_roots, left_arm, method, subset)
    right, _ = stack(right_roots, right_arm, method, subset)
    left = left[[left_seeds.index(s) for s in shared]]
    right = right[[right_seeds.index(s) for s in shared]]
    return left - right, shared


def bootstrap(matrix, resamples, rng):
    n = matrix.shape[1]
    picks = rng.integers(0, n, size=(resamples, n))
    return matrix[:, picks].mean(axis=0).mean(axis=1)


def interval(samples, level):
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def analyse(v0_roots, new_roots, method, comparison, resamples, rng):
    """comparison: 'v1_minus_v0' or 'v1_minus_sham'."""
    if comparison == "v1_minus_v0":
        left_roots, left_arm, right_roots, right_arm = new_roots, "v1", v0_roots, ""
    else:
        left_roots, left_arm, right_roots, right_arm = new_roots, "v1", new_roots, "sham"

    per_subset, samples = {}, []
    for subset in SUBSETS:
        matrix, seeds = paired(left_roots, left_arm, right_roots, right_arm, method, subset)
        if matrix is None:
            continue
        per_subset[subset] = {"delta": float(matrix.mean()),
                              "n_engines": int(matrix.shape[1]), "seeds": seeds}
        samples.append(bootstrap(matrix, resamples, rng))
    if not samples:
        return None
    combined = np.mean(np.vstack(samples), axis=0)
    delta = float(np.mean([v["delta"] for v in per_subset.values()]))
    low, high = interval(combined, 95.0)
    low_b, high_b = interval(combined, BONFERRONI)
    weights = np.array([v["n_engines"] for v in per_subset.values()], float)
    values = np.array([v["delta"] for v in per_subset.values()], float)
    weighted = float(np.sum(weights * values) / np.sum(weights))
    return {"method": method, "comparison": comparison, "delta": delta,
            "ci95": [low, high], "ci9833": [low_b, high_b],
            "engine_weighted": weighted,
            "direction_negative": int(sum(1 for v in per_subset.values() if v["delta"] < 0)),
            "n_subsets": len(per_subset), "complete": len(per_subset) == len(SUBSETS),
            "per_subset": per_subset}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", required=True)
    parser.add_argument("--new-roots", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20261114)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    v0_roots = [Path(p) for p in args.v0_root.split(",") if p]
    new_roots = [Path(p) for p in args.new_roots.split(",") if p]
    rng = np.random.default_rng(args.boot_seed)

    report = {"v0_roots": [str(p) for p in v0_roots],
              "new_roots": [str(p) for p in new_roots],
              "resamples": args.resamples, "results": {}, "claims": {}}

    print("## Primary comparisons (engine-level, four subsets equally weighted)\n")
    print("| method | comparison | subsets | delta | 95% CI | 98.33% CI | weighted | direction |")
    print("|---|---|---|---|---|---|---|---|")
    for method in METHODS:
        primary = "v1_minus_sham" if method in SHAM else "v1_minus_v0"
        entry = analyse(v0_roots, new_roots, method, primary, args.resamples, rng)
        secondary = analyse(v0_roots, new_roots, method, "v1_minus_v0",
                            args.resamples, rng)
        report["results"][method] = {"primary": entry, "secondary": secondary}
        if entry is None:
            print(f"| {method} | {primary} | 0/4 | pending | | | | |")
            continue
        report["results"][method]["secondary"] = secondary
        print(f"| {method} | {primary} | {entry['n_subsets']}/4 | {entry['delta']:+.4f} | "
              f"[{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}] | "
              f"[{entry['ci9833'][0]:+.4f}, {entry['ci9833'][1]:+.4f}] | "
              f"{entry['engine_weighted']:+.4f} | {entry['direction_negative']}/"
              f"{entry['n_subsets']} |")

    print("\n## V1 - V0 (for the three sham methods this is capacity + information, "
          "not an information effect)\n")
    print("| method | subsets | delta | 95% CI | direction |")
    print("|---|---|---|---|---|")
    for method in METHODS:
        entry = report["results"][method].get("secondary")
        if entry is None:
            print(f"| {method} | 0/4 | pending | | |")
            continue
        print(f"| {method} | {entry['n_subsets']}/4 | {entry['delta']:+.4f} | "
              f"[{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}] | "
              f"{entry['direction_negative']}/{entry['n_subsets']} |")

    # Claim R: aidgn / two_phase / ffve / fcstgnn, each with its own primary.
    claim_r = [m for m in ("aidgn", "two_phase", "ffve", "fcstgnn")]
    passed, aligned = 0, 0
    for method in claim_r:
        entry = report["results"][method]["primary"]
        if not entry or not entry["complete"]:
            continue
        if entry["direction_negative"] >= 3:
            aligned += 1
        if (entry["delta"] < 0 and entry["direction_negative"] >= 3
                and entry["ci95"][1] < 0):
            passed += 1
    if aligned >= 3 and passed >= 3:
        tier = "strong"
    elif aligned >= 3:
        tier = "weak"
    else:
        tier = "not_supported"
    report["claims"]["R"] = {"passed": passed, "aligned": aligned, "tier": tier,
                             "methods": claim_r}
    print(f"\n## Claim R\n\npassed={passed}  aligned={aligned}  tier=**{tier}**")
    print("(p >= 3 -> strong; p < 3 and m >= 3 -> weak; m < 3 -> not supported; "
          "not-supported switches the main line to protocol audit, strong baselines and "
          "mechanism attribution)")

    # Claim I: dual_mixer only, single method, no cross-method pooling.
    dual = report["results"]["dual_mixer"]["primary"]
    print("\n## Claim I (dual_mixer only, n=1)\n")
    if dual and dual["complete"]:
        print(f"V1 - sham = {dual['delta']:+.4f} "
              f"95% CI [{dual['ci95'][0]:+.4f}, {dual['ci95'][1]:+.4f}] "
              f"direction {dual['direction_negative']}/4")
    else:
        print("pending")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
