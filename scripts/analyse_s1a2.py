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
# Claim R is these four only. dual_mixer is deliberately excluded: it carries
# claim I on its own (n = 1) and must never enter a cross-method average. This
# is an explicit whitelist, not an accident of loop order.
CLAIM_R_METHODS = ("aidgn", "two_phase", "ffve", "fcstgnn")
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEEDS = (101, 202, 303)
ENGINES = {"FD001": 20, "FD002": 52, "FD003": 20, "FD004": 50}
CUTS = 5
EXPECTED_CELLS = (len(METHODS) + len(SHAM)) * len(SUBSETS) * len(SEEDS)
BONFERRONI = 100.0 * (1.0 - 0.05 / 3.0)


def engine_rmse(path):
    frame = pd.read_csv(path)
    squared = (frame["y_pred"].to_numpy(float) - frame["y_true"].to_numpy(float)) ** 2
    work = pd.DataFrame({"unit": frame["unit"].to_numpy(), "se": squared})
    return work.groupby("unit")["se"].mean().pow(0.5).sort_index()


def locate(roots, arm, method, subset, seed):
    """Return (predictions path, problem). ``problem`` is None when the cell is valid.

    A cell counts as complete only when all three hold: metrics.json exists, the
    recorded config agrees with the path, and the prediction file has exactly
    ``engines x 5 cuts`` rows. The row check catches both a run that exited
    halfway and a wrong number of truncation cuts.
    """
    for root in roots:
        base = Path(root) / arm if arm else Path(root)
        folder = base / method / subset / f"s{seed}"
        preds = folder / "validation_predictions.csv"
        if not folder.is_dir():
            continue
        if not (folder / "metrics.json").exists():
            return None, "no metrics.json"
        config = folder / "config.json"
        if config.exists():
            try:
                recorded = json.loads(config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, "unreadable config.json"
            if arm:
                recorded_arm = recorded.get("arm", recorded.get("cycle_arm"))
                if recorded_arm is None:
                    return None, "config.arm missing"
                value = str(recorded_arm).strip().lower()
                if value not in ("v1", "sham"):
                    return None, f"config.arm={recorded_arm!r} not in v1/sham"
                if value != arm:
                    return None, f"config.arm={recorded_arm!r}!={arm!r}"
            name = recorded.get("method", recorded.get("model"))
            if name != method:
                return None, f"config.method={name!r}!={method!r}"
            if recorded.get("subset") != subset:
                return None, f"config.subset={recorded.get('subset')!r}!={subset!r}"
            if recorded.get("seed") != seed:
                return None, f"config.seed={recorded.get('seed')!r}!={seed!r}"
        if not preds.exists():
            return None, "no validation_predictions.csv"
        with preds.open() as stream:
            rows = sum(1 for _ in stream) - 1
        want = ENGINES[subset] * CUTS
        if rows != want:
            return None, f"prediction rows {rows}!={want}"
        return preds, None
    return None, "missing"


def load(roots, arm, method, subset, seed):
    path, problem = locate(roots, arm, method, subset, seed)
    return (None, problem) if path is None else (engine_rmse(path), None)


def stack(roots, arm, method, subset):
    """(n_seeds, n_engines) matrix, the seeds present, and why any are missing."""
    usable, problems = [], {}
    for seed in SEEDS:
        value, problem = load(roots, arm, method, subset, seed)
        if value is None:
            problems[seed] = problem
        else:
            usable.append((seed, value))
    if not usable:
        return None, [], problems
    index = usable[0][1].index
    for seed, value in usable[1:]:
        if not value.index.equals(index):
            problems[seed] = "engine set differs"
            usable = [(s, v) for s, v in usable if s != seed]
    if not usable:
        return None, [], problems
    return np.vstack([value.to_numpy() for _, value in usable]), [s for s, _ in usable], problems


def bootstrap(matrix, resamples, rng):
    n = matrix.shape[1]
    picks = rng.integers(0, n, size=(resamples, n))
    return matrix[:, picks].mean(axis=0).mean(axis=1)


def interval(samples, level):
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def paired(left_roots, left_arm, right_roots, right_arm, method, subset):
    left, left_seeds, left_problems = stack(left_roots, left_arm, method, subset)
    right, right_seeds, right_problems = stack(right_roots, right_arm, method, subset)
    problems = {f"{left_arm or 'v0'}:s{s}": v for s, v in left_problems.items()}
    problems.update({f"{right_arm or 'v0'}:s{s}": v for s, v in right_problems.items()})
    if left is None or right is None:
        return None, [], problems
    shared = [s for s in left_seeds if s in right_seeds]
    if not shared:
        return None, [], problems
    return left[[left_seeds.index(s) for s in shared]] - \
        right[[right_seeds.index(s) for s in shared]], shared, problems


def analyse(v0_roots, new_roots, method, comparison, resamples, rng):
    """comparison: 'v1_minus_v0', 'sham_minus_v0' or 'v1_minus_sham'.

    ``sham_minus_v0`` exists only to decompose the frozen primary:
    V1 - sham = (V1 - V0) - (sham - V0). If the sham arm damages training, the
    apparent primary effect comes from that damage, not from V1 improving.
    """
    if comparison == "v1_minus_v0":
        left_roots, left_arm, right_roots, right_arm = new_roots, "v1", v0_roots, ""
    elif comparison == "sham_minus_v0":
        left_roots, left_arm, right_roots, right_arm = new_roots, "sham", v0_roots, ""
    else:
        left_roots, left_arm, right_roots, right_arm = new_roots, "v1", new_roots, "sham"

    per_subset, samples = {}, []
    for subset in SUBSETS:
        matrix, seeds, problems = paired(left_roots, left_arm, right_roots, right_arm,
                                         method, subset)
        if matrix is None:
            per_subset[subset] = {"delta": None, "n_engines": 0, "seeds": [],
                                  "problems": problems}
            continue
        per_subset[subset] = {"delta": float(matrix.mean()),
                              "n_engines": int(matrix.shape[1]), "seeds": seeds,
                              "matrix": matrix.tolist(), "problems": problems}
        samples.append(bootstrap(matrix, resamples, rng))

    usable = {s: v for s, v in per_subset.items() if v["delta"] is not None}
    # "complete" means every subset carries all three seeds, not merely that all
    # four subsets have something. With one seed per subset the estimate is a
    # single run, and emitting a tier from it would report an intermediate state
    # as a result.
    full_seeds = all(len(v["seeds"]) == len(SEEDS) for v in usable.values())
    if not samples:
        return {"method": method, "comparison": comparison, "delta": None,
                "ci95": [None, None], "ci9833": [None, None], "engine_weighted": None,
                "direction_negative": 0, "n_subsets": 0, "complete": False,
                "per_subset": per_subset}

    combined = np.mean(np.vstack(samples), axis=0)
    delta = float(np.mean([v["delta"] for v in usable.values()]))
    low, high = interval(combined, 95.0)
    low_b, high_b = interval(combined, BONFERRONI)
    weights = np.array([v["n_engines"] for v in usable.values()], float)
    values = np.array([v["delta"] for v in usable.values()], float)
    return {"method": method, "comparison": comparison, "delta": delta,
            "ci95": [low, high], "ci9833": [low_b, high_b],
            "engine_weighted": float(np.sum(weights * values) / np.sum(weights)),
            "direction_negative": int(sum(1 for v in usable.values() if v["delta"] < 0)),
            "n_subsets": len(usable), "complete": len(usable) == len(SUBSETS) and full_seeds,
            "per_subset": per_subset}


def count_valid(new_roots):
    """How many of the 96 cells are terminal and valid right now.

    A cell that is still training has its directory but no metrics.json, so it
    fails the validity check. That is correct for the statistics, but such a
    cell is *running*, not *failed*, and must not enter a failure list.
    """
    valid = 0
    for arm, methods in (("v1", METHODS), ("sham", SHAM)):
        for method in methods:
            for subset in SUBSETS:
                for seed in SEEDS:
                    path, _ = locate(new_roots, arm, method, subset, seed)
                    valid += path is not None
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", required=True)
    parser.add_argument("--new-roots", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20261114)
    parser.add_argument("--drop-cell", default=None,
                        help="influence check only: method/subset/seed, e.g. aidgn/FD003/101")
    parser.add_argument("--partial", action="store_true",
                        help="allow a partial report; without it the script refuses to "
                             "emit a failure list while any of the 96 cells is missing")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    v0_roots = [Path(p) for p in args.v0_root.split(",") if p]
    new_roots = [Path(p) for p in args.new_roots.split(",") if p]
    rng = np.random.default_rng(args.boot_seed)

    valid = count_valid(new_roots)
    print(f"## Cell inventory\n\n{valid}/{EXPECTED_CELLS} cells terminal and valid")
    if valid < EXPECTED_CELLS:
        print(f"**PARTIAL REPORT** — {EXPECTED_CELLS - valid} cells are not terminal yet. "
              f"Missing cells are either still running or genuinely absent; they are NOT "
              f"failures until the grid is idle. Confirm no live runner "
              f"(`pgrep -f run_grid_cycle` = 0) before treating any of them as failed.")
        if not args.partial:
            print("Refusing to emit a failure list from a partial grid. Re-run with "
                  "--partial for a clearly-labelled intermediate report.")

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
        if entry["n_subsets"] == 0:
            print(f"| {method} | {primary} | 0/4 | pending | | | | |")
            continue
        report["results"][method]["secondary"] = secondary
        print(f"| {method} | {primary} | {entry['n_subsets']}/4 | {entry['delta']:+.4f} | "
              f"[{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}] | "
              f"[{entry['ci9833'][0]:+.4f}, {entry['ci9833'][1]:+.4f}] | "
              f"{entry['engine_weighted']:+.4f} | {entry['direction_negative']}/"
              f"{entry['n_subsets']} |")

    print("\n## Per-subset delta and influence (descriptive; the decision still uses "
          "every cell)\n")
    print("| method | FD001 | FD002 | FD003 | FD004 | drop FD003 | drop FD004 |")
    print("|---|---|---|---|---|---|---|")
    for method in METHODS:
        primary = report["results"][method]["primary"]
        entry = primary if primary["n_subsets"] else report["results"][method]["secondary"]
        if entry["n_subsets"] == 0:
            print(f"| {method} | pending | | | | | |")
            continue
        cells = [f"{entry['per_subset'][s]['delta']:+.3f}"
                 if entry["per_subset"].get(s, {}).get("delta") is not None else "-"
                 for s in SUBSETS]
        drops = []
        for drop in ("FD003", "FD004"):
            rest = [v["delta"] for s, v in entry["per_subset"].items()
                    if s != drop and v["delta"] is not None]
            drops.append(f"{np.mean(rest):+.3f}" if rest else "-")
        print(f"| {method} | " + " | ".join(cells) + " | " + " | ".join(drops) + " |")

    print("\n## V1 - V0 (for the three sham methods this is capacity + information, "
          "not an information effect)\n")
    print("| method | subsets | delta | 95% CI | direction |")
    print("|---|---|---|---|---|")
    for method in METHODS:
        entry = report["results"][method]["secondary"]
        if entry["n_subsets"] == 0:
            print(f"| {method} | 0/4 | pending | | |")
            continue
        print(f"| {method} | {entry['n_subsets']}/4 | {entry['delta']:+.4f} | "
              f"[{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}] | "
              f"{entry['direction_negative']}/{entry['n_subsets']} |")

    print("\n## sham - V0 (decomposition: V1 - sham = (V1 - V0) - (sham - V0))\n")
    print("A positive sham - V0 means the sham arm is worse than the untouched method. "
          "That is damage, not capacity, and it inflates V1 - sham.\n")
    print("| method | subsets | sham-V0 | 95% CI | V1-V0 | V1-sham |")
    print("|---|---|---|---|---|---|")
    for method in SHAM:
        entry = analyse(v0_roots, new_roots, method, "sham_minus_v0",
                        args.resamples, rng)
        report["results"][method]["sham_minus_v0"] = entry
        if entry["n_subsets"] == 0:
            print(f"| {method} | 0/4 | pending | | | |")
            continue
        v1v0 = report["results"][method]["secondary"]
        primary = report["results"][method]["primary"]
        v1v0_text = "pending" if v1v0["delta"] is None else f"{v1v0['delta']:+.4f}"
        primary_text = "pending" if primary["delta"] is None else f"{primary['delta']:+.4f}"
        print(f"| {method} | {entry['n_subsets']}/4 | {entry['delta']:+.4f} | "
              f"[{entry['ci95'][0]:+.4f}, {entry['ci95'][1]:+.4f}] | "
              f"{v1v0_text} | {primary_text} |")

    # Claim R: aidgn / two_phase / ffve / fcstgnn, each with its own primary.
    claim_r = list(CLAIM_R_METHODS)
    passed, aligned = 0, 0
    for method in claim_r:
        entry = report["results"][method]["primary"]
        if not entry["complete"]:
            continue
        if entry["direction_negative"] >= 3:
            aligned += 1
        if (entry["delta"] < 0 and entry["direction_negative"] >= 3
                and entry["ci95"][1] < 0):
            passed += 1
    # No tier until every method's primary comparison is complete. With sham
    # missing, aidgn and two_phase would silently fall back to V1-V0, which is
    # capacity + information, not the pre-registered primary.
    missing = [m for m in claim_r if not report["results"][m]["primary"]["complete"]]
    if missing:
        tier = None
        print(f"\n## Claim R\n\ntier withheld: primary comparison incomplete for "
              f"{', '.join(missing)}. V1-V0 for a sham method is capacity + "
              f"information, not the pre-registered primary.\n")
        for method in missing:
            entry = report["results"][method]["primary"]
            print(f"- {method} ({entry['comparison']}): {entry['n_subsets']}/4 subsets usable")
            for subset, value in entry["per_subset"].items():
                if value["delta"] is not None:
                    continue
                detail = "; ".join(f"{k}:{v}" for k, v in value.get("problems", {}).items())
                print(f"    {subset}: {detail or 'no usable cell'}")
        report["claims"]["R"] = {"passed": passed, "aligned": aligned, "tier": None,
                                 "incomplete": missing, "methods": claim_r}
        if args.json:
            Path(args.json).write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"\nWrote {args.json}")
        return
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
    if dual["complete"]:
        print(f"V1 - sham = {dual['delta']:+.4f} "
              f"95% CI [{dual['ci95'][0]:+.4f}, {dual['ci95'][1]:+.4f}] "
              f"direction {dual['direction_negative']}/4")
    else:
        print("pending")

    if args.drop_cell:
        method, subset, seed = args.drop_cell.split("/")
        seed = int(seed)
        primary = report["results"][method]["primary"]
        entry = primary if primary["n_subsets"] else report["results"][method]["secondary"]
        print(f"\n## Influence check: {args.drop_cell} removed (descriptive only; "
              f"the decision above still uses every cell)\n")
        if entry["n_subsets"] == 0:
            print("pending")
        else:
            deltas, emptied = [], []
            for name, value in entry["per_subset"].items():
                matrix = np.array(value["matrix"])
                seeds = value["seeds"]
                keep = [i for i, s in enumerate(seeds) if not (name == subset and s == seed)]
                if not keep:
                    emptied.append(name)
                    continue
                deltas.append(float(matrix[keep].mean()))
            full = entry["delta"]
            dropped = float(np.mean(deltas)) if deltas else float("nan")
            print(f"full = {full:+.4f}   without the cell = {dropped:+.4f}   "
                  f"shift = {dropped - full:+.4f}")
            if emptied:
                print(f"note: dropping this cell empties {emptied}; that subset then "
                      "contributes nothing, which is an artefact of the partial data, "
                      "not a result")
            report["influence"] = {"drop_cell": args.drop_cell, "full": full,
                                   "without": dropped, "shift": dropped - full}

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
