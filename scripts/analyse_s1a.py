#!/usr/bin/env python3
"""S1-A analysis: engine-level RMSE and the frozen position-selection rule.

The rule is ``docs/paper_b_next/S1A_POSITION_PREREG.md`` (v1, frozen before the
80 cells landed). This script implements that document and adds nothing to it.

Inputs are each cell's ``val_pred.csv`` only. ``metrics.json`` is read for
``params`` and for the pooled-vs-engine comparison, never for the score: its
``val_engine_rmse`` field is a pooled alias (S1-C audit).

The 80 cells sit on two hosts, in two different copies of the repository:

    /home/user/s1a/paper_b/results/raw     (3090, shard 1)
    /home/yndx/s1a/paper_b/results/raw     (a40,  shard 0)

Pass both to ``--root`` as a comma-separated list to read them where they are,
or run ``scripts/s1a_collect_and_analyse.sh`` to pull them into this checkout
first and analyse one directory.

Usage
-----
    python scripts/analyse_s1a.py --json s1a_report.json
    python scripts/analyse_s1a.py --root /home/user/s1a/paper_b/results/raw,/home/yndx/s1a/paper_b/results/raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

POSITIONS = ("none", "pre_norm", "post_norm", "head_concat")
CYCLE_POSITIONS = POSITIONS[1:]
TAGS = {
    "none": "S1A_none",
    "pre_norm": "S1A_pre",
    "post_norm": "S1A_post",
    "head_concat": "S1A_head",
}
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SEEDS = (101, 202, 303, 42, 43)
N_CUTS = 5
RULE_VERSION = "v1"
PREREG_DOC = "docs/paper_b_next/S1A_POSITION_PREREG.md"
REPO = Path(__file__).resolve().parents[1]
BONFERRONI_LEVEL = 100.0 * (1.0 - 0.05 / 3.0)
# Default search order, so the acceptance command
#   python scripts/analyse_s1a.py --json s1a_report.json
# works verbatim on any host. The collector writes the first entry; the other
# two are the training hosts' own copies, which exist only on that host.
DEFAULT_ROOTS = ",".join([
    "paper_b/results/s1a_collected",
    "paper_b/results/raw",
    "/home/user/s1a/paper_b/results/raw",
    "/home/yndx/s1a/paper_b/results/raw",
])


# --------------------------------------------------------------------- inputs
def engine_rmse(frame):
    """Per-engine RMSE over that engine's truncation cuts, indexed by unit."""
    squared = (frame["y_pred"].to_numpy(float) - frame["y_true"].to_numpy(float)) ** 2
    work = pd.DataFrame({"unit": frame["unit"].to_numpy(), "se": squared})
    return work.groupby("unit")["se"].mean().pow(0.5).sort_index()


def check_config(directory, position, subset, seed):
    """A cell left behind by another configuration must not be reused silently.

    run_grid skips any cell whose metrics.json exists, so a stale cell would
    never show up as missing. Compare the recorded config with its own path.
    """
    path = Path(directory) / "config.json"
    if not path.exists():
        return None
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable config.json"
    problems = []
    if recorded.get("cycle_pos") != position:
        problems.append(f"cycle_pos={recorded.get('cycle_pos')!r} != {position!r}")
    if recorded.get("fp32") is not True:
        problems.append(f"fp32={recorded.get('fp32')!r}")
    if recorded.get("subset") != subset:
        problems.append(f"subset={recorded.get('subset')!r} != {subset!r}")
    if recorded.get("seed") != seed:
        problems.append(f"seed={recorded.get('seed')!r} != {seed}")
    return "; ".join(problems) or None


def find_cell(roots, position, subset, seed):
    """Every copy of one cell across the given roots."""
    return [Path(root) / TAGS[position] / subset / f"s{seed}" / "val_pred.csv" for root in roots
            if (Path(root) / TAGS[position] / subset / f"s{seed}" / "val_pred.csv").exists()]


def read_metrics(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load(roots, positions, subsets, seeds):
    cells, missing, bad_cuts, bad_config, duplicate, gaps = {}, [], [], [], [], []
    for subset in subsets:
        for position in positions:
            for seed in seeds:
                found = find_cell(roots, position, subset, seed)
                if not found:
                    missing.append(f"{TAGS[position]}/{subset}/s{seed}")
                    continue
                if len(found) > 1:
                    digests = [hashlib.sha256(item.read_bytes()).hexdigest() for item in found]
                    duplicate.append({"cell": f"{position}/{subset}/s{seed}",
                                      "paths": [str(item) for item in found],
                                      "identical": len(set(digests)) == 1})
                path = found[0]
                problem = check_config(path.parent, position, subset, seed)
                if problem:
                    bad_config.append({"path": str(path), "problem": problem})
                frame = pd.read_csv(path)
                sizes = frame.groupby("unit").size()
                if not (sizes == N_CUTS).all():
                    bad_cuts.append({"path": str(path),
                                     "units_off": int((sizes != N_CUTS).sum()),
                                     "min": int(sizes.min()), "max": int(sizes.max())})
                series = engine_rmse(frame)
                cells[(subset, position, seed)] = series
                pooled = read_metrics(path.parent / "metrics.json").get("val_engine_rmse")
                if isinstance(pooled, (int, float)):
                    gaps.append({"path": str(path), "pooled": float(pooled),
                                 "engine": float(series.mean())})
    return cells, missing, bad_cuts, bad_config, duplicate, gaps


def load_params(roots, positions, subsets, seeds):
    """Total and trainable parameter counts from each cell's metrics.json."""
    counts = {}
    for subset in subsets:
        for position in positions:
            totals, trainables = [], []
            for seed in seeds:
                for path in find_cell(roots, position, subset, seed):
                    recorded = read_metrics(path.parent / "metrics.json")
                    if "params" in recorded:
                        totals.append(int(recorded["params"]))
                    if "trainable_params" in recorded:
                        trainables.append(int(recorded["trainable_params"]))
                    break
            if totals:
                counts[(subset, position)] = {"params": totals,
                                              "trainable": trainables or None}
    return counts


# ---------------------------------------------------------------- statistics
def stack(cells, subset, position, seeds):
    """(n_seeds, n_engines) matrix; engine sets must agree across seeds."""
    usable = [(seed, cells[(subset, position, seed)]) for seed in seeds
              if (subset, position, seed) in cells]
    if not usable:
        return None, []
    index = usable[0][1].index
    for seed, series in usable[1:]:
        if not series.index.equals(index):
            raise SystemExit(f"engine sets differ across seeds for {subset}/{position}: s{seed}")
    matrix = np.vstack([series.to_numpy() for _, series in usable])
    return matrix, [seed for seed, _ in usable]


def delta_matrix(cells, subset, position, baseline, seeds):
    """Per-engine RMSE difference position - baseline on the seeds both have."""
    left, seeds_left = stack(cells, subset, position, seeds)
    right, seeds_right = stack(cells, subset, baseline, seeds)
    if left is None or right is None:
        return None, []
    shared = [seed for seed in seeds_left if seed in seeds_right]
    if not shared:
        return None, []
    left, _ = stack(cells, subset, position, shared)
    right, _ = stack(cells, subset, baseline, shared)
    return left - right, shared


def bootstrap(matrix, resamples, rng):
    """Engine-level bootstrap: seeds first, then resampled engines."""
    n_engines = matrix.shape[1]
    picks = rng.integers(0, n_engines, size=(resamples, n_engines))
    return matrix[:, picks].mean(axis=0).mean(axis=1)


def describe(samples, level=95.0):
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(np.mean(samples)), float(low), float(high)


def normalise_pairs(pairs):
    """Lookup for any ordered pair, mirroring the sign when needed."""
    table = {}
    for entry in pairs:
        table[(entry["left"], entry["right"])] = entry
        table[(entry["right"], entry["left"])] = {
            **entry,
            "left": entry["right"], "right": entry["left"],
            "delta": -entry["delta"],
            "ci95": [-entry["ci95"][1], -entry["ci95"][0]],
            "ci9833": [-entry["ci9833"][1], -entry["ci9833"][0]],
            "engine_weighted_delta": -entry["engine_weighted_delta"],
        }
    return table


# ------------------------------------------------------------------- decision
def decide(vs_none, pairwise, capacity_flags):
    """Apply S1A_POSITION_PREREG.md v1 section 3, in order."""
    blocked = {}
    for position in CYCLE_POSITIONS:
        entry = vs_none.get(position)
        if entry is None:
            blocked[position] = "no comparable cells"
        elif not entry["complete"]:
            blocked[position] = "missing cells or seeds"
        elif capacity_flags.get(position):
            blocked[position] = "capacity differs by more than 1% from `none`"

    eligible = [p for p in CYCLE_POSITIONS if p not in blocked]
    qualified = [p for p in eligible
                 if vs_none[p]["ci95"][1] < 0 and vs_none[p]["direction_negative"] >= 3]

    notes, winner = [], None
    if not qualified:
        notes.append("no cycle position has its combined 95% CI upper bound below 0 "
                     "with at least 3 of 4 subsets pointing the same way")
    elif len(qualified) == 1:
        winner = qualified[0]
        notes.append(f"a single position qualifies: {winner}")
    else:
        ordered = sorted(qualified, key=lambda p: (vs_none[p]["delta"], p != "pre_norm"))
        best = ordered[0]
        dominant = all(pairwise[(best, other)]["ci95"][1] < 0
                       for other in qualified if other != best)
        if dominant:
            winner = best
            notes.append(f"{len(qualified)} positions qualify; {best} dominates each of "
                         f"them pairwise")
        else:
            notes.append(f"{len(qualified)} positions qualify ({', '.join(qualified)}) but "
                         f"{best} does not dominate all of them pairwise")

    if winner and vs_none[winner]["ci9833"][1] >= 0:
        notes.append(f"{winner} passes the 95% gate but its 98.33% CI crosses 0: "
                     "direction consistent, below the multiplicity-corrected threshold")

    cycle_negative = bool(eligible) and all(vs_none[p]["ci95"][0] > 0 for p in eligible)
    if cycle_negative:
        notes.append("`none` beats every eligible cycle position with a combined CI "
                     "lower bound above 0: on DSPT the cycle encoding is a negative effect")

    selected = winner or "pre_norm"
    if winner is None:
        notes.append("no unique winner; stage two continues with `pre_norm` as the "
                     "implementation default, which is a default and not evidence")

    return {
        "rule_version": RULE_VERSION,
        "eligible": eligible,
        "blocked": blocked,
        "qualified": qualified,
        "winner": winner,
        "selected_position": selected,
        "fallback_used": winner is None,
        "cycle_is_negative_effect": cycle_negative,
        "notes": notes,
    }


# ----------------------------------------------------------------------- main
def git_field(relative, fmt):
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", f"--format={fmt}", "--", relative],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOTS,
                        help="one out-root or several separated by commas; the first copy "
                             "found wins and duplicates are reported with whether the bytes agree")
    parser.add_argument("--subsets", default=",".join(SUBSETS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20261114)
    parser.add_argument("--prereg-commit", default=None)
    parser.add_argument("--frozen-at", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    roots = [Path(item) for item in args.root.split(",") if item]
    subsets = tuple(s for s in args.subsets.split(",") if s)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    rng = np.random.default_rng(args.boot_seed)

    prereg_commit = args.prereg_commit or git_field(PREREG_DOC, "%H") or "unknown"
    frozen_at = args.frozen_at or git_field(PREREG_DOC, "%cI") or "unknown"

    print("## Roots (absolute)\n")
    for root in roots:
        count = sum(len(list(root.glob(f"{tag}/*/s*/val_pred.csv"))) for tag in TAGS.values()) \
            if root.is_dir() else 0
        print(f"- `{root.resolve()}` — {count} cells")
    print(f"\nRule `{RULE_VERSION}` frozen at `{frozen_at}` (`{prereg_commit}`)\n")

    cells, missing, bad_cuts, bad_config, duplicate, gaps = load(roots, POSITIONS, subsets, seeds)
    counts = load_params(roots, POSITIONS, subsets, seeds)

    report = {"roots": [str(root.resolve()) for root in roots],
              "subsets": list(subsets), "seeds": list(seeds),
              "resamples": args.resamples, "boot_seed": args.boot_seed,
              "rule_version": RULE_VERSION, "prereg_commit": prereg_commit,
              "frozen_at": frozen_at,
              "missing": missing, "bad_cuts": bad_cuts, "bad_config": bad_config,
              "duplicate": duplicate, "absolute": {}, "pairs": []}

    # ---- absolute engine-level RMSE per (subset, position) -----------------
    print("## Absolute engine-level RMSE (mean over seeds, engine as unit)\n")
    print("| subset | " + " | ".join(POSITIONS) + " | n_eng | seeds_max |")
    print("|" + "---|" * (len(POSITIONS) + 3))
    for subset in subsets:
        row, n_eng, n_seeds = [subset], None, 0
        for position in POSITIONS:
            matrix, used = stack(cells, subset, position, seeds)
            if matrix is None:
                row.append("-")
                continue
            n_eng = matrix.shape[1]
            n_seeds = max(n_seeds, len(used))
            spread = matrix.mean(axis=1).std(ddof=1) if len(used) > 1 else 0.0
            row.append(f"{matrix.mean():.4f} +/- {spread:.4f}" if len(used) > 1
                       else f"{matrix.mean():.4f}")
            report["absolute"].setdefault(subset, {})[position] = {
                "engine_mean_rmse": float(matrix.mean()),
                "seed_means": [float(value) for value in matrix.mean(axis=1)],
                "seed_sd": float(spread) if len(used) > 1 else None,
                "n_engines": int(n_eng), "seeds": used,
            }
        row.extend([str(n_eng), str(n_seeds)])
        print("| " + " | ".join(row) + " |")

    # ---- parameter counts --------------------------------------------------
    capacity_flags = {}
    if counts:
        print("\n## Parameter counts (`params` from metrics.json)\n")
        print("| subset | " + " | ".join(POSITIONS) + " | spread |")
        print("|" + "---|" * (len(POSITIONS) + 2))
        per_position = {}
        for subset in subsets:
            row, known = [subset], []
            for position in POSITIONS:
                entry = counts.get((subset, position))
                if not entry:
                    row.append("-")
                    continue
                unique = sorted(set(entry["params"]))
                per_position.setdefault(position, set()).update(unique)
                known.extend(unique)
                row.append(str(unique[0]) if len(unique) == 1 else "/".join(map(str, unique)))
            row.append(f"{(max(known) - min(known)) / min(known) * 100:.2f}%" if known else "-")
            print("| " + " | ".join(row) + " |")

        base = per_position.get("none")
        summary = {"per_position": {p: sorted(v) for p, v in per_position.items()}}
        if base and len(base) == 1:
            base_value = next(iter(base))
            parts, deltas = [], {}
            for position in CYCLE_POSITIONS:
                values = per_position.get(position) or set()
                if len(values) != 1:
                    continue
                difference = next(iter(values)) - base_value
                deltas[position] = difference
                parts.append(f"{position} {difference:+d}")
                if abs(difference) / base_value > 0.01:
                    capacity_flags[position] = True
            print(f"\nAgainst `none` = {base_value}: " + ", ".join(parts))
            summary["base"] = base_value
            summary["delta"] = deltas
            if capacity_flags:
                print(f"Confound flag: {', '.join(sorted(capacity_flags))} differ by more "
                      "than 1% in parameter count; those positions cannot win on RMSE alone "
                      "and need a capacity-matched control.")
            else:
                print("No position differs from `none` by more than 1% in parameter count.")
            if not any(entry["trainable"] for entry in counts.values()):
                print("`trainable_params` is not recorded for this grid. The counts above are "
                      "optimizer-visible totals (`metrics.json:params`; AdamW receives "
                      "`net.parameters()`), which is *not* the capacity gate. `head_concat` "
                      "creates 576 `pe_proj` parameters that never enter the forward pass, so "
                      "its gradient-path count is 115 587, not 116 163. The gate uses the "
                      "gradient-path count; see S1A_CYCLE_MATCHED.md §4.1.")
        report["params"] = summary

    # ---- pairwise comparisons ---------------------------------------------
    print("\n## Paired engine-level differences (negative = left arm better)\n")
    print("| pair | subsets | d_mean | 95% CI | 98.33% CI | weighted | direction | seed_d |")
    print("|---|---|---|---|---|---|---|---|")
    pairs = [(p, "none") for p in CYCLE_POSITIONS]
    pairs += [(POSITIONS[j], POSITIONS[i])
              for i in range(1, len(POSITIONS)) for j in range(i + 1, len(POSITIONS))]
    for left, right in pairs:
        per_subset, samples = {}, []
        for subset in subsets:
            matrix, used = delta_matrix(cells, subset, left, right, seeds)
            if matrix is None:
                continue
            per_subset[subset] = {
                "delta": float(matrix.mean()),
                "seed_deltas": {int(seed): float(value)
                                for seed, value in zip(used, matrix.mean(axis=1))},
                "n_engines": int(matrix.shape[1]), "seeds": used,
            }
            samples.append(bootstrap(matrix, args.resamples, rng))
        if not samples:
            continue
        combined = np.mean(np.vstack(samples), axis=0)
        _, low, high = describe(combined, 95.0)
        _, low_b, high_b = describe(combined, BONFERRONI_LEVEL)
        delta = float(np.mean([entry["delta"] for entry in per_subset.values()]))

        weights = np.array([entry["n_engines"] for entry in per_subset.values()], float)
        values = np.array([entry["delta"] for entry in per_subset.values()], float)
        weighted = float(np.sum(weights * values) / np.sum(weights)) if weights.sum() else float("nan")

        seed_values = {}
        for entry in per_subset.values():
            for seed, value in entry["seed_deltas"].items():
                seed_values.setdefault(seed, []).append(value)
        seed_deltas = [float(np.mean(v)) for _, v in sorted(seed_values.items())]

        complete = (len(per_subset) == len(subsets)
                    and all(len(entry["seeds"]) == len(seeds) for entry in per_subset.values()))
        direction = sum(1 for entry in per_subset.values() if entry["delta"] < 0)

        report["pairs"].append({
            "left": left, "right": right, "delta": delta,
            "ci95": [low, high], "ci9833": [low_b, high_b],
            "engine_weighted_delta": weighted,
            "direction_negative": int(direction), "n_subsets": len(per_subset),
            "complete": bool(complete), "seed_deltas": seed_deltas,
            "per_subset": per_subset,
        })
        mark = "**" if right == "none" else ""
        print(f"| {mark}{left} - {right}{mark} | {len(per_subset)}/{len(subsets)} | {delta:+.4f} | "
              f"[{low:+.4f}, {high:+.4f}] | [{low_b:+.4f}, {high_b:+.4f}] | {weighted:+.4f} | "
              f"{direction}/{len(per_subset)} | "
              + " / ".join(f"{value:+.3f}" for value in seed_deltas) + " |")

    # ---- frozen decision ---------------------------------------------------
    vs_none = {entry["left"]: entry for entry in report["pairs"] if entry["right"] == "none"}
    pairwise = normalise_pairs(report["pairs"])
    decision = decide(vs_none, pairwise, capacity_flags)

    # Aggregation-weight sensitivity: equal weight is the rule, engine weight
    # is only a check.
    for position, entry in vs_none.items():
        if np.sign(entry["engine_weighted_delta"]) != np.sign(entry["delta"]):
            decision["notes"].append(
                f"{position}: equal-weight and engine-weighted aggregation have opposite "
                "signs; the conclusion is not robust to the aggregation weights")
    report["decision"] = decision

    print("\n## Frozen decision (S1A_POSITION_PREREG.md v1)\n")
    print(f"- eligible: {decision['eligible'] or 'none'}")
    if decision["blocked"]:
        for position, reason in decision["blocked"].items():
            print(f"- blocked: {position} -- {reason}")
    print(f"- qualified: {decision['qualified'] or 'none'}")
    print(f"- winner: {decision['winner'] or 'none'}")
    print(f"- selected for stage two: **{decision['selected_position']}**"
          + ("  (default, not evidence)" if decision["fallback_used"] else ""))
    if decision["cycle_is_negative_effect"]:
        print("- cycle is a negative effect on DSPT (see notes)")
    for note in decision["notes"]:
        print(f"  - {note}")

    # ---- diagnostics -------------------------------------------------------
    if gaps:
        worst = max(gaps, key=lambda item: abs(item["pooled"] - item["engine"]))
        difference = abs(worst["pooled"] - worst["engine"])
        print("\n## Pooled field vs recomputed engine-level\n")
        print(f"max |val_engine_rmse - engine_mean| = {difference:.4f} at `{worst['path']}`")
        print("`val_engine_rmse` is a pooled alias; every number above is recomputed "
              "from val_pred.csv and never read from that field.")
        report["pooled_gap"] = {"max": difference, "worst": worst}

    if duplicate:
        print(f"\nCells found under more than one root: {len(duplicate)}")
        for item in duplicate:
            print("  ", item["cell"], "identical" if item["identical"] else "**BYTES DIFFER**")

    print(f"\nMissing cells: {len(missing)}")
    if bad_config:
        print(f"Cells whose config.json contradicts their path: {len(bad_config)}")
        for item in bad_config[:10]:
            print("  ", item["path"], "->", item["problem"])
    if bad_cuts:
        print(f"Cells whose engines do not all have {N_CUTS} cuts: {len(bad_cuts)}")
        for item in bad_cuts[:10]:
            print("  ", item)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
