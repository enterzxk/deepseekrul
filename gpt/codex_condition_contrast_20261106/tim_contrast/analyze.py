"""Read completed jobs, compare exact common queries, and emit a bounded next step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import engine_statistics, summarize, paired_comparison
from .run import write_json, code_fingerprint
from .data import sha256


def read_runs(root, prior="full"):
    frames, metadata = {}, {}
    for path in sorted(Path(root).glob(f"FD*/*_{prior}/done.json")):
        folder = path.parent
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["code_sha256"] != code_fingerprint():
            raise ValueError(f"Mixed code versions: {folder}")
        for name in ("predictions.csv", "metrics.json"):
            if sha256(folder/name) != receipt["artifacts"][name]:
                raise ValueError(f"Changed artifact: {folder/name}")
        meta = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(folder/"predictions.csv")
        key = (meta["subset"], meta["label"], meta["seed"], meta["fold"])
        if key in frames:
            raise ValueError(f"Duplicate run {key}")
        frames[key], metadata[key] = frame, meta
    return frames, metadata


def align_queries(a, b):
    keys = ["unit", "end_cycle"]
    a, b = [p.sort_values(keys).reset_index(drop=True) for p in (a, b)]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate prediction keys")
    if not a[keys].equals(b[keys]) or not np.array_equal(a.y_true.to_numpy(), b.y_true.to_numpy()):
        raise ValueError("Unmatched query keys/targets; intersections or dropped engines are forbidden")
    return a, b


def materialize(frames, subset, label, seed, folds):
    if label.startswith("scheduled"):
        suffix = "_risk" if label.endswith("_risk") else ""
        a = materialize(frames, subset, "pe50"+suffix, seed, folds)
        b = materialize(frames, subset, "w100_late"+suffix, seed, folds)
        a, b = align_queries(a, b)
        take = a.end_cycle >= 100
        a.loc[take, "y_pred"] = b.loc[take, "y_pred"]
        return a
    pieces = []
    for fold in folds:
        key = (subset, label, seed, fold)
        if key not in frames:
            raise KeyError(f"Missing completed run {key}")
        pieces.append(frames[key])
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["unit", "end_cycle"]).any():
        raise ValueError("An engine query appears in multiple outer folds")
    return result.sort_values(["unit", "end_cycle"]).reset_index(drop=True)


def aggregate(frames, subset, label, seeds, folds, anchor=None):
    engine_tables, seed_rows = [], []
    for seed in seeds:
        p = materialize(frames, subset, label, seed, folds)
        if anchor is not None:
            align_queries(p, materialize(frames, subset, anchor, seed, folds))
        e = engine_statistics(p)
        seed_rows.append({"subset": subset, "label": label, "seed": seed, **summarize(e)})
        engine_tables.append(e)
    # Average error contributions across seeds, then equal-weight engines.
    # This is not an ensemble of predictions; both labels are explicit.
    if any(not e.index.equals(engine_tables[0].index) for e in engine_tables):
        raise ValueError("Different engine populations across seeds")
    pooled = sum(engine_tables)/len(engine_tables)
    return pooled, seed_rows


def screen(root, frames, risk=False, prior="full"):
    suffix = "_risk" if risk else ""
    labels = [x+suffix for x in ("pe50", "pe100", "scheduled", "fixed_trend", "learned_trend")]
    rows, results, refs = [], {}, {}
    positive, tail_ok = True, True
    previous = None
    if risk:
        previous = json.loads((Path(root)/"analysis"/"screen.json").read_text(encoding="utf-8"))
        if previous.get("next") != "TAIL_SCREEN":
            raise ValueError("The single tail rescue was not authorized by the frozen screening rule")
    for subset in ("FD002", "FD004"):
        tables = {}
        for label in labels:
            table, r = aggregate(frames, subset, label, [101], [0, 2], labels[0])
            tables[label] = table
            rows.extend(r)
        reference = min(labels[:3], key=lambda x: summarize(tables[x])["rmse"])
        controls = [reference, "fixed_trend"+suffix]
        if risk:
            old_ref = previous["references"][subset][0]
            tables[old_ref], old_rows = aggregate(frames, subset, old_ref, [101], [0, 2], labels[0])
            rows.extend(old_rows)
            controls.append(old_ref)
        controls = list(dict.fromkeys(controls))
        refs[subset] = controls
        contrasts = {c: paired_comparison(tables["learned_trend"+suffix], tables[c]) for c in controls}
        results[subset] = contrasts
        # Screening is deliberately a directional gate, not confirmation.
        positive &= all(x["gain_percent"] > 0 for x in contrasts.values())
        # Both the engineering reference and same-information fixed trend
        # must avoid a material MAE/NASA regression at screening.
        tail_ok &= all(x["mae_ratio"] <= 1.005 and x["nasa_ratio"] <= 1.005 for x in contrasts.values())
    gain = np.mean([results[s][refs[s][0]]["gain_percent"] for s in refs])
    positive = bool(positive and gain >= 1.0)
    next_step = "CONFIRM" if positive and tail_ok else ("TAIL_SCREEN" if positive and not risk else "STOP_NO_GAIN")
    report = {"status": "DEVELOPMENT_ONLY", "next": next_step, "prior": prior,
              "candidate": "learned_trend"+suffix, "references": refs,
              "mean_reference_gain_percent": float(gain), "comparisons": results,
              "code_sha256": code_fingerprint(), "seeds": [101], "folds": [0, 2],
              "not_a_blind_test": True}
    name = "screen_risk" if risk else "screen"
    out = Path(root)/"analysis"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f"{name}_metrics.csv", index=False)
    write_json(out/f"{name}.json", report)
    if next_step == "CONFIRM":
        write_json(out/"promotion.json", report)
    return report


def confirmation(root, frames, prior="full"):
    promotion = json.loads((Path(root)/"analysis"/"promotion.json").read_text(encoding="utf-8"))
    if promotion["code_sha256"] != code_fingerprint() or promotion["prior"] != prior:
        raise ValueError("Promotion belongs to another protocol")
    candidate = promotion["candidate"]
    contrasts, rows, stats_save, directional = {}, [], [], {}
    tables_by_subset = {}
    pass_all = True
    for subset, controls in promotion["references"].items():
        tables, seed_results = {}, {}
        for label in [candidate]+controls:
            e, r = aggregate(frames, subset, label, [101, 202, 303], list(range(5)), candidate)
            tables[label], seed_results[label] = e, r
            rows.extend(r)
            stats_save.append(e.reset_index().assign(subset=subset, label=label))
        tables_by_subset[subset] = tables
        contrasts[subset] = {label: paired_comparison(tables[candidate], tables[label]) for label in controls}
        directional[subset] = {}
        for control in controls:
            wins = sum(a["rmse"] < b["rmse"] for a, b in zip(seed_results[candidate], seed_results[control]))
            directional[subset][control] = wins
            x = contrasts[subset][control]
            pass_all &= (x["gain_percent"] > 0 and wins >= 2
                         and x["delta_rmse_ci95_cycles"][1] < 0
                         and x["mae_ratio"] <= 1.005 and x["nasa_ratio"] <= 1.005)
    gains = [contrasts[s][promotion["references"][s][0]]["gain_percent"] for s in contrasts]
    # Stratify by subset, resample engines, retain all seed contributions.
    rng = np.random.default_rng(20261106)
    bootstrap_gains = []
    for subset, tables in tables_by_subset.items():
        a = tables[candidate].mse.to_numpy()
        b = tables[promotion["references"][subset][0]].mse.to_numpy()
        ix = rng.integers(0, len(a), size=(5000, len(a)))
        bootstrap_gains.append(100*(1-np.sqrt(a[ix].mean(1))/np.sqrt(b[ix].mean(1))))
    gain_ci = np.quantile(np.mean(bootstrap_gains, axis=0), [.025, .975]).tolist()
    pass_all &= np.mean(gains) >= 3.0 and gain_ci[0] > 0
    out = Path(root)/"analysis"
    pd.DataFrame(rows).to_csv(out/"seed_metrics.csv", index=False)
    pd.concat(stats_save, ignore_index=True).to_csv(out/"per_engine_metrics.csv", index=False)
    report = {"status": "DEVELOPMENT_GATE_PASS" if pass_all else "NO_GAIN_OR_TAIL_FAILURE",
              "candidate": candidate, "comparisons": contrasts, "seed_wins_of_3": directional,
              "mean_reference_gain_percent": float(np.mean(gains)), "gain_ci95_percent": gain_ci,
              "code_sha256": code_fingerprint(), "prior": prior,
              "next": "ABLATION_AND_GUARDS" if pass_all else "STOP_NO_GAIN",
              "not_a_blind_test": True, "novelty_proven": False, "submission_ready": False}
    write_json(out/"confirmation.json", report)
    return report


def support(root, frames, prior="full"):
    gate = json.loads((Path(root)/"analysis"/"confirmation.json").read_text(encoding="utf-8"))
    if gate["status"] != "DEVELOPMENT_GATE_PASS" or gate["prior"] != prior:
        raise ValueError("Support analysis requires a matching confirmation pass")
    candidate = gate["candidate"]
    suffix = "_risk" if candidate.endswith("_risk") else ""
    rows, ablations, guards = [], {}, {}
    passed = True
    for s in ("FD002", "FD004"):
        a, r = aggregate(frames, s, candidate, [101], list(range(5)))
        rows.extend(r)
        b, r = aggregate(frames, s, "unmatched_trend"+suffix, [101], list(range(5)), candidate)
        rows.extend(r)
        ablations[s] = paired_comparison(a, b)
    for s in ("FD001", "FD003"):
        tables = {}
        for label in ("pe50"+suffix, "pe100"+suffix, "fixed_trend"+suffix, candidate):
            e, r = aggregate(frames, s, label, [101], list(range(5)), candidate)
            tables[label] = e
            rows.extend(r)
        reference = min(("pe50"+suffix, "pe100"+suffix), key=lambda x: summarize(tables[x])["rmse"])
        c = paired_comparison(tables[candidate], tables[reference])
        guards[s] = {"reference": reference, **c}
        passed &= c["gain_percent"] >= -1.0 and c["mae_ratio"] <= 1.01 and c["nasa_ratio"] <= 1.02
    # This distinguishes a proved algebraic property from performance attribution.
    attribution = (all(x["gain_percent"] > 0 for x in ablations.values())
                   and any(x["delta_rmse_ci95_cycles"][1] < 0 for x in ablations.values()))
    result = {"status": "SUPPORT_PASS" if passed and attribution else "SUPPORT_INCOMPLETE_OR_FAILED",
              "single_seed_guards_pass": bool(passed), "condition_attribution_supported": bool(attribution),
              "guard_comparisons": guards, "unmatched_ablation": ablations,
              "code_sha256": code_fingerprint(), "candidate": candidate, "prior": prior,
              "novelty_proven": False, "submission_ready": False,
              "next": "FROZEN_BENCHMARK_EVALUATION" if passed and attribution else "REPORT_BOUNDARY_AND_STOP"}
    pd.DataFrame(rows).to_csv(Path(root)/"analysis"/"support_metrics.csv", index=False)
    write_json(Path(root)/"analysis"/"support.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="outputs")
    p.add_argument("--stage", choices=("screen", "tail", "confirm", "support"), required=True)
    p.add_argument("--prior", choices=("full", "none"), default="full")
    args = p.parse_args()
    try:
        frames, _ = read_runs(args.out, args.prior)
        if args.stage == "confirm":
            report = confirmation(args.out, frames, args.prior)
        elif args.stage == "support":
            report = support(args.out, frames, args.prior)
        else:
            report = screen(args.out, frames, args.stage == "tail", args.prior)
    except KeyError as exc:
        report = {"status": "INCOMPLETE", "missing": str(exc), "next": "COMPLETE_EXISTING_QUEUE"}
        dest = Path(args.out)/"analysis"
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest/f"{args.stage}_incomplete.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
