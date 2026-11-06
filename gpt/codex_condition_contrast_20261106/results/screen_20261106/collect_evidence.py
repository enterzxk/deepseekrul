"""Export checked trial and paired-engine summaries from a finished CCT run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from tim_contrast.analyze import align_queries, materialize, read_runs
from tim_contrast.data import sha256, split_units
from tim_contrast.metrics import engine_statistics, paired_comparison, summarize
from tim_contrast.run import code_fingerprint, write_json
from tim_contrast.vendor.cmapss import load_cmapss_train


def audit_trials(frames, metadata, root, data_root):
    records, source = [], {}
    for subset in sorted({key[0] for key in frames}):
        frame = load_cmapss_train(data_root, subset)
        lifetimes = frame.groupby("unit").cycle.max()
        raw_hash = sha256(data_root / f"train_{subset}.txt")
        source[subset] = {
            "train_sha256": raw_hash,
            "train_engines": len(lifetimes),
            "train_rows": len(frame),
        }
        for key, predictions in frames.items():
            s, label, seed, fold = key
            if s != subset:
                continue
            meta = metadata[key]
            _, valid = split_units(frame, fold)
            if meta["train_sha256"] != raw_hash or meta["valid_units"] != valid.tolist():
                raise ValueError(f"Input or fold mismatch in {key}")
            queries = [
                (int(unit), end, min(125, int(lifetimes[unit]) - end))
                for unit in valid for end in range(50, int(lifetimes[unit]) + 1)
            ]
            expected = pd.DataFrame(queries, columns=["unit", "end_cycle", "y_true"])
            align_queries(predictions, expected)
            if meta["epochs"] != 80 or meta["epoch_selection"] != "fixed_final":
                raise ValueError(f"Incomplete training budget in {key}")
            dest = root / subset / f"{label}_f{fold}_s{seed}_full"
            receipt = json.loads((dest / "done.json").read_text())
            for name, expected_sha in receipt["artifacts"].items():
                if sha256(dest / name) != expected_sha:
                    raise ValueError(f"Artifact hash mismatch in {key}: {name}")
            # Restore the float32 prediction values used for the original run metrics.
            scored = predictions.assign(y_pred=predictions.y_pred.astype(np.float32))
            metrics = summarize(engine_statistics(scored))
            for name in ("rmse", "mae", "nasa", "bias"):
                if not np.isclose(metrics[name], meta[name], atol=1e-6, rtol=1e-6):
                    raise ValueError(f"Stored metric mismatch in {key}: {name}")
            records.append({
                "subset": subset, "label": label, "seed": seed, "fold": fold,
                **metrics, "abs_bias": abs(metrics["bias"]), "epochs": meta["epochs"],
                "parameters": meta["parameters"], "n_train_queries": meta["n_training_queries"],
                "elapsed_seconds": meta["elapsed_seconds"], "train_sha256": raw_hash,
                "started_utc": meta["started"], "completed_utc": meta["completed"],
            })
    return records, source


def comparison_specs(analysis):
    for stage, filename in (
        ("screen", "screen.json"), ("tail", "screen_risk.json"),
        ("confirmation", "confirmation.json"),
    ):
        path = analysis / filename
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        folds = list(range(5)) if stage == "confirmation" else report["folds"]
        seeds = [101, 202, 303] if stage == "confirmation" else report["seeds"]
        for subset, refs in report["comparisons"].items():
            for reference in refs:
                yield stage, subset, report["candidate"], reference, folds, seeds
    path = analysis / "support.json"
    if path.is_file():
        report = json.loads(path.read_text())
        candidate = report["candidate"]
        suffix = "_risk" if candidate.endswith("_risk") else ""
        for subset, _ in report["unmatched_ablation"].items():
            yield "mechanism", subset, candidate, "unmatched_trend" + suffix, list(range(5)), [101]
        for subset, guard in report["guard_comparisons"].items():
            yield "guard", subset, candidate, guard["reference"], list(range(5)), [101]


def export_pairs(frames, analysis, destination):
    pairs, comparisons, strata = [], [], []
    for stage, subset, candidate, reference, folds, seeds in comparison_specs(analysis):
        tag = {"stage": stage, "subset": subset, "candidate": candidate, "reference": reference}
        a_list, b_list = [], []
        for seed in seeds:
            a, b = align_queries(
                materialize(frames, subset, candidate, seed, folds),
                materialize(frames, subset, reference, seed, folds),
            )
            fold_by_unit = a.groupby("unit").fold.first()
            ea, eb = engine_statistics(a), engine_statistics(b)
            a_list.append(ea)
            b_list.append(eb)
            paired = ea.add_prefix("candidate_").join(eb.add_prefix("reference_"), validate="one_to_one")
            paired["fold"] = fold_by_unit
            paired["delta_engine_rmse"] = np.sqrt(ea.mse) - np.sqrt(eb.mse)
            paired["delta_engine_nasa"] = ea.nasa - eb.nasa
            paired["candidate_nasa_share"] = ea.nasa / max(ea.nasa.sum(), 1e-300)
            pairs.append(paired.reset_index().assign(**tag, seed=seed))
            for age, keep in (("t<100", a.end_cycle < 100), ("t>=100", a.end_cycle >= 100)):
                if keep.any():
                    for arm, frame in ((candidate, a), (reference, b)):
                        stats = summarize(engine_statistics(frame.loc[keep]))
                        strata.append({**tag, "seed": seed, "age": age, "label": arm, **stats})
        ap = sum(a_list) / len(seeds)
        bp = sum(b_list) / len(seeds)
        delta = paired_comparison(ap, bp)
        lo, hi = delta.pop("delta_rmse_ci95_cycles")
        comparisons.append({
            **tag, **delta, "rmse_ci95_lo": lo, "rmse_ci95_hi": hi,
            "engines": len(ap), "queries_per_seed": int(ap.queries.sum()),
            "seeds": ",".join(map(str, seeds)), "folds": ",".join(map(str, folds)),
            "candidate_rmse": summarize(ap)["rmse"], "reference_rmse": summarize(bp)["rmse"],
            "engine_rmse_wins": int((ap.mse < bp.mse).sum()),
        })
    if pairs:
        table = pd.concat(pairs, ignore_index=True)
        table.to_csv(destination / "paired_engines.csv", index=False)
        worst = (table.sort_values("candidate_nasa", ascending=False)
                 .groupby(["stage", "subset", "reference", "seed"], sort=False)
                 .head(5))
        worst.to_csv(destination / "worst_engines_nasa.csv", index=False)
        pd.DataFrame(comparisons).to_csv(destination / "paired_comparisons.csv", index=False)
        pd.DataFrame(strata).to_csv(destination / "age_strata_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    state_path = root / "PIPELINE_STATE.json"
    state = json.loads(state_path.read_text())
    terminal = {
        "STOP_NO_GAIN", "STOP_MECHANISM_OR_SCOPE", "BUDGET_STOP_PARTIAL",
        "FAILED_STOP", "DEVELOPMENT_COMPLETE_OFFICIAL_NOT_RUN",
        "BENCHMARK_EVALUATED_REVIEW_REQUIRED",
    }
    if state["status"] not in terminal:
        raise ValueError(f"Expected a terminal pipeline state, got {state['status']}")
    frames, metadata = read_runs(root)
    records, inputs = audit_trials(frames, metadata, root, args.data_root)
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(state_path, destination / state_path.name)
    shutil.copyfile(root.parent / "launch.json", destination / "launch.json")
    if records:
        pd.DataFrame(records).sort_values(["subset", "seed", "fold", "label"]).to_csv(
            destination / "trial_metrics.csv", index=False,
        )
    analysis = root / "analysis"
    for name in (
        "screen.json", "screen_metrics.csv", "screen_risk.json", "screen_risk_metrics.csv",
        "promotion.json", "confirmation.json", "seed_metrics.csv", "per_engine_metrics.csv",
        "support.json", "support_metrics.csv",
    ):
        if (analysis / name).is_file():
            shutil.copyfile(analysis / name, destination / name)
    for name in ("official_metrics.csv", "sealed.json", "scored.json"):
        path = root / "official" / name
        if path.is_file():
            shutil.copyfile(path, destination / name)
    export_pairs(frames, analysis, destination)
    write_json(destination / "EVIDENCE_AUDIT.json", {
        "pipeline_status": state["status"], "code_sha256": code_fingerprint(),
        "validated_completed_trials": len(records), "training_inputs": inputs,
        "query_population_and_labels_checked": bool(records),
        "completed_artifact_hashes_checked": bool(records),
        "exporter_reads_official_labels": False,
        "official_scores_exported": (destination / "official_metrics.csv").is_file(),
    })
    files = sorted(path for path in destination.iterdir() if path.is_file())
    (destination / "MANIFEST.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files), encoding="utf-8",
    )
    print(json.dumps({"status": state["status"], "trials": len(records),
                      "files": len(files) + 1, "path": str(destination)}))


if __name__ == "__main__":
    main()
