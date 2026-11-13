"""Descriptive development-only condition/sensor audit, separate from model selection.

A fixed quadratic condition regression is a diagnostic, not a RUL baseline or HI.
Its results do not change the already-frozen neural experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np


SENSORS = ("Wf", "Nf", "Nc", "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24", "Ps30", "P40", "P50")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(w):
    columns = [np.ones(len(w))] + [w[:, k] for k in range(4)]
    columns += [w[:, i]*w[:, j] for i in range(4) for j in range(i, 4)]
    return np.stack(columns, -1)


def correlation(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []; fitted = []; sources = {}
    for subset in ("DS02", "DS05", "DS06", "DS07"):
        path = args.cache/f"{subset}_dev.npz"
        manifest = json.loads((args.cache/f"{subset}_dev_manifest.json").read_text())
        if sha(path) != manifest["cache_sha256"]:
            raise AssertionError("cache was modified")
        with np.load(path, allow_pickle=False) as z:
            # No RUL, health, simulator parameters or official test are read.
            raw, unit, cycle = z["raw"], z["unit"], z["cycle"]
        sources[subset] = manifest["cache_sha256"]
        for held in sorted(set(map(int, unit))):
            train = unit != held; test = unit == held
            inputs = raw[train].reshape(-1, 18).astype(np.float64)
            mu = inputs.mean(0); sd = np.maximum(inputs.std(0), 1e-6)
            train_scaled = (inputs-mu)/sd
            phi = design(train_scaled[:, :4]); x = train_scaled[:, 4:]
            gram = phi.T @ phi / len(phi)
            regularizer = np.eye(15)*1e-3; regularizer[0, 0] = 0
            coef = np.linalg.solve(gram+regularizer, phi.T @ x/len(phi))
            observed = (raw[test].astype(np.float64)-mu)/sd
            pred = (design(observed[:, :, :4].reshape(-1, 4)) @ coef).reshape(-1, 512, 14)
            sensors = observed[:, :, 4:]; residual = sensors-pred
            var = sensors.reshape(-1, 14).var(0)
            r2 = 1-(residual**2).mean((0, 1))/np.maximum(var, 1e-12)
            for k, name in enumerate(SENSORS):
                rows.append({"subset": subset, "held_out_engine": held, "sensor": name,
                             "heldout_condition_R2": float(r2[k]),
                             "raw_flight_mean_age_correlation": correlation(sensors[:, :, k].mean(1), cycle[test]),
                             "conditional_residual_mean_age_correlation": correlation(residual[:, :, k].mean(1), cycle[test]),
                             "heldout_flights": int(test.sum())})
            fitted.append({"subset": subset, "held_out_engine": held,
                           "training_engines": sorted(set(map(int, unit[train]))),
                           "mu": mu.tolist(), "sd": sd.tolist(), "coefficients": coef.tolist()})
    with open(args.output/"sensor_condition_audit.csv", "x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for subset in sources:
        selected = [r for r in rows if r["subset"] == subset]
        summary.append({"subset": subset,
                        "median_condition_R2_across_engine_sensor_pairs": float(np.median([r["heldout_condition_R2"] for r in selected])),
                        "median_abs_raw_age_correlation": float(np.median([abs(r["raw_flight_mean_age_correlation"]) for r in selected if r["raw_flight_mean_age_correlation"] is not None])),
                        "median_abs_residual_age_correlation": float(np.median([abs(r["conditional_residual_mean_age_correlation"]) for r in selected if r["conditional_residual_mean_age_correlation"] is not None]))})
    report = {"purpose": "descriptive_condition_response_audit_not_RUL_comparison_or_health_identification",
              "cache_sha256": sources, "fit": "quadratic_W_to_Xs_ridge0.001; five-engine_fit_one-engine_audit",
              "RUL_labels_read": False, "official_test_read": False, "neural_optimizer_updates": 0,
              "not_used_to_change_frozen_experiment": True, "summary": summary,
              "interpretation_limit": "residual includes model error, operating dynamics, initial offsets and degradation; correlation with age is not a physical HI proof",
              "diagnostic_source_sha256": sha(__file__)}
    for name, value in (("DATA_DIAGNOSTICS.json", report), ("FITTED_DIAGNOSTIC_MODELS.json", fitted)):
        with open(args.output/name, "x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    lines = ["# Development sensor/condition diagnostics", "", "Descriptive audit; no RUL labels or TEST decoded. Not a RUL comparison or module success claim.",
             "", "| Subset | Median condition R² | Median abs raw-age correlation | Median abs residual-age correlation |", "|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['subset']} | {r['median_condition_R2_across_engine_sensor_pairs']:.4f} | {r['median_abs_raw_age_correlation']:.4f} | {r['median_abs_residual_age_correlation']:.4f} |")
    lines += ["", report["interpretation_limit"], "", "All fits use only the five training engines. Complete fold/sensor values and fitted coefficients accompany this file."]
    (args.output/"DATA_DIAGNOSTICS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
