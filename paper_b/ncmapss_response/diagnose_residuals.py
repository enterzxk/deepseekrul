"""Fixed post-failure D1: CPU training-side residual diagnostics, never RUL.

Only raw/unit/cycle are decoded from development NPZs. There is no torch,
optimizer, TEST loader, target access, HPO, or modification of frozen models.
The protocol is registered separately before execution. All120 overlapping
outer/inner crossfits are executed; they are not120 independent observations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# Process-local settings, before NumPy import; do not change a user profile.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
import numpy as np

PROTOCOL_FILE = Path(__file__).with_name("RESIDUAL_DIAGNOSTIC_PROTOCOL.json")
PROTOCOL_SHA256 = "ca0177ded81b6979b677cf7501e420f6f4dd3121d39389b981adf5008fc4c9ff"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_new(path, value):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def quadratic(w):
    """15 columns in the registered order, including the unpenalized constant."""
    columns = [np.ones(w.shape[:-1], np.float64)] + [w[..., k] for k in range(4)]
    columns += [w[..., i] * w[..., j] for i in range(4) for j in range(i, 4)]
    return np.stack(columns, axis=-1)


def dynamic_features(w):
    require(w.ndim == 3 and w.shape[1:] == (512, 4), "wrong flight W shape")
    delta = np.zeros_like(w)
    delta[:, 1:] = w[:, 1:] - w[:, :-1]
    columns = [quadratic(w), delta,
               np.stack([w[..., i] * delta[..., j] for i in range(4) for j in range(4)], axis=-1)]
    position = np.broadcast_to(np.arange(512, dtype=np.float64) / 511., w.shape[:2])
    columns += [np.stack((position, position ** 2), axis=-1)]
    result = np.concatenate(columns, axis=-1)
    require(result.shape == (*w.shape[:2], 37), "wrong dynamic feature count")
    return result


def ridge(design, target):
    require(design.ndim == target.ndim == 2 and len(design) == len(target), "ridge shape mismatch")
    regularizer = np.eye(design.shape[1], dtype=np.float64) * .001
    regularizer[0, 0] = 0.
    gram = design.T @ design / len(design)
    coef = np.linalg.solve(gram + regularizer, design.T @ target / len(design))
    require(bool(np.isfinite(coef).all()), "nonfinite closed-form ridge coefficients")
    return coef, float(np.linalg.cond(gram + regularizer))


def correlation(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    aa, bb = a - a.mean(), b - b.mean()
    denom = float(np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))
    if denom == 0:
        return None
    return float(np.dot(aa, bb) / denom)


def load_development(cache, subset, protocol):
    manifest_path = cache / f"{subset}_dev_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["official_test_decoded"] is False and manifest["decoded_keys"] ==
            ["A_dev", "W_dev", "X_s_dev", "Y_dev"], "not an approved development cache")
    path = cache / f"{subset}_dev.npz"
    digest = sha(path)
    require(digest == manifest["cache_sha256"] == protocol["expected_cache_sha256"][subset], "wrong frozen development cache")
    with np.load(path, allow_pickle=False) as archive:
        # Intentionally do not request y, source indices, duration, or any other key.
        raw, unit, cycle = archive["raw"], archive["unit"], archive["cycle"]
    require(raw.shape == (len(unit), 512, 18) and cycle.shape == unit.shape, "wrong development shape")
    require(bool(np.isfinite(raw).all() and np.isfinite(cycle).all()), "nonfinite allowed data")
    units = sorted(map(int, np.unique(unit)))
    require(len(units) == 6, "requires six development engines")
    for u in units:
        require(bool((np.diff(cycle[unit == u]) > 0).all()), "nonincreasing flight cycle")
    return raw, unit, cycle, units, {"cache_sha256": digest, "manifest_sha256": sha(manifest_path),
                                    "cache": str(path.resolve()), "decoded_npz_keys": ["raw", "unit", "cycle"],
                                    "flights": len(unit), "engines": units}


def fit_predict(train_raw, inner_raw):
    """No outer-array argument exists; fit both models on train-four only."""
    train = train_raw.astype(np.float64)
    mu = train.mean(axis=(0, 1)); sd_unfloored = train.std(axis=(0, 1))
    sd = np.maximum(sd_unfloored, 1e-6)
    train = (train - mu) / sd
    inner = (inner_raw.astype(np.float64) - mu) / sd
    train_phi = quadratic(train[..., :4]).reshape(-1, 15)
    inner_phi = quadratic(inner[..., :4]).reshape(-1, 15)
    coef, static_condition = ridge(train_phi, train[..., 4:].reshape(-1, 14))
    train_residual = train[..., 4:] - (train_phi @ coef).reshape(*train.shape[:2], 14)
    residual = inner[..., 4:] - (inner_phi @ coef).reshape(*inner.shape[:2], 14)
    empirical_unfloored = train_residual.std(axis=(0, 1))
    empirical_scale = np.maximum(empirical_unfloored, 1e-6)
    train_dynamic = dynamic_features(train[..., :4]).reshape(-1, 37)
    dynamic_mu = train_dynamic.mean(axis=0)
    dynamic_sd_unfloored = train_dynamic.std(axis=0)
    dynamic_sd = np.maximum(dynamic_sd_unfloored, 1e-6)
    train_dynamic = (train_dynamic - dynamic_mu) / dynamic_sd
    # Original constant1 becomes zero; add the separate, unpenalized intercept.
    require(bool((train_dynamic[:, 0] == 0).all()), "standardized constant is not zero")
    design = np.column_stack((np.ones(len(train_dynamic)), train_dynamic))
    dynamic_coef, dynamic_condition = ridge(design, train_residual.reshape(-1, 14))
    inner_dynamic = dynamic_features(inner[..., :4]).reshape(-1, 37)
    inner_dynamic = (inner_dynamic - dynamic_mu) / dynamic_sd
    inner_design = np.column_stack((np.ones(len(inner_dynamic)), inner_dynamic))
    dynamic_prediction = (inner_design @ dynamic_coef).reshape(*inner.shape[:2], 14)
    require(all(bool(np.isfinite(v).all()) for v in (residual, empirical_scale, dynamic_prediction)), "nonfinite diagnostic prediction")
    fit = {"raw_mu": mu.tolist(), "raw_sd": sd.tolist(), "raw_sd_unfloored": sd_unfloored.tolist(),
           "static_coefficients": coef.tolist(), "static_gram_condition_number": static_condition,
           "empirical_residual_scale": empirical_scale.tolist(),
           "empirical_residual_scale_unfloored": empirical_unfloored.tolist(),
           "dynamic_feature_mu": dynamic_mu.tolist(), "dynamic_feature_sd": dynamic_sd.tolist(),
           "dynamic_feature_sd_unfloored": dynamic_sd_unfloored.tolist(),
           "dynamic_coefficients": dynamic_coef.tolist(), "dynamic_gram_condition_number": dynamic_condition,
           "training_flights": len(train_raw), "prediction_flights": len(inner_raw),
           "training_rows": len(train_phi), "prediction_rows": len(inner_phi)}
    return fit, inner[..., 4:], residual, residual / empirical_scale, dynamic_prediction


def sensor_rows(subset, outer, inner, cycle, raw, observed, residual, scaled, dynamic, fit, names):
    rows = []
    observed_std, residual_std = observed.std(axis=(0, 1)), residual.std(axis=(0, 1))
    mse0 = (residual ** 2).mean(axis=(0, 1)); mse_dynamic = ((residual - dynamic) ** 2).mean(axis=(0, 1))
    equality_max = 0.
    for k, name in enumerate(names):
        raw_corr = correlation(raw[..., k + 4].mean(axis=1), cycle)
        resid_corr = correlation(residual[..., k].mean(axis=1), cycle)
        scaled_corr = correlation(scaled[..., k].mean(axis=1), cycle)
        require((resid_corr is None) == (scaled_corr is None), "scaled correlation definedness changed")
        if resid_corr is not None:
            difference = abs(resid_corr - scaled_corr)
            require(difference <= 1e-10, "constant scale changed a Pearson age correlation")
            equality_max = max(equality_max, difference)
        rows.append({"subset": subset, "excluded_outer_engine": outer, "predicted_inner_engine": inner,
                     "sensor_index": k, "sensor": name, "inner_flights": len(cycle),
                     "raw_std_physical": float(raw[..., k + 4].std()),
                     "raw_std_standardized": float(observed_std[k]),
                     "residual_std_standardized": float(residual_std[k]),
                     "residual_std_physical": float(residual_std[k] * fit["raw_sd"][k + 4]),
                     "residual_std_raw_std_ratio": float(residual_std[k] / observed_std[k]) if observed_std[k] > 0 else None,
                     "raw_flight_mean_age_correlation": raw_corr,
                     "residual_flight_mean_age_correlation": resid_corr,
                     "empirical_scaled_residual_mean_age_correlation": scaled_corr,
                     "empirical_residual_scale_standardized": fit["empirical_residual_scale"][k],
                     "residual_zero_predictor_mse": float(mse0[k]),
                     "dynamic_residual_prediction_mse": float(mse_dynamic[k]),
                     "dynamic_error_reduction": float(1. - mse_dynamic[k] / mse0[k]) if mse0[k] > 0 else None})
    return rows, equality_max


def group_summary(rows, **identity):
    ratios = [r["residual_std_raw_std_ratio"] for r in rows]
    improvements = [r["dynamic_error_reduction"] for r in rows]
    ratio_defined = all(v is not None for v in ratios)
    improvement_defined = all(v is not None for v in improvements)
    result = {**identity, "engine_sensor_rows": len(rows), "all_ratios_defined": ratio_defined,
              "all_dynamic_improvements_defined": improvement_defined,
              "fraction_residual_std_raw_std_below_0_1": float(np.mean([v < .1 for v in ratios])) if ratio_defined else None,
              "median_residual_std_raw_std_ratio": float(np.median(ratios)) if ratio_defined else None,
              "median_dynamic_error_reduction": float(np.median(improvements)) if improvement_defined else None}
    for key in ("raw_flight_mean_age_correlation", "residual_flight_mean_age_correlation", "empirical_scaled_residual_mean_age_correlation"):
        valid = [abs(r[key]) for r in rows if r[key] is not None]
        result["median_abs_" + key] = float(np.median(valid)) if valid else None
        result["defined_" + key] = len(valid)
    return result


def write_csv(path, rows):
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_report(output, report):
    lines = ["# Post-failure D1 — training-side residual diagnostic", "",
             "Exploratory mechanism diagnostic after observing the closed common-route failure. No RUL/TEST labels were decoded; no new model was trained.", "",
             f"Execution: **{report['status']}**, {report['completed_fits']}/120 fixed crossfits; CPU elapsed {report['elapsed_seconds']:.2f} s.", "",
             "|Subset|Crossfit engine-sensor rows|Fraction residual/raw std <0.1|Median residual/raw std|Median dynamic error reduction|", "|---|---:|---:|---:|---:|"]
    for r in report["subset_summaries"]:
        values = [r[k] for k in ("fraction_residual_std_raw_std_below_0_1", "median_residual_std_raw_std_ratio", "median_dynamic_error_reduction")]
        text = ["undefined" if v is None else f"{v:.6f}" for v in values]
        lines.append(f"|{r['subset']}|{r['engine_sensor_rows']}|{'|'.join(text)}|")
    lines += ["", f"Fixed scale-suppression gate: **{report['gates']['scale_suppression']}**.",
              f"Fixed dynamic-omission gate: **{report['gates']['dynamic_omission']}** ({report['gates']['dynamic_qualifying_subsets']}).", "",
              "Empirical residual scale is a fitted descriptive scale, not a noise estimate. Dividing by a fixed positive scale cannot change a within-engine/sensor Pearson age correlation; this equality was checked.", "",
              "The dynamic predictor includes the original quadratic W terms, an intercept, position, and differences; its total gain cannot isolate difference terms from recalibration or position effects. Error reduction is versus a zero residual predictor and is not evidence of health or RUL prediction.", "",
              "All120 crossfits overlap in engines/training sets. They are not120 independent samples; no p-value, independence-based CI, or generalization claim is supplied. No feature, ridge, threshold, outer evaluation, or frozen model was selected using this diagnostic."]
    (output / "RESIDUAL_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    require(sha(PROTOCOL_FILE) == PROTOCOL_SHA256, "registered D1 protocol changed")
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    output, cache = args.output.resolve(), args.cache.resolve()
    require(not output.exists() and not output.is_relative_to(cache) and not cache.is_relative_to(output), "output exists or overlaps input")
    registration_path = output.parent / "REGISTRATION.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    require(registration["protocol_sha256"] == PROTOCOL_SHA256 and registration["D1_execution_started"] is False, "missing pre-execution registration")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); rows = []; sources = {}; fits = []; correlation_max_difference = 0.
    runtime = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
               "platform": platform.platform(), "thread_environment": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}}
    try:
        from threadpoolctl import threadpool_info
        runtime["threadpools"] = threadpool_info()
    except ImportError:
        runtime["threadpools"] = "threadpoolctl unavailable; environment fixed before NumPy import"
    identity = {"protocol_sha256": PROTOCOL_SHA256, "diagnostic_source_sha256": sha(__file__),
                "registration_sha256": sha(registration_path), "started_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime, "argv": sys.argv, "RUL_labels_read": False, "official_TEST_read": False,
                "decoded_npz_keys": ["raw", "unit", "cycle"], "iterative_optimizer_steps": 0, "GPU_jobs": 0}
    dump_new(output / "EXECUTION.json", identity)
    dump_new(output / "PROTOCOL_COPY.json", protocol)
    try:
        for subset in protocol["subsets"]:
            raw, unit, cycle, units, source = load_development(cache, subset, protocol)
            sources[subset] = source
            for outer in units:
                for inner in [u for u in units if u != outer]:
                    require(time.monotonic() - started < protocol["budget"]["maximum_seconds"], "CPU diagnostic budget exceeded")
                    fold_started = time.monotonic()
                    training_units = [u for u in units if u not in (outer, inner)]
                    require(len(training_units) == 4 and outer not in training_units and inner not in training_units, "invalid train-four split")
                    train_keep, inner_keep = np.isin(unit, training_units), unit == inner
                    fit, observed, residual, scaled, dynamic = fit_predict(raw[train_keep], raw[inner_keep])
                    fold_rows, difference = sensor_rows(subset, outer, inner, cycle[inner_keep], raw[inner_keep], observed, residual, scaled, dynamic, fit, protocol["Xs_order"])
                    correlation_max_difference = max(correlation_max_difference, difference)
                    rows.extend(fold_rows)
                    name = f"{subset}_exclude{outer}_predict{inner}"
                    folder = output / "folds" / name
                    folder.mkdir(parents=True, exist_ok=False)
                    fit_identity = {"subset": subset, "excluded_outer_engine": outer, "predicted_inner_engine": inner,
                                    "training_engines": training_units, "cache_sha256": source["cache_sha256"],
                                    "protocol_sha256": PROTOCOL_SHA256, "source_sha256": identity["diagnostic_source_sha256"],
                                    "sensor_order": protocol["Xs_order"], "seconds": time.monotonic() - fold_started}
                    dump_new(folder / "MODEL.json", {**fit_identity, **fit})
                    with open(folder / "FLIGHT_MEANS.npz", "xb") as f:
                        np.savez_compressed(f, unit=unit[inner_keep], cycle=cycle[inner_keep],
                                            raw_physical=raw[inner_keep, :, 4:].astype(np.float64).mean(axis=1),
                                            residual_standardized=residual.mean(axis=1),
                                            empirical_scaled_residual=scaled.mean(axis=1),
                                            dynamic_prediction_standardized=dynamic.mean(axis=1))
                    dump_new(folder / "FOLD_SUCCESS.json", {"fit": name,
                              "files_sha256": {n: sha(folder / n) for n in ("MODEL.json", "FLIGHT_MEANS.npz")}})
                    fits.append(fit_identity)
                    require(time.monotonic() - started < protocol["budget"]["maximum_seconds"], "CPU diagnostic budget exceeded")
                print(json.dumps({"subset": subset, "excluded_outer_engine": outer, "completed_fits": len(fits), "elapsed_seconds": time.monotonic() - started}), flush=True)
            del raw, unit, cycle
        require(len(fits) == 120 and len(rows) == 120 * 14, "incomplete fixed diagnostic matrix")
        subset_summaries = [group_summary([r for r in rows if r["subset"] == s], subset=s) for s in protocol["subsets"]]
        outer_summaries = [group_summary([r for r in rows if r["subset"] == s and r["excluded_outer_engine"] == u], subset=s, excluded_outer_engine=u)
                           for s in protocol["subsets"] for u in sources[s]["engines"]]
        pooled = group_summary(rows)
        fraction = pooled["fraction_residual_std_raw_std_below_0_1"]
        qualifying = [r["subset"] for r in subset_summaries if r["median_dynamic_error_reduction"] is not None and r["median_dynamic_error_reduction"] >= .1]
        gates = {"scale_suppression": "UNIDENTIFIABLE" if fraction is None else "MET" if fraction >= .75 else "NOT_MET",
                 "dynamic_omission": "UNIDENTIFIABLE" if any(r["median_dynamic_error_reduction"] is None for r in subset_summaries) else "MET" if len(qualifying) >= 2 else "NOT_MET",
                 "dynamic_qualifying_subsets": qualifying}
        report = {**identity, "status": "COMPLETE", "completed_fits": len(fits), "engine_sensor_rows": len(rows),
                  "elapsed_seconds": time.monotonic() - started, "inputs": sources, "gates": gates,
                  "subset_summaries": subset_summaries, "outer_summaries": outer_summaries, "pooled": pooled,
                  "max_residual_scaled_correlation_difference": correlation_max_difference,
                  "interpretation": protocol["gates"]["interpretation"], "dependence_limit": protocol["dependence_limit"],
                  "no_new_RUL_model_authorized_or_evaluated": True}
        dump_new(output / "RESIDUAL_DIAGNOSTICS.json", report)
        write_csv(output / "engine_sensor_metrics.csv", rows)
        write_csv(output / "subset_summary.csv", subset_summaries)
        write_csv(output / "outer_summary.csv", outer_summaries)
        write_csv(output / "diagnostic_fit_index.csv", fits)
        write_report(output, report)
        files = {str(f.relative_to(output)).replace('\\', '/'): sha(f) for f in sorted(output.rglob("*")) if f.is_file()}
        dump_new(output / "SUCCESS.json", {"status": "COMPLETE", "completed_fits": 120, "files_sha256": files,
                  "protocol_sha256": PROTOCOL_SHA256, "diagnostic_source_sha256": identity["diagnostic_source_sha256"]})
        print(json.dumps({"status": report["status"], "completed_fits": len(fits), "elapsed_seconds": report["elapsed_seconds"], "gates": gates, "subset_summaries": subset_summaries}), flush=True)
        return 0
    except BaseException as exc:
        dump_new(output / "FAILED.json", {**identity, "completed_fits": len(fits), "elapsed_seconds": time.monotonic() - started,
                  "type": type(exc).__name__, "message": str(exc), "failure_branch": protocol["failure_branch"]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
