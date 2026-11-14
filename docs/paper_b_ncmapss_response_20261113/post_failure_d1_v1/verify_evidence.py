"""Read-only independent audit of persisted D1 fits; no fitting or RUL access."""
import csv
import hashlib
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "4"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check(a, b, tol=1e-10):
    assert np.allclose(a, b, atol=tol, rtol=tol), (a, b)


def phi(w):
    return np.stack([np.ones(w.shape[:-1])] + [w[..., i] for i in range(4)] +
                    [w[..., i] * w[..., j] for i in range(4) for j in range(i, 4)], -1)


def dynamics(w):
    delta = np.concatenate((np.zeros((len(w), 1, 4)), np.diff(w, axis=1)), axis=1)
    position = np.broadcast_to(np.linspace(0, 1, 512), w.shape[:2])
    return np.concatenate((phi(w), delta,
        np.stack([w[..., i] * delta[..., j] for i in range(4) for j in range(4)], -1),
        position[..., None], (position ** 2)[..., None]), -1)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    seal = read(RESULTS / "SUCCESS.json")
    for relative, expected in seal["files_sha256"].items():
        assert sha(RESULTS / relative) == expected, relative
    protocol = read(ROOT / "paper_b/ncmapss_response/RESIDUAL_DIAGNOSTIC_PROTOCOL.json")
    report = read(RESULTS / "RESIDUAL_DIAGNOSTICS.json")
    execution = read(RESULTS / "EXECUTION.json")
    assert sha(ROOT / "paper_b/ncmapss_response/diagnose_residuals.py") == execution["diagnostic_source_sha256"]
    assert sha(ROOT / "paper_b/ncmapss_response/RESIDUAL_DIAGNOSTIC_PROTOCOL.json") == execution["protocol_sha256"]
    assert read(HERE / "EXIT.json")["exit_code"] == 0
    assert (HERE / "run.stderr.txt").stat().st_size == 0
    with open(RESULTS / "engine_sensor_metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = {(r["subset"], int(r["excluded_outer_engine"]), int(r["predicted_inner_engine"]), r["sensor"]): r for r in rows}
    assert len(rows) == len(index) == 1680
    folder_count = 0
    raw_mean_precision_corr_max_difference = 0.
    for subset in protocol["subsets"]:
        path = ROOT / "paper_b/ncmapss_response/_data" / (subset + "_dev.npz")
        assert sha(path) == protocol["expected_cache_sha256"][subset]
        with np.load(path, allow_pickle=False) as z:
            raw, unit, cycle = z["raw"], z["unit"], z["cycle"]
        units = sorted(map(int, np.unique(unit)))
        for outer in units:
            for inner in (u for u in units if u != outer):
                folder_count += 1
                folder = RESULTS / "folds" / f"{subset}_exclude{outer}_predict{inner}"
                model = read(folder / "MODEL.json")
                assert model["training_engines"] == [u for u in units if u not in (outer, inner)]
                assert model["sensor_order"] == protocol["Xs_order"]
                mask = np.isin(unit, model["training_engines"])
                training = raw[mask].astype(np.float64)
                mu = training.mean((0, 1)); sd = np.maximum(training.std((0, 1)), 1e-6)
                check(model["raw_mu"], mu); check(model["raw_sd"], sd)
                keep = unit == inner
                x = (raw[keep].astype(np.float64) - mu) / sd
                coefficient = np.asarray(model["static_coefficients"])
                dynamic_coefficient = np.asarray(model["dynamic_coefficients"])
                assert coefficient.shape == (15, 14) and dynamic_coefficient.shape == (38, 14)
                assert np.isfinite(coefficient).all() and np.isfinite(dynamic_coefficient).all()
                residual = x[..., 4:] - phi(x[..., :4]) @ coefficient
                dyn = (dynamics(x[..., :4]) - np.asarray(model["dynamic_feature_mu"])) / np.asarray(model["dynamic_feature_sd"])
                design = np.concatenate((np.ones((*dyn.shape[:2], 1)), dyn), -1)
                prediction = design @ dynamic_coefficient
                empirical = np.asarray(model["empirical_residual_scale"])
                assert (empirical >= 1e-6).all()
                ratio = residual.std((0, 1)) / x[..., 4:].std((0, 1))
                zero_mse = np.mean(residual ** 2, (0, 1))
                dynamic_mse = np.mean((residual - prediction) ** 2, (0, 1))
                improvement = 1 - dynamic_mse / zero_mse
                with np.load(folder / "FLIGHT_MEANS.npz", allow_pickle=False) as z:
                    assert np.array_equal(z["unit"], unit[keep]) and np.array_equal(z["cycle"], cycle[keep])
                    check(z["residual_standardized"], residual.mean(1))
                    check(z["empirical_scaled_residual"], (residual / empirical).mean(1))
                    check(z["dynamic_prediction_standardized"], prediction.mean(1))
                for k, sensor in enumerate(protocol["Xs_order"]):
                    row = index[subset, outer, inner, sensor]
                    assert int(row["sensor_index"]) == k
                    check(float(row["residual_std_raw_std_ratio"]), ratio[k])
                    check(float(row["residual_zero_predictor_mse"]), zero_mse[k])
                    check(float(row["dynamic_residual_prediction_mse"]), dynamic_mse[k])
                    check(float(row["dynamic_error_reduction"]), improvement[k])
                    # The frozen D1 metric averages original FP32 observations
                    # before its FP64 Pearson calculation. Preserve that order;
                    # quantify the alternative FP64-mean effect separately.
                    raw_corr = corr(raw[keep, :, k + 4].mean(1), cycle[keep])
                    check(float(row["raw_flight_mean_age_correlation"]), raw_corr)
                    raw_mean_precision_corr_max_difference = max(raw_mean_precision_corr_max_difference,
                        abs(raw_corr - corr(raw[keep, :, k + 4].astype(np.float64).mean(1), cycle[keep])))
                    check(float(row["residual_flight_mean_age_correlation"]), corr(residual[..., k].mean(1), cycle[keep]))
                    check(float(row["empirical_scaled_residual_mean_age_correlation"]), corr((residual[..., k] / empirical[k]).mean(1), cycle[keep]))
    assert folder_count == seal["completed_fits"] == report["completed_fits"] == 120
    fraction = np.mean([float(r["residual_std_raw_std_ratio"]) < .1 for r in rows])
    qualifying = []
    for summary in report["subset_summaries"]:
        selected = [r for r in rows if r["subset"] == summary["subset"]]
        median = float(np.median([float(r["dynamic_error_reduction"]) for r in selected]))
        check(median, summary["median_dynamic_error_reduction"])
        if median >= .1:
            qualifying.append(summary["subset"])
    assert report["gates"]["scale_suppression"] == ("MET" if fraction >= .75 else "NOT_MET")
    assert report["gates"]["dynamic_omission"] == ("MET" if len(qualifying) >= 2 else "NOT_MET")
    assert qualifying == report["gates"]["dynamic_qualifying_subsets"]
    h = hashlib.sha256()
    for name in ("__init__.py", "models.py", "prepare.py", "run.py", "PROTOCOL.json"):
        h.update(name.encode()); h.update((ROOT / "paper_b/ncmapss_response" / name).read_bytes())
    assert h.hexdigest() == "0d5ab920d31ec11767bb38f9ed717ce21f31ece3c6d981de7453cbd672d08009"
    acceptance = {"status": "PASS", "sealed_files_verified": len(seal["files_sha256"]),
        "diagnostic_fits_verified": folder_count, "engine_sensor_values_recomputed": len(rows),
        "input_scaler_train_four_recomputations": folder_count,
        "persisted_coefficient_predictions_recomputed": folder_count,
        "fit_or_optimizer_executed": False, "decoded_input_keys": ["raw", "unit", "cycle"],
        "official_TEST_or_RUL_labels_read": False, "frozen_response_execution_unchanged": h.hexdigest(),
        "pooled_fraction_residual_std_raw_std_below_0_1": float(fraction), "gates": report["gates"],
        "verifier_sha256": sha(__file__), "source_sha256": execution["diagnostic_source_sha256"],
        "protocol_sha256": execution["protocol_sha256"], "actual_run_exit_code": 0,
        "raw_mean_precision_corr_max_difference": raw_mean_precision_corr_max_difference,
        "evidence_sha256": {name: sha(HERE / name) for name in ("REGISTRATION.json", "SOFTWARE_CONTRACTS.json", "EXIT.json", "run.stdout.txt", "run.stderr.txt")}}
    with open(HERE / "ACCEPTANCE.json", "x", encoding="utf-8") as f:
        json.dump(acceptance, f, indent=2); f.write("\n")
    print(json.dumps(acceptance))


if __name__ == "__main__":
    main()
