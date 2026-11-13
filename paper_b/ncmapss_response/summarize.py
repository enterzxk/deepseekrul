"""Audit sealed DEVELOPMENT_LOEO results; generate a five-engine analytic age prior.

No model forward, HDF5 loader, TEST array, training, or author source is imported.
Only new output directories may be written. Missing evidence never passes a gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
VIEWS = ("clean", "density", "unpaired")
LABELS = ("raw", "clip125")
CONTROLS = ("raw", "fe", "delta", "supported", "age", "age_prior", "w_only")
DIAG_REQUIRED = (
    "rank_current", "rank_reference", "common_rank", "common_supported",
    "supported_fraction", "rank_below_latent_fraction", "support_current_count",
    "support_ref_count", "parallel_sum", "common_projector",
    "projection_removed_norm", "weighting_change_norm", "weighting_relative_change",
)
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20261113
REQUIRED_FILES = {
    "CLAIM.json", "config.json", "scalers.json", "initial.pt", "checkpoint.pt",
    "training_trace.json", "prediction_clean.csv", "prediction_density.csv",
    "prediction_unpaired.csv", "metrics.json",
}
PREDICTION_COLUMNS = (
    "unit", "cycle", "y_true_raw", "y_pred", "y_pred_unbounded", "seed", "view",
)


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def execution_hash():
    h = hashlib.sha256()
    for name in ("__init__.py", "models.py", "prepare.py", "run.py", "PROTOCOL.json"):
        h.update(name.encode())
        h.update((HERE / name).read_bytes())
    return h.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise AuditError(f"nonfinite JSON constant {value}: {path}")
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def new_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def score(y, prediction):
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    require(y.ndim == prediction.ndim == 1 and len(y) == len(prediction) > 0,
            "invalid scoring vectors")
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), "nonfinite scoring vectors")
    error = prediction - y
    with np.errstate(over="raise", invalid="raise"):
        nasa = np.expm1(np.where(error >= 0, error / 10., -error / 13.))
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "nasa_sum": float(nasa.sum()), "nasa_mean": float(nasa.mean()),
            "bias": float(error.mean()), "n": int(len(y))}


def assert_close(actual, expected, context, atol=1e-8, rtol=1e-9):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{context}: expected dictionary")
        for key, value in expected.items():
            require(key in actual, f"{context}: missing {key}")
            assert_close(actual[key], value, f"{context}.{key}", atol, rtol)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        a, b = np.asarray(actual), np.asarray(expected)
        require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False),
                f"{context}: array mismatch")
    elif isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, float(expected), abs_tol=atol, rel_tol=rtol),
                f"{context}: numeric mismatch ({actual!r} != {expected!r})")
    else:
        require(actual == expected, f"{context}: mismatch")


def load_development(cache, subset, protocol):
    manifest_path = cache / f"{subset}_dev_manifest.json"
    path = cache / f"{subset}_dev.npz"
    manifest = read_json(manifest_path)
    require(manifest.get("subset") == subset, "cache subset mismatch")
    require(manifest.get("official_test_decoded") is False, "cache TEST boundary not verified")
    require(manifest.get("decoded_keys") == ["A_dev", "W_dev", "X_s_dev", "Y_dev"],
            "cache decoded-key boundary mismatch")
    require(manifest.get("A_columns_decoded") == ["unit", "cycle"], "privileged A columns decoded")
    require(manifest.get("duration_is_model_input") is False, "original duration used as input")
    require(digest(path) == manifest.get("cache_sha256"), "cache SHA-256 mismatch")
    # Closed list: do not decode source_indices, original_points, HDF5 or TEST.
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("raw", "y", "unit", "cycle")}
    n = len(data["y"])
    require(data["raw"].shape == (n, protocol["points_per_flight"], 18), "raw shape mismatch")
    require(all(data[key].shape == (n,) for key in ("y", "unit", "cycle")), "metadata shape mismatch")
    require(all(np.isfinite(value).all() for value in data.values()), "nonfinite development data")
    require(np.all(data["y"] >= 0) and np.all(data["cycle"] >= 0), "negative labels/age")
    require(np.issubdtype(data["unit"].dtype, np.integer)
            and np.issubdtype(data["cycle"].dtype, np.integer), "noninteger unit/cycle")
    units = sorted(map(int, np.unique(data["unit"])))
    require(len(units) == 6 and units == manifest.get("units"), "six-engine identity mismatch")
    require(n == manifest.get("flights"), "cache flight-count mismatch")
    require(len(set(zip(data["unit"].tolist(), data["cycle"].tolist()))) == n, "duplicate flight")
    for unit in units:
        ix = np.flatnonzero(data["unit"] == unit)
        require(np.all(np.diff(data["cycle"][ix]) > 0), "nonmonotone engine flight order")
        require(int(len(ix)) == manifest["per_unit_flights"][str(unit)], "per-engine count mismatch")
    return data, manifest, {"npz_sha256": digest(path), "manifest_sha256": digest(manifest_path),
                            "units": units, "flights": n, "decoded_fields": ["raw", "y", "unit", "cycle"]}


def fit_scalers(data, units, cap):
    keep = np.isin(data["unit"], units)
    raw = data["raw"][keep].astype(np.float64)
    y = np.minimum(data["y"][keep].astype(np.float64), cap)
    return {"raw_mu": raw.mean(axis=(0, 1)).tolist(),
            "raw_sd": np.maximum(raw.std(axis=(0, 1)), 1e-6).tolist(),
            "age_q95": max(1., float(np.percentile(data["cycle"][keep], 95))),
            "target_mu": float(y.mean()), "target_sd": max(1., float(y.std())),
            "fitted_units": list(map(int, units))}


def make_age_prior(data, manifest, held_out, protocol, output, source_execution):
    units = sorted(int(unit) for unit in np.unique(data["unit"]) if unit != held_out)
    lifetimes = {}
    for unit in units:
        keep = data["unit"] == unit
        values = data["y"][keep].astype(np.float64) + data["cycle"][keep]
        require(np.ptp(values) <= 1e-5, f"age prior: nonconstant y_raw+cycle in training engine {unit}")
        lifetimes[str(unit)] = float(values.mean())
    mean_lifetime = float(np.mean(list(lifetimes.values())))
    keep = data["unit"] == held_out
    unbounded = mean_lifetime - data["cycle"][keep]
    prediction = np.maximum(unbounded, 0.)
    cell = f"{manifest['subset']}_age_prior_u{held_out}"
    folder = output / "age_prior" / cell
    folder.mkdir(parents=True, exist_ok=False)
    config = {"cell": cell, "subset": manifest["subset"], "mode": "age_prior",
              "held_out_unit": int(held_out), "fitted_units": units,
              "estimator": "equal-engine mean of training-only constant(y_raw+cycle)",
              "prediction": "max(0, mean_training_engine_lifetime-current_observed_cycle)",
              "seed": None, "training_performed": False, "fit_engine_count": 5,
              "source_sha256": manifest["source_sha256"], "cache_sha256": manifest["cache_sha256"],
              "summary_source_sha256": digest(Path(__file__)), "neural_execution_sha256": source_execution,
              "label_cap_applied_during_fit": False, "official_test_read": False}
    config["config_sha256"] = hashlib.sha256(canonical(config)).hexdigest()
    checkpoint = {"kind": "analytic_estimator_not_trained_network",
                  "training_engine_lifetimes": lifetimes, "mean_training_engine_lifetime": mean_lifetime,
                  "fitted_units": units, "held_out_unit": int(held_out),
                  "config_sha256": config["config_sha256"], "optimizer_updates": 0,
                  "training_performed": False}
    views = {view: prediction for view in VIEWS}
    scores = {view: {label: score(np.minimum(data["y"][keep], protocol["label_cap"])
                                      if label == "clip125" else data["y"][keep], prediction)
                     for label in LABELS} for view in VIEWS}
    record = {"cell": cell, "subset": manifest["subset"], "mode": "age_prior",
              "held_out_unit": int(held_out), "seed": None, "predictions": views,
              "y": data["y"][keep].astype(np.float64), "scores": scores,
              "training_seconds": 0., "parameters": 1, "optimizer_updates": 0,
              "drifts": {"density": 0., "unpaired": 0.}, "diagnostics": {},
              "checkpoint_state_verified": True, "replicate_kind": "deterministic_no_seed"}
    new_json(folder / "config.json", config)
    new_json(folder / "checkpoint.json", checkpoint)
    new_csv(folder / "prediction.csv",
            [{"unit": int(unit), "cycle": int(cycle), "y_true_raw": float(y),
              "y_pred": float(p), "y_pred_unbounded": float(z)}
             for unit, cycle, y, p, z in zip(data["unit"][keep], data["cycle"][keep],
                                            data["y"][keep], prediction, unbounded)],
            ["unit", "cycle", "y_true_raw", "y_pred", "y_pred_unbounded"])
    new_json(folder / "metrics.json", {"cell": cell, "scores": scores,
             "training_performed": False, "seed": None, "official_test_read": False,
             "prediction_view_reuse": "one analytic prediction CSV; density/unpaired cannot change observed age"})
    files = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}
    new_json(folder / "SUCCESS.json", {"cell": cell, "files_sha256": files})
    return record


def verify_seal(folder, cell):
    seal = read_json(folder / "SUCCESS.json")
    require(seal.get("cell") == cell, "SUCCESS cell mismatch")
    files = seal.get("files_sha256")
    require(isinstance(files, dict) and REQUIRED_FILES.issubset(files), "incomplete SUCCESS file inventory")
    require(set(files) == {p.name for p in folder.iterdir() if p.is_file() and p.name != "SUCCESS.json"},
            "sealed/unsealed file inventory mismatch")
    require(not any(p.is_dir() for p in folder.iterdir()), "unexpected nested artifact directory")
    for name, expected in files.items():
        require(Path(name).name == name and "/" not in name and "\\" not in name, "unsafe sealed path")
        target = folder / name
        require(not target.is_symlink() and target.resolve().parent == folder.resolve(), "artifact outside cell")
        require(digest(target) == expected, f"sealed hash mismatch: {name}")
    return len(files), digest(folder / "SUCCESS.json")


def read_prediction(path, data, held_out, seed, view):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == PREDICTION_COLUMNS, "prediction CSV schema mismatch")
        rows = list(reader)
    keep = data["unit"] == held_out
    require(len(rows) == int(keep.sum()), "prediction flight-count mismatch")
    require(all(int(row["unit"]) == held_out and int(row["seed"]) == seed
                and row["view"] == view for row in rows), "prediction identity mismatch")
    assert_close([int(row["cycle"]) for row in rows], data["cycle"][keep], "prediction cycle", atol=0, rtol=0)
    assert_close([float(row["y_true_raw"]) for row in rows], data["y"][keep], "prediction raw labels")
    p = np.asarray([float(row["y_pred"]) for row in rows], np.float64)
    unbounded = np.asarray([float(row["y_pred_unbounded"]) for row in rows], np.float64)
    require(np.isfinite(p).all() and np.isfinite(unbounded).all(), "nonfinite prediction")
    assert_close(p, np.maximum(unbounded, 0), "zero-only postprocessing")
    return p


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        require(hasattr(value, "detach"), "non-tensor state entry")
        a = value.detach().cpu().contiguous().numpy()
        require(np.isfinite(a).all(), f"nonfinite checkpoint tensor {key}")
        if np.issubdtype(a.dtype, np.floating):
            require(a.dtype == np.float32, f"non-FP32 checkpoint tensor {key}: {a.dtype}")
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(canonical(list(a.shape))); h.update(a.tobytes())
    return h.hexdigest()


def expected_state_shapes(mode, protocol):
    """Registered model structure, checked without constructing/importing it."""
    shapes = {}
    def linear(prefix, output, inputs):
        shapes[prefix + '.weight'] = (output, inputs)
        shapes[prefix + '.bias'] = (output,)
    if mode == 'age':
        linear('age_net.0', 64, 1); linear('age_net.2', 64, 64); linear('age_net.4', 1, 64)
        return shapes
    linear('encoder.raw_encoder.0', 64, 18); linear('encoder.raw_encoder.2', 64, 64)
    linear('encoder.response_body.0', 32, 4); linear('encoder.background', 14, 32)
    linear('encoder.basis', 56, 32); linear('encoder.aux_context_head', 4, 64)
    linear('encoder.embedding', 64, 73)
    d = protocol['hidden']
    linear('head.input', d, d + 1); linear('head.output', 1, d)
    shapes['head.pe'] = (protocol['span'], d)
    for layer in range(protocol['layers']):
        prefix = f'head.body.layers.{layer}'
        shapes[prefix + '.self_attn.in_proj_weight'] = (3*d, d)
        shapes[prefix + '.self_attn.in_proj_bias'] = (3*d,)
        linear(prefix + '.self_attn.out_proj', d, d)
        linear(prefix + '.linear1', 2*d, d); linear(prefix + '.linear2', d, 2*d)
        for norm in ('norm1', 'norm2'):
            shapes[prefix + '.' + norm + '.weight'] = (d,)
            shapes[prefix + '.' + norm + '.bias'] = (d,)
    return shapes


def verify_checkpoint(folder, config, scalers, result, protocol, expected_updates, torch_module):
    if torch_module is None:
        return False
    initial = torch_module.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    final = torch_module.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    for value in (initial, final):
        require(value["config"] == config and value["scalers"] == scalers, "checkpoint identity/scaler mismatch")
        require(value["initial_state_sha256"] == result["initial_state_sha256"], "checkpoint initial identity mismatch")
        expected_shapes = expected_state_shapes(config['mode'], protocol)
        require(set(value['model']) == set(expected_shapes), 'registered model state keys differ')
        require(all(tuple(value['model'][key].shape) == shape for key, shape in expected_shapes.items()),
                'registered model tensor shapes differ')
        parameters = sum(int(np.prod(shape)) for key, shape in expected_shapes.items() if key != 'head.pe')
        require(result['parameters'] == parameters, 'reported parameter count differs from complete registered model')
    require(state_hash(initial["model"]) == result["initial_state_sha256"], "initial tensor-state hash mismatch")
    require(state_hash(final["model"]) == result["final_state_sha256"]
            == final["final_state_sha256"], "final tensor-state hash mismatch")
    require(set(initial["model"]) == set(final["model"]), "initial/final state schemas differ")
    require(final["epochs_completed"] == protocol["epochs"], "checkpoint epoch budget mismatch")
    require(final["optimizer_updates"] == expected_updates, "checkpoint update count mismatch")
    optimizer_state = final["optimizer"]["state"]
    require(bool(optimizer_state), "empty optimizer state")
    steps = [float(value["step"]) for value in optimizer_state.values() if "step" in value]
    require(steps and all(step == expected_updates for step in steps), "optimizer step evidence mismatch")
    return True


def parameter_evidence(result, mode, protocol):
    total = sum(int(np.prod(shape)) for key, shape in expected_state_shapes(mode, protocol).items() if key != 'head.pe')
    require(result.get('parameters') == total, 'reported total parameter count inconsistent with registered model')
    require(result.get('auxiliary_disabled_for_reference') is (mode in ('age', 'w_only')),
            'reference auxiliary-loss policy mismatch')
    report = result.get('encoder_parameter_report')
    if mode == 'age':
        require(report is None, 'unexpected age encoder parameter report')
        return {'encoder_total_parameters': None, 'encoder_reported_effective_parameters': None,
                'encoder_audited_effective_parameters': None, 'audited_model_effective_parameters': total,
                'encoder_parameter_report_json': 'null', 'auxiliary_disabled_for_reference': True}
    require(isinstance(report, dict), 'missing encoder parameter report')
    inactive = 4138 if mode == 'w_only' else 512 if mode == 'raw' else 516 if mode == 'fe' else 260
    require(report.get('total_parameters') == 12842 and report.get('inactive_by_design') == inactive
            and report.get('structurally_effective_parameters') == 12842 - inactive,
            'encoder parameter report inconsistent with registered branches')
    # W-only has no sensor auxiliary loss; its controller report excludes the
    # auxiliary-only modules and the zero sensor input columns.
    audited_inactive = 4138 if mode == 'w_only' else inactive
    return {'encoder_total_parameters': 12842,
            'encoder_reported_effective_parameters': report['structurally_effective_parameters'],
            'encoder_audited_effective_parameters': 12842 - audited_inactive,
            'audited_model_effective_parameters': total - audited_inactive,
            'encoder_parameter_report_json': json.dumps(report, sort_keys=True),
            'auxiliary_disabled_for_reference': mode in ('age', 'w_only')}


def diagnostic_rows(value, prefix=""):
    if isinstance(value, dict) and {"shape", "mean", "minimum", "maximum"}.issubset(value):
        for key in ("mean", "minimum", "maximum"):
            require(isinstance(value[key], (int, float)) and math.isfinite(value[key]), "nonfinite diagnostic summary")
        require(value["minimum"] <= value["mean"] + 1e-6
                and value["mean"] <= value["maximum"] + 1e-6, "diagnostic extrema inconsistent")
        return [{"statistic": prefix, "shape": json.dumps(value["shape"]), "mean": value["mean"],
                 "minimum": value["minimum"], "maximum": value["maximum"]}]
    if isinstance(value, dict):
        return [row for key, item in value.items() for row in diagnostic_rows(item, f"{prefix}.{key}" if prefix else key)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), "nonfinite diagnostic scalar")
        return [{"statistic": prefix, "shape": "[]", "mean": value, "minimum": value, "maximum": value}]
    return []


def verify_cell(folder, subset, mode, unit, seed, data, manifest, expected_scalers,
                protocol, execution, torch_module):
    cell = f"{subset}_{mode}_u{unit}_s{seed}"
    verified_files, seal_sha = verify_seal(folder, cell)
    config, scalers, result = (read_json(folder / name) for name in ("config.json", "scalers.json", "metrics.json"))
    plain = {key: value for key, value in config.items() if key != "config_sha256"}
    require(hashlib.sha256(canonical(plain)).hexdigest() == config.get("config_sha256"), "config self-hash mismatch")
    train_units = sorted(int(u) for u in np.unique(data["unit"]) if int(u) != unit)
    expected = {"protocol": protocol, "cell": cell, "seed": seed, "mode": mode,
                "train_units": train_units, "held_out_unit": unit,
                "source_sha256": manifest["source_sha256"], "cache_sha256": manifest["cache_sha256"],
                "execution_sha256": execution}
    require(plain == expected, "frozen execution/config/cache identity mismatch")
    assert_close(scalers, expected_scalers, "five-engine scaler", atol=1e-10, rtol=1e-10)
    require(set(scalers) == set(expected_scalers), "unexpected scaler fields")
    for key, value in {"cell": cell, "subset": subset, "mode": mode, "held_out_unit": unit,
                       "seed": seed, "config_sha256": config["config_sha256"],
                       "execution_sha256": execution, "source_sha256": manifest["source_sha256"]}.items():
        require(result.get(key) == value, f"metrics identity mismatch: {key}")
    require(result.get("official_test_read") is False, "metrics TEST boundary mismatch")
    require(result.get("checkpoint_sha256") == digest(folder / "checkpoint.pt"), "checkpoint metric hash mismatch")
    require(result.get("initial_state_sha256") != result.get("final_state_sha256"), "no model state change")
    for key in ("initial_state_sha256", "final_state_sha256"):
        require(isinstance(result.get(key), str) and len(result[key]) == 64
                and all(c in "0123456789abcdef" for c in result[key]), "invalid tensor-state digest")
    ntrain = int(np.isin(data["unit"], train_units).sum())
    updates = protocol["epochs"] * math.ceil(ntrain / protocol["batch"])
    require(result.get("optimizer_updates") == updates, "metrics update count mismatch")
    require(isinstance(result.get("parameters"), int) and result["parameters"] > 0, "invalid parameter count")
    parameter_info = parameter_evidence(result, mode, protocol)
    require(0 < result["training_seconds"] <= 3600., "single-cell training time outside budget")
    require(0 < result["peak_allocated_MiB"] <= protocol["maximum_device_memory_MiB"], "device-memory budget mismatch")
    trace = read_json(folder / "training_trace.json")
    epochs = sorted({1, protocol["epochs"], *range(20, protocol["epochs"] + 1, 20)})
    require([row["epoch"] for row in trace] == epochs, "training epoch trace incomplete")
    for row in trace:
        require(all(math.isfinite(row[key]) and row[key] >= 0
                    for key in ("task", "auxiliary", "gradient_norm_last", "elapsed_seconds")), "invalid training trace")
    require(all(a["elapsed_seconds"] <= b["elapsed_seconds"] for a, b in zip(trace, trace[1:]))
            and trace[-1]["elapsed_seconds"] <= result["training_seconds"] + 1e-6,
            "training trace timing inconsistent")
    checkpoint_verified = verify_checkpoint(folder, config, scalers, result, protocol, updates, torch_module)
    keep = data["unit"] == unit
    predictions, scores, diagnostics = {}, {}, []
    missing_diagnostics = []
    for view in VIEWS:
        prediction = read_prediction(folder / f"prediction_{view}.csv", data, unit, seed, view)
        predictions[view] = prediction
        scores[view] = {label: score(np.minimum(data["y"][keep], protocol["label_cap"])
                                     if label == "clip125" else data["y"][keep], prediction) for label in LABELS}
        for label in LABELS:
            assert_close(result["scores"][view][label], scores[view][label], f"{view}.{label} score")
        d = result["scores"][view].get("diagnostics")
        require(isinstance(d, dict), "diagnostics object missing")
        if mode in ("common", "supported"):
            missing_diagnostics.extend(f"{view}.{key}" for key in DIAG_REQUIRED if key not in d)
            for key in ("rank_current", "rank_reference", "common_rank"):
                if key in d:
                    require(d[key].get("shape") == [int(keep.sum())]
                            and 0 <= d[key]["minimum"] <= d[key]["maximum"] <= 4, "rank diagnostic mismatch")
        for row in diagnostic_rows(d):
            diagnostics.append({"cell": cell, "subset": subset, "mode": mode, "held_out_unit": unit,
                                "seed": seed, "view": view, **row})
    drifts = {view: float(np.mean(np.abs(predictions[view] - predictions["clean"]))) for view in ("density", "unpaired")}
    for view, value in drifts.items():
        assert_close(result["scores"][view].get("mean_absolute_prediction_drift"), value, f"{view}.drift")
    return {"cell": cell, "subset": subset, "mode": mode, "held_out_unit": unit, "seed": seed,
            "y": data["y"][keep].astype(np.float64), "predictions": predictions, "scores": scores,
            "drifts": drifts, "diagnostics": diagnostics, "checkpoint_state_verified": checkpoint_verified,
            "missing_diagnostics": missing_diagnostics, "training_seconds": result["training_seconds"],
            "parameters": result["parameters"], "optimizer_updates": updates,
            "peak_allocated_MiB": result["peak_allocated_MiB"], "replicate_kind": "neural_seed",
            "seal_sha256": seal_sha, "verified_artifact_files": verified_files, **parameter_info}


def summarize_modes(records, age_records, protocol):
    index = {(r["subset"], r["mode"], r["held_out_unit"], r["seed"]): r for r in records + age_records}
    seed_rows, summaries = [], []
    for subset in protocol["subsets"]:
        for mode in (*protocol["modes"], "age_prior"):
            seeds = [None] if mode == "age_prior" else protocol["seeds"]
            for view in VIEWS:
                for label in LABELS:
                    group = []
                    for seed in seeds:
                        rows = [r for key, r in index.items()
                                if key[0] == subset and key[1] == mode and key[3] == seed]
                        complete = len(rows) == 6
                        pooled = score(np.concatenate([np.minimum(r["y"], protocol["label_cap"])
                                                       if label == "clip125" else r["y"] for r in rows]),
                                       np.concatenate([r["predictions"][view] for r in rows])) if complete else None
                        row = {"subset": subset, "mode": mode, "view": view, "label": label, "seed": seed,
                               "status": "COMPLETE" if complete else "PARTIAL", "heldout_engines": len(rows),
                               "engine_mean_rmse": float(np.mean([r["scores"][view][label]["rmse"] for r in rows])) if complete else None,
                               "flight_pooled_rmse": pooled["rmse"] if pooled else None,
                               "nasa_sum": pooled["nasa_sum"] if pooled else None,
                               "nasa_mean": pooled["nasa_mean"] if pooled else None,
                               "flights": pooled["n"] if pooled else None}
                        seed_rows.append(row); group.append(row)
                    complete = all(r["status"] == "COMPLETE" for r in group)
                    summary = {"subset": subset, "mode": mode, "view": view, "label": label,
                               "status": "COMPLETE" if complete else "PARTIAL",
                               "seed_count": 0 if mode == "age_prior" else sum(r["status"] == "COMPLETE" for r in group),
                               "replicate_kind": "deterministic_no_seed" if mode == "age_prior" else "six_engine_mean_then_three_seeds",
                               "expected_engine_seed_cells": 6 * len(seeds),
                               "available_engine_seed_cells": sum(r["heldout_engines"] for r in group)}
                    for name in ("engine_mean_rmse", "flight_pooled_rmse", "nasa_sum", "nasa_mean"):
                        values = [r[name] for r in group] if complete else []
                        summary[name] = float(np.mean(values)) if values else None
                        summary[name + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                    summaries.append(summary)
    return index, seed_rows, summaries


def paired_results(index, protocol):
    rows, engine_rows, summaries = [], [], []
    for subset in protocol["subsets"]:
        for control in CONTROLS:
            for view in VIEWS:
                for label in LABELS:
                    groups = defaultdict(list)
                    for key, common in index.items():
                        if key[0] != subset or key[1] != "common":
                            continue
                        unit, seed = key[2:]
                        reference = index.get((subset, control, unit, None if control == "age_prior" else seed))
                        if reference is None:
                            continue
                        a, b = common["scores"][view][label]["rmse"], reference["scores"][view][label]["rmse"]
                        row = {"subset": subset, "control": control, "view": view, "label": label,
                               "held_out_unit": unit, "seed": seed, "common_rmse": a, "control_rmse": b,
                               "delta_rmse": a - b, "relative_gain_percent": 100 * (b - a) / b if b > 0 else None,
                               "age_prior_shared_across_seeds": control == "age_prior"}
                        rows.append(row); groups[unit].append(row)
                    complete_engines = []
                    for unit, group in sorted(groups.items()):
                        complete = {r["seed"] for r in group} == set(protocol["seeds"])
                        row = {"subset": subset, "control": control, "view": view, "label": label,
                               "held_out_unit": unit, "paired_seeds": len(group),
                               "status": "COMPLETE" if complete else "PARTIAL",
                               "common_rmse_mean_seed": float(np.mean([r["common_rmse"] for r in group])) if complete else None,
                               "control_rmse_mean_seed": float(np.mean([r["control_rmse"] for r in group])) if complete else None,
                               "delta_rmse_mean_seed": float(np.mean([r["delta_rmse"] for r in group])) if complete else None}
                        engine_rows.append(row)
                        if complete:
                            complete_engines.append(row)
                    output = {"subset": subset, "control": control, "view": view, "label": label,
                              "status": "COMPLETE" if len(complete_engines) == 6 else "PARTIAL",
                              "paired_engines": len(complete_engines), "paired_engine_seed_cells": sum(map(len, groups.values())),
                              "delta_rmse": None, "relative_gain_percent": None,
                              "delta_ci95_low": None, "delta_ci95_high": None,
                              "gain_ci95_low": None, "gain_ci95_high": None}
                    if len(complete_engines) == 6:
                        a = np.asarray([r["common_rmse_mean_seed"] for r in complete_engines])
                        b = np.asarray([r["control_rmse_mean_seed"] for r in complete_engines])
                        output["delta_rmse"] = float((a - b).mean())
                        output["relative_gain_percent"] = float(100 * (b.mean() - a.mean()) / b.mean()) if b.mean() > 0 else None
                        tag = f"{subset}/{control}/{view}/{label}"
                        salt = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:4], "little")
                        rng = np.random.default_rng(BOOTSTRAP_SEED + salt)
                        sample = rng.integers(0, 6, (BOOTSTRAP_REPETITIONS, 6))
                        boot_delta = (a[sample] - b[sample]).mean(1)
                        low, high = np.percentile(boot_delta, [2.5, 97.5])
                        output.update(delta_ci95_low=float(low), delta_ci95_high=float(high))
                        means_b = b[sample].mean(1)
                        if np.all(means_b > 0):
                            boot_gain = 100 * (means_b - a[sample].mean(1)) / means_b
                            low, high = np.percentile(boot_gain, [2.5, 97.5])
                            output.update(gain_ci95_low=float(low), gain_ci95_high=float(high))
                    summaries.append(output)
    return rows, engine_rows, summaries


def drift_results(records, protocol):
    output = []
    for subset in protocol["subsets"]:
        for mode in protocol["modes"]:
            for view in ("density", "unpaired"):
                rows = [r for r in records if r["subset"] == subset and r["mode"] == mode]
                complete = len(rows) == 6 * len(protocol["seeds"])
                values = [float(np.mean([r["drifts"][view] for r in rows if r["seed"] == seed]))
                          for seed in protocol["seeds"]] if complete else []
                output.append({"subset": subset, "mode": mode, "view": view,
                               "status": "COMPLETE" if complete else "PARTIAL", "engine_seed_cells": len(rows),
                               "engine_mean_drift": float(np.mean(values)) if values else None,
                               "between_seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return output


def diagnostic_summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["subset"], row["mode"], row["view"], row["statistic"])].append(row)
    return [{"subset": key[0], "mode": key[1], "view": key[2], "statistic": key[3],
             "engine_seed_cells": len(group), "mean_of_cell_means": float(np.mean([r["mean"] for r in group])),
             "minimum_across_cells": min(r["minimum"] for r in group),
             "maximum_across_cells": max(r["maximum"] for r in group),
             "aggregation": "equal-cell tensor summaries only; no inferred rank histogram or covariance"}
            for key, group in sorted(groups.items())]


def evaluate_gates(paired, drifts, diagnostics, protocol, evidence_complete):
    pairs = {(r["subset"], r["control"], r["view"]): r for r in paired if r["label"] == "raw"}
    drift = {(r["subset"], r["mode"], r["view"]): r for r in drifts}
    diag = {(r["subset"], r["mode"], r["view"], r["statistic"]): r for r in diagnostics}
    def gain(subset, control, view="clean"):
        return pairs.get((subset, control, view), {}).get("relative_gain_percent")
    def comparison(controls):
        subset_rows = []
        for subset in protocol["subsets"]:
            gains = {control: gain(subset, control) for control in controls}
            identifiable = all(value is not None for value in gains.values())
            subset_rows.append({"subset": subset, "gains_percent": gains,
                                "joint_two_percent_gain": identifiable and all(v >= 2. for v in gains.values()),
                                "no_more_than_two_percent_failure": identifiable and all(v >= -2. for v in gains.values())})
        ready = all(all(v is not None for v in row["gains_percent"].values()) for row in subset_rows)
        passed = ready and sum(row["joint_two_percent_gain"] for row in subset_rows) >= 2 and all(row["no_more_than_two_percent_failure"] for row in subset_rows)
        return {"status": "PASS" if passed else ("FAIL" if ready else "NOT_EVALUABLE"),
                "subsets": subset_rows}
    primary, attribution = comparison(("raw", "fe")), comparison(("delta", "supported"))
    age_rows = [{"subset": subset, "gains_percent": {c: gain(subset, c) for c in ("age", "age_prior")}}
                for subset in protocol["subsets"]]
    age_ready = all(all(v is not None for v in r["gains_percent"].values()) for r in age_rows)
    age_pass = age_ready and all(all(v > 0 for v in r["gains_percent"].values()) for r in age_rows)
    age = {"status": "PASS" if age_pass else ("FAIL" if age_ready else "NOT_EVALUABLE"), "subsets": age_rows,
           "rule": "common strictly better than BOTH age MLP and analytic prior on EACH subset; no significance claim"}
    if primary["status"] == "PASS" and age["status"] != "PASS":
        primary["status"] = age["status"]
    sensor_rows = [{"subset": subset, "gain_percent": gain(subset, 'w_only')}
                   for subset in protocol['subsets']]
    sensor_ready = all(row['gain_percent'] is not None for row in sensor_rows)
    sensor_pass = sensor_ready and all(row['gain_percent'] > 0 for row in sensor_rows)
    sensor = {'status': 'PASS' if sensor_pass else ('FAIL' if sensor_ready else 'NOT_EVALUABLE'),
              'subsets': sensor_rows, 'rule': 'common strictly better than W-only on EACH subset; necessary sensor-reference check, not isolated causal attribution'}
    mechanism_rows = []
    for subset in protocol["subsets"]:
        comparisons = []
        for control in ("fe", "delta", "supported"):
            dc = drift.get((subset, control, "density"), {}).get("engine_mean_drift")
            dn = drift.get((subset, "common", "density"), {}).get("engine_mean_drift")
            clean, unpaired, density_gain = gain(subset, control), gain(subset, control, "unpaired"), gain(subset, control, "density")
            ready = all(v is not None for v in (dc, dn, clean, unpaired, density_gain))
            identifiable = ready and dc >= .05
            reduction = 100 * (dc - dn) / dc if identifiable else None
            passed = (identifiable and reduction >= 25. and clean > 0
                      and unpaired <= .5 * clean and density_gain >= -2.)
            comparisons.append({"control": control, "control_density_drift": dc, "common_density_drift": dn,
                                "drift_reduction_percent": reduction, "clean_relative_gain_percent": clean,
                                "unpaired_relative_gain_percent": unpaired, "density_relative_gain_percent": density_gain,
                                "status": "PASS" if passed else ("NOT_EVALUABLE" if not ready else
                                          "NOT_IDENTIFIABLE" if not identifiable else "FAIL")})
        support = diag.get((subset, "common", "clean", "supported_fraction"))
        effect = diag.get((subset, "common", "clean", "weighting_change_norm"))
        diagnostic_present = support is not None and effect is not None and support["engine_seed_cells"] == effect["engine_seed_cells"] == 18
        observed_effect = diagnostic_present and support["maximum_across_cells"] > 0 and effect["maximum_across_cells"] > 0
        mechanism_rows.append({"subset": subset, "controls": comparisons,
                               "rank_weight_diagnostics_available": diagnostic_present,
                               "any_observed_supported_weighting_effect": bool(observed_effect),
                               "joint_pass": all(r["status"] == "PASS" for r in comparisons) and bool(observed_effect)})
    mechanism_ready = all(all(c["status"] != "NOT_EVALUABLE" for c in r["controls"])
                          and r["rank_weight_diagnostics_available"] for r in mechanism_rows)
    mechanism_pass = sum(r["joint_pass"] for r in mechanism_rows) >= 2
    identifiable_subsets = sum(all(c["status"] not in ("NOT_EVALUABLE", "NOT_IDENTIFIABLE")
                                   for c in r["controls"]) for r in mechanism_rows)
    mechanism = {"status": "PASS" if mechanism_pass and mechanism_ready else
                 ("NOT_EVALUABLE" if not mechanism_ready else
                  "NOT_IDENTIFIABLE" if identifiable_subsets < 2 else "FAIL"), "subsets": mechanism_rows,
                 "zero_effect_rule": "absent/zero support or weighting effect cannot identify a weighting mechanism"}
    numerical_pass = all(g["status"] == "PASS" for g in (primary, attribution, age, sensor, mechanism))
    return {"overall_status": "PASS" if evidence_complete and numerical_pass else
            ("FAILED_HYPOTHESIS" if evidence_complete else "PARTIAL"),
            "evidence_complete": evidence_complete, "primary": primary, "age_references": age,
            "sensor_specificity_gate": sensor,
            "attribution": attribution, "mechanism": mechanism,
            "thresholds": {"relative_gain_percent": 2., "max_subset_worsening_percent": 2.,
                           "minimum_same_subsets": 2, "minimum_control_drift_cycles": .05,
                           "density_drift_reduction_percent": 25., "unpaired_max_fraction_of_clean_gain": .5,
                           "density_accuracy_rule": "common no more than 2% worse than each perturbed control"},
            "interpretation": "all controls must pass on the SAME >=2 subsets; other subsets cannot exceed failure bound; no TEST or significance inference"}


def markdown_report(summary, mode_rows, pairs, diagnostics):
    lines = ["# DEVELOPMENT_LOEO", "", f"Evidence status: **{summary['evidence_status']}**. Research gate: **{summary['gates']['overall_status']}**.",
             "", f"Verified neural cells: {summary['verified_neural_cells']}/{summary['expected_neural_cells']}; "
             f"analytic age-prior folds: {summary['analytic_age_prior_folds']}/24; invalid cells: {len(summary['invalid_cells'])}.",
             "", "All inputs and predictions are development-only. No official TEST was opened by this collector.",
             "", "Primary: within each seed, average the six held-out-engine RMSEs; then report the three-seed mean and sample SD. "
             "Flight-pooled RMSE and NASA sum/mean are separate. Analytic age prior is deterministic, with no artificial seed replicates.",
             "", "|Subset|Mode|Raw engine-mean RMSE|Between-seed SD|Evidence|",
             "|---|---|---:|---:|---|"]
    def fmt(v): return "missing" if v is None else f"{v:.6g}"
    for row in mode_rows:
        if row["view"] == "clean" and row["label"] == "raw":
            lines.append(f"|{row['subset']}|{row['mode']}|{fmt(row['engine_mean_rmse'])}|{fmt(row['engine_mean_rmse_seed_sd'])}|{row['status']}|")
    lines += ["", "|Subset|Common versus|Paired RMSE delta (common-control)|Engine-bootstrap 95% interval|Gain %|",
              "|---|---|---:|---|---:|"]
    for row in pairs:
        if row["view"] == "clean" and row["label"] == "raw":
            interval = f"[{fmt(row['delta_ci95_low'])}, {fmt(row['delta_ci95_high'])}]"
            lines.append(f"|{row['subset']}|{row['control']}|{fmt(row['delta_rmse'])}|{interval}|{fmt(row['relative_gain_percent'])}|")
    lines += ["", "Bootstrap first averages paired differences over seeds within each engine, then resamples the six engines "
              f"({BOOTSTRAP_REPETITIONS} draws, percentile 95% intervals). The six LOEO training sets overlap; "
              "these are descriptive clustered intervals, not independent-fold hypothesis tests. Seeds are not engines. No p-values or significance claims.",
              "", "## Frozen gates", "", "|Gate|Status|", "|---|---|"]
    for name in ("primary", "age_references", "sensor_specificity_gate", "attribution", "mechanism"):
        lines.append(f"|{name}|{summary['gates'][name]['status']}|")
    lines += ["", "Primary requires common jointly >=2% better than raw and FE on the same >=2 subsets; attribution similarly "
              "requires delta and supported. No control/subset may be worse by >2%. Common must strictly beat both age references on every subset. "
              "Common must also strictly beat W-only on every subset. W-only retains W/coverage/age and the same temporal head, zeroes Xs and disables its auxiliary loss; it is a reference, not a matched H-attribution arm. "
              "Mechanism requires the same >=2 subsets jointly passing all FE/delta/supported controls: >=25% density-drift reduction, "
              "control drift >=0.05 cycles, positive clean gain, unpaired relative gain <=half clean gain, and perturbed common RMSE no >2% worse. "
              "Missing evidence blocks a full PASS. Near-zero control drift is not identifiable.",
              "", "## Observed rank and weighting", "",
              "|Subset|Statistic (common, clean)|Mean of cell means|Minimum|Maximum|Cells|",
              "|---|---|---:|---:|---:|---:|"]
    for row in diagnostics:
        if row["mode"] == "common" and row["view"] == "clean" and row["statistic"] in (
            "common_rank", "rank_below_latent_fraction", "supported_fraction",
            "projection_removed_norm", "weighting_change_norm", "weighting_relative_change"):
            lines.append(f"|{row['subset']}|{row['statistic']}|{fmt(row['mean_of_cell_means'])}|"
                         f"{fmt(row['minimum_across_cells'])}|{fmt(row['maximum_across_cells'])}|{row['engine_seed_cells']}|")
    if not diagnostics:
        lines.append("|—|missing; no observations available|—|—|—|0|")
    lines += ["", "Only the controller's actual per-tensor mean/minimum/maximum summaries are available. "
              "No per-flight rank histogram, conditional weighting distribution, uncertainty calibration or physical-health interpretation is inferred. "
              "All available diagnostics are preserved in diagnostics_cells.csv and diagnostics_summary.csv.",
              "", "## Resource and evidence limits", "",
              f"Recorded training time: {summary['recorded_training_hours']:.6g} process-hours, excluding evaluation and unrecorded preparation/overhead. "
              f"Registered maximum: {summary['protocol']['maximum_gpu_training_hours']} hours. "
              "Per-cell parameters, optimizer updates, memory and seals are in cells.csv.",
              "", "Total parameters include inactive branches. cells.csv retains the controller's encoder report and a separate task-only W-only count: "
              "Xs-zero input plus disabled auxiliary loss leaves 8704 effective encoder parameters (79937 including head), versus total 12842/84075. "
              "These structural counts do not remove sample-dependent support fallbacks or saturation.",
              "", "Four external-paper baselines are source-audited separately; their uniform-protocol training is not established by this response-module report.",
              "", "Invalid/missing caches, cells, unavailable CPU checkpoint validation and missing diagnostics are listed in summary.json. "
              "No partial matrix is promoted to a complete comparison or a validated novel mechanism."]
    return "\n".join(lines) + "\n"


def collect(cache, runs, output):
    cache, runs, output = map(lambda p: Path(p).resolve(), (cache, runs, output))
    require(output != cache and output != runs, "output must differ from input directories")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    protocol = read_json(HERE / "PROTOCOL.json")
    require(set(protocol["modes"]) == {"raw", "fe", "delta", "supported", "common", "age", "w_only"}
            and len(protocol["subsets"]) == 4 and len(protocol["seeds"]) == 3
            and protocol["training_runs"] == 6 * len(protocol["modes"]) * len(protocol["seeds"]) * len(protocol["subsets"]),
            "unsupported registration matrix")
    source_execution = execution_hash()
    try:
        import torch
        checkpoint_dependency = {"available": True, "version": torch.__version__, "device": "CPU map_location only"}
    except ImportError as error:
        torch = None
        checkpoint_dependency = {"available": False, "reason": str(error)}
    output.mkdir(parents=True, exist_ok=False)
    new_json(output / "COLLECTION_CLAIM.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
             "execution_sha256": source_execution, "summary_source_sha256": digest(Path(__file__)),
             "cache": str(cache), "runs": str(runs), "official_test_read": False})
    records, ages, missing, invalid, cache_audit, cache_errors = [], [], [], [], {}, []
    expected_names = set()
    for subset in protocol["subsets"]:
        try:
            data, manifest, cache_info = load_development(cache, subset, protocol)
            cache_audit[subset] = cache_info
        except Exception as error:
            cache_errors.append({"subset": subset, "type": type(error).__name__, "reason": str(error),
                                 "status": "MISSING" if isinstance(error, FileNotFoundError) else "INVALID"})
            continue
        for unit in cache_info["units"]:
            train_units = [u for u in cache_info["units"] if u != unit]
            expected_scalers = fit_scalers(data, train_units, protocol["label_cap"])
            try:
                ages.append(make_age_prior(data, manifest, unit, protocol, output, source_execution))
            except (ValueError, KeyError, AssertionError) as error:
                invalid.append({"cell": f"{subset}_age_prior_u{unit}", "reason": str(error)})
            for mode in protocol["modes"]:
                for seed in protocol["seeds"]:
                    cell = f"{subset}_{mode}_u{unit}_s{seed}"
                    expected_names.add(cell)
                    folder = runs / cell
                    if not (folder / "SUCCESS.json").is_file():
                        missing.append({"cell": cell, "status": "INCOMPLETE" if folder.exists() else "MISSING"})
                        continue
                    try:
                        records.append(verify_cell(folder, subset, mode, unit, seed, data, manifest,
                                                   expected_scalers, protocol, source_execution, torch))
                    except Exception as error:
                        invalid.append({"cell": cell, "type": type(error).__name__, "reason": str(error)})
    unexpected = [p.parent.name for p in runs.glob("*/SUCCESS.json")
                  if p.parent.name not in expected_names] if len(cache_audit) == 4 else []
    missing_diag = [{"cell": r["cell"], "fields": r["missing_diagnostics"]} for r in records if r["missing_diagnostics"]]
    cpu_unverified = [r["cell"] for r in records if not r["checkpoint_state_verified"]]
    index, seeds, modes = summarize_modes(records, ages, protocol)
    paired_cells, paired_engines, paired = paired_results(index, protocol)
    drifts = drift_results(records, protocol)
    diag_rows = [d for r in records for d in r["diagnostics"]]
    diag_summary = diagnostic_summaries(diag_rows)
    recorded_hours = sum(r["training_seconds"] for r in records) / 3600.
    # Concurrent workers share one physical GPU. Summing their elapsed times
    # double-counts contention, so verify the supervisor's elapsed wall budget.
    resource_hours = recorded_hours
    budget_basis = "sum_of_cell_times_serial_fallback"
    for name in ("ALL_TRAINING_COMPLETE.json", "STATUS.json"):
        status_path = runs.parent / name
        if status_path.is_file():
            status = read_json(status_path)
            require(status.get("execution_sha256") == source_execution, "supervisor budget identity mismatch")
            require(math.isfinite(status["elapsed_seconds"]) and status["elapsed_seconds"] >= 0,
                    "invalid supervisor elapsed time")
            resource_hours = status["elapsed_seconds"] / 3600.
            budget_basis = name
            break
    budget_pass = resource_hours <= protocol["maximum_gpu_training_hours"]
    execution_unchanged = execution_hash() == source_execution
    complete = (len(records) == protocol["training_runs"] and len(ages) == 24 and not invalid
                and not cache_errors and not missing and not unexpected and not missing_diag
                and not cpu_unverified and budget_pass and execution_unchanged)
    gates = evaluate_gates(paired, drifts, diag_summary, protocol, complete)
    evidence_status = "INVALID" if (invalid or unexpected or not budget_pass or not execution_unchanged
                                   or any(e["status"] == "INVALID" for e in cache_errors)) else ("COMPLETE" if complete else "PARTIAL")
    if evidence_status == "INVALID":
        gates["overall_status"] = "INVALID_EVIDENCE"
    summary = {"title": "DEVELOPMENT_LOEO", "evidence_status": evidence_status, "protocol": protocol,
               "execution_sha256": source_execution, "protocol_sha256": digest(HERE / "PROTOCOL.json"),
               "summary_source_sha256": digest(Path(__file__)), "official_test_read": False,
               "execution_unchanged_during_collection": execution_unchanged,
               "decoded_cache_fields": ["raw", "y", "unit", "cycle"], "cache_audit": cache_audit,
               "cache_errors": cache_errors, "expected_neural_cells": protocol["training_runs"],
               "verified_neural_cells": len(records), "analytic_age_prior_folds": len(ages),
               "shared_gpu_wall_hours": resource_hours, "budget_basis": budget_basis,
               "missing_cells": missing, "invalid_cells": invalid, "unexpected_sealed_cells": unexpected,
               "missing_diagnostics": missing_diag, "cpu_checkpoint_dependency": checkpoint_dependency,
               "cpu_checkpoint_unverified_cells": cpu_unverified,
               "verified_artifact_file_hashes": sum(r["verified_artifact_files"] for r in records),
               "recorded_training_hours": recorded_hours, "recorded_training_budget_pass": budget_pass,
               "budget_scope": "one shared GPU supervisor wall time; recorded_training_hours sums successful process times and can double-count concurrent contention",
               "bootstrap": {"replicates": BOOTSTRAP_REPETITIONS, "base_seed": BOOTSTRAP_SEED,
                             "cluster": "held-out engine, after averaging seeds", "interval": "descriptive percentile95",
                             "limitations": "six engines; overlapping LOEO training sets; no independent-fold p-values"},
               "gates": gates, "mode_summary": modes, "paired_summary": paired,
               "drift_summary": drifts, "diagnostic_summary": diag_summary,
               "external_four_paper_baselines": "PARTIAL: source preservation only; no uniform-training completion certified"}
    cell_fields = ["cell", "subset", "mode", "held_out_unit", "seed", "parameters", "training_seconds",
                   "optimizer_updates", "peak_allocated_MiB", "checkpoint_state_verified", "seal_sha256", "verified_artifact_files",
                   "encoder_total_parameters", "encoder_reported_effective_parameters", "encoder_audited_effective_parameters",
                   "audited_model_effective_parameters", "encoder_parameter_report_json", "auxiliary_disabled_for_reference"]
    cells = [{k: r.get(k) for k in cell_fields} for r in records]
    tables = [
        ("cells.csv", cells, cell_fields),
        ("seed_metrics.csv", seeds, ["subset", "mode", "view", "label", "seed", "status", "heldout_engines",
                                    "engine_mean_rmse", "flight_pooled_rmse", "nasa_sum", "nasa_mean", "flights"]),
        ("mode_summary.csv", modes, list(modes[0])),
        ("paired_engine_seed.csv", paired_cells, ["subset", "control", "view", "label", "held_out_unit", "seed",
               "common_rmse", "control_rmse", "delta_rmse", "relative_gain_percent", "age_prior_shared_across_seeds"]),
        ("paired_engines.csv", paired_engines, ["subset", "control", "view", "label", "held_out_unit", "paired_seeds",
               "status", "common_rmse_mean_seed", "control_rmse_mean_seed", "delta_rmse_mean_seed"]),
        ("paired_summary.csv", paired, list(paired[0])),
        ("drift_summary.csv", drifts, list(drifts[0])),
        ("diagnostics_cells.csv", diag_rows, ["cell", "subset", "mode", "held_out_unit", "seed", "view", "statistic",
               "shape", "mean", "minimum", "maximum"]),
        ("diagnostics_summary.csv", diag_summary, ["subset", "mode", "view", "statistic", "engine_seed_cells",
               "mean_of_cell_means", "minimum_across_cells", "maximum_across_cells", "aggregation"]),
    ]
    for name, rows, fields in tables:
        new_csv(output / name, rows, fields)
    new_json(output / "summary.json", summary)
    new_json(output / "gates.json", gates)
    with (output / "DEVELOPMENT_LOEO.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_report(summary, modes, paired, diag_summary))
    artifacts = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    new_json(output / "COLLECTION_COMPLETE.json", {"title": "DEVELOPMENT_LOEO",
             "evidence_status": evidence_status, "research_gate_status": gates["overall_status"],
             "files_sha256": artifacts, "official_test_read": False})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.cache, args.runs, args.output)
    print(json.dumps({k: summary[k] for k in ("title", "evidence_status", "verified_neural_cells",
                                            "expected_neural_cells", "analytic_age_prior_folds")} |
                     {"research_gate": summary["gates"]["overall_status"]}))
    return 2 if summary["evidence_status"] == "INVALID" else 3 if summary["evidence_status"] == "PARTIAL" else 0


if __name__ == "__main__":
    sys.exit(main())

