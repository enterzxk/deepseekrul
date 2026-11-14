"""Independent DEVELOPMENT_LOEO collector for the frozen full-BayesRUL-HNN migration.

Read only development NPZs and already sealed artifacts. No training or forward.
Run as a standalone file; --runtime is the unmodified frozen source root.
PARTIAL means only the visible sealed snapshot was validated, not full success.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
COMMIT = "1cdfc7fb130e909877cab93067fd53b8cfc30318"
EXECUTION = "3b76be258fc6829f4c54208d29c1a1131d44014119d754a78e8f971d752f53af"
RUNTIME_MANIFEST = "da1ca56e3792b1694aafceafad69cf20438bd58c2dc58488cd283552def44aef"
AUTHOR_COMMIT = "2943f4d85a1bb71910c100c956b32c3f6b595ceb"
AUTHOR_MODEL_SHA = "83ea189b83264ef53a004d4aea2848f011a8f316bc7b0bc693aca45fcee052f6"
AUTHOR_MANIFEST_SHA = "fd224fb473d933d41718aede7fa6b4b70ed569ab99423a5d298bc2c3f0b32a6a"
VIEWS = ("clean", "density", "unpaired")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    # The frozen writer uses exclusive-create rather than rename; tolerate only
    # its brief in-progress JSON write, never a persistently malformed artifact.
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.05)


def metrics(y, prediction):
    import numpy as np
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    require(y.ndim == 1 and len(y) > 0 and y.shape == prediction.shape, "invalid metric vectors")
    require(bool(np.isfinite(y).all() and np.isfinite(prediction).all()), "nonfinite metric inputs")
    e = prediction - y
    score = np.expm1(np.where(e >= 0, e / 10., -e / 13.))
    values = {"rmse": float(np.sqrt(np.mean(e * e))), "mae": float(np.mean(abs(e))),
              "nasa_sum": float(score.sum()), "nasa_mean": float(score.mean()),
              "bias": float(e.mean()), "n": len(y)}
    require(all(math.isfinite(v) for v in values.values()), "nonfinite derived metrics")
    return values


def close(actual, expected, label, atol=1e-10, rtol=1e-10):
    import numpy as np
    require(np.shape(actual) == np.shape(expected) and
            bool(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)), label)


def load_runtime(runtime, author_file):
    """Check explicit frozen Git bytes and original recipe before any import."""
    runtime, author_file = Path(runtime).resolve(), Path(author_file).resolve()
    manifest_path = runtime/"GIT_SOURCE_MANIFEST.json"
    require(digest(manifest_path) == RUNTIME_MANIFEST, "unrecognized frozen runtime manifest")
    manifest = read_json(manifest_path)
    require(manifest["commit"] == COMMIT, "wrong frozen commit")
    verified = {}
    for entry in manifest["files"]:
        path = (runtime/entry["path"]).resolve()
        require(path.is_relative_to(runtime) and path.is_file(), "unsafe runtime file")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == entry["sha256"] and len(raw) == entry["bytes"], "changed frozen runtime bytes")
        require(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() == entry["git_blob_sha1"], "Git blob mismatch")
        verified[entry["path"]] = entry["sha256"]
    require(digest(author_file) == AUTHOR_MODEL_SHA, "changed author model")
    author_root = author_file.parents[3]
    author_manifest_path = author_root/"SOURCE_MANIFEST.json"
    require(digest(author_manifest_path) == AUTHOR_MANIFEST_SHA, "changed author source manifest")
    author_manifest = read_json(author_manifest_path)
    require(author_manifest["full_commit_sha"] == AUTHOR_COMMIT, "wrong author commit")
    recipe = read_json(runtime/"paper_b/ncmapss_bayes_20261114/NATIVE_RECIPE.json")
    entries = {r["path"]: r for r in author_manifest["included"]}
    author_hashes = {}
    for rel, expected in recipe["verified_source_sha256"].items():
        path = (author_root/rel).resolve()
        require(path.is_relative_to(author_root), "unsafe author path")
        require(digest(path) == expected == entries[rel]["sha256"], "author core/loss/init/recipe changed: "+rel)
        author_hashes[rel] = expected
    for key in tuple(sys.modules):
        if key == "paper_b" or key.startswith("paper_b."):
            del sys.modules[key]
    sys.path.insert(0, str(runtime))
    runner = importlib.import_module("paper_b.ncmapss_bayes_20261114.run_bayes")
    for key, module in tuple(sys.modules.items()):
        if key == "paper_b" or key.startswith("paper_b."):
            file = getattr(module, "__file__", None)
            if file is None and key == "paper_b":
                paths = list(module.__path__)
                require(bool(paths) and all(Path(p).resolve().is_relative_to(runtime) for p in paths), "mixed namespace package")
            else:
                require(file is not None and Path(file).resolve().is_relative_to(runtime), "mixed runtime import: "+key)
    require(runner.execution_hash(author_file) == EXECUTION, "wrong full execution hash")
    protocol = read_json(runtime/"paper_b/ncmapss_bayes_20261114/PROTOCOL.json")
    require(runner.shared_execution() == protocol["shared_data_execution_sha256"], "wrong shared helpers")
    runner.verify_schema_contract()
    return runner, protocol, {"commit": COMMIT, "execution_sha256": EXECUTION,
                             "runtime_manifest_sha256": RUNTIME_MANIFEST, "runtime": str(runtime),
                             "files_sha256": verified, "author_files_sha256": author_hashes,
                             "author_manifest_sha256": AUTHOR_MANIFEST_SHA}


def load_development(cache, subset):
    import numpy as np
    require(subset in ("DS02", "DS05", "DS06", "DS07"), "unregistered development subset")
    cache = Path(cache)
    manifest = read_json(cache / f"{subset}_dev_manifest.json")
    require(manifest["official_test_decoded"] is False and
            manifest["decoded_keys"] == ["A_dev", "W_dev", "X_s_dev", "Y_dev"], "non-development data manifest")
    path = cache / f"{subset}_dev.npz"
    require(digest(path) == manifest["cache_sha256"], "changed development cache")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("raw", "y", "unit", "cycle")}
    require(data["raw"].shape == (len(data["y"]), 512, 18), "wrong physical feature contract")
    units = sorted(map(int, np.unique(data["unit"])))
    require(len(units) == 6, "expected six development engines")
    require(all(bool(np.isfinite(v).all()) for v in data.values()), "nonfinite development data")
    for unit in units:
        require(bool((np.diff(data["cycle"][data["unit"] == unit]) > 0).all()), "noncausal development flight order")
    return data, manifest, units


def independent_scalers(data, units):
    import numpy as np
    keep = np.isin(data["unit"], units)
    x = data["raw"][keep].astype(np.float64)
    return {"raw_mu": x.mean((0, 1)).tolist(), "raw_sd": np.maximum(x.std((0, 1)), 1e-6).tolist(),
            "age_q95": max(1., float(np.percentile(data["cycle"][keep], 95))),
            "target_mu": 0., "target_sd": 125., "fitted_units": list(units),
            "physical_transform": "training-only global channel mean/std; std floor1e-6"}


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(canonical(list(a.shape)))
        h.update(a.tobytes())
    return h.hexdigest()


def read_prediction(path, data, config, view):
    import numpy as np
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ["unit", "cycle", "y_true_raw", "y_pred", "y_pred_unbounded", "predictive_sd_cycles", "seed", "view"], "wrong prediction schema")
        rows = list(reader)
    keep = data["unit"] == config["validation_unit"]
    require(len(rows) == int(keep.sum()) and len(rows) > 0, "wrong flight count")
    for row, u, c, y in zip(rows, data["unit"][keep], data["cycle"][keep], data["y"][keep]):
        require(int(row["unit"]) == int(u) and float(row["cycle"]) == float(c) and
                float(row["y_true_raw"]) == float(y), "prediction engine/cycle/label mismatch")
        require(int(row["seed"]) == config["seed"] and row["view"] == view, "prediction seed/view mismatch")
    pred = np.array([float(r["y_pred"]) for r in rows], np.float64)
    unbounded = np.array([float(r["y_pred_unbounded"]) for r in rows], np.float64)
    sigma = np.array([float(r["predictive_sd_cycles"]) for r in rows], np.float64)
    require(bool(np.isfinite(sigma).all() and (sigma > 0).all()), "invalid original HNN predictive sigma")
    require(bool(np.isfinite(pred).all() and np.isfinite(unbounded).all()), "nonfinite prediction")
    require(bool(np.array_equal(pred, np.maximum(unbounded, 0))), "prediction clipping mismatch")
    return {"y": data["y"][keep].astype(np.float64), "prediction": pred, "unbounded": unbounded,
            "unit": data["unit"][keep], "cycle": data["cycle"][keep], "sigma_cycles": sigma}


def verify_cell(folder, runner, protocol, data, provenance, units, author_file):
    import numpy as np
    import torch
    seal = read_json(folder / "SUCCESS.json")
    require(seal["cell"] == folder.name, "SUCCESS identity mismatch")
    files = seal["files_sha256"]
    actual_files = {p.name for p in folder.iterdir() if p.is_file()}
    require(actual_files == set(files) | {"SUCCESS.json"}, "unsealed or missing cell files")
    for name, sha in files.items():
        require(Path(name).name == name and not (folder / name).is_symlink(), "unsafe artifact filename")
        require(digest(folder / name) == sha, "sealed artifact hash mismatch: " + name)
    config, saved_metrics, scalers = (read_json(folder / name) for name in ("config.json", "metrics.json", "scalers.json"))
    phase, subset = config["phase"], config["subset"]
    require(phase in ("inner", "outer") and subset in protocol["subsets"], "unregistered cell phase/subset")
    outer, val, seed = (config[k] for k in ("outer_heldout_unit", "validation_unit", "seed"))
    require(outer in units, "unknown outer engine")
    inner = units[(units.index(outer) + 1) % len(units)]
    expected_train = [u for u in units if u not in ((outer, inner) if phase == "inner" else (outer,))]
    require(config["train_units"] == expected_train, "wrong engine training split")
    require(val == (inner if phase == "inner" else outer), "wrong validation engine")
    require(seed == protocol["inner_selection_seed"] if phase == "inner" else seed in protocol["seeds"], "wrong phase seed")
    window = [config["history"], config["points_per_flight"]]
    require(window in protocol["window_catalog"], "unregistered window")
    cell = f"{subset}_{phase}_out{outer}_val{val}_h{window[0]}_p{window[1]}_s{seed}"
    require(config["cell"] == saved_metrics["cell"] == cell == folder.name, "cell name mismatch")
    require(config["protocol"] == protocol and config["execution_sha256"] == EXECUTION, "wrong cell protocol/execution")
    bare_config = {k: v for k, v in config.items() if k != "config_sha256"}
    require(hashlib.sha256(canonical(bare_config)).hexdigest() == config["config_sha256"], "config digest mismatch")
    require(config["source_sha256"] == provenance["source_sha256"] and
            config["cache_sha256"] == provenance["cache_sha256"] and
            config["author_file_sha256"] == AUTHOR_MODEL_SHA, "wrong data/source identity")
    expected_scalers = independent_scalers(data, expected_train)
    require(scalers == expected_scalers, "independently recomputed training-engine scalers differ")
    required = {"CLAIM.json", "config.json", "scalers.json", "initial.pt", "checkpoint.pt", "training_trace.json", "metrics.json"}
    views = ("clean",) if phase == "inner" else VIEWS
    required.update("prediction_" + view + ".csv" for view in views)
    require(set(files) == required, "unexpected sealed artifact set")
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    final = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    for checkpoint in (initial, final):
        require(checkpoint["config"] == config and checkpoint["scalers"] == scalers, "checkpoint config/scalers mismatch")
        require(checkpoint["author_provenance"] == saved_metrics["author_provenance"], "checkpoint author provenance mismatch")
    ih, fh = state_hash(initial["model"]), state_hash(final["model"])
    require(ih != fh and initial["initial_state_sha256"] == final["initial_state_sha256"] ==
            saved_metrics["initial_state_sha256"] == ih, "initial state evidence mismatch")
    require(final["final_state_sha256"] == saved_metrics["final_state_sha256"] == fh, "final state evidence mismatch")
    require(initial["model"].keys() == final["model"].keys(), "initial/final architecture mismatch")
    changed, delta2 = 0, 0.
    for key in initial["model"]:
        a, b = initial["model"][key], final["model"][key]
        require(a.shape == b.shape and a.dtype == b.dtype and bool(torch.isfinite(a).all()) and
                bool(torch.isfinite(b).all()), "nonfinite or incompatible state tensor")
        changed += int(not torch.equal(a, b))
        delta2 += float((b.double() - a.double()).square().sum())
    require(changed > 0 and delta2 > 0, "no observed training weight change")
    # CPU construction only, no model forward; independently checks complete
    # checkpoint structure, native parameter count, and seeded initial weights.
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = runner.BayesHNNAdapter(author_file, history=window[0], points=window[1]).cpu()
            require(state_hash(model.state_dict()) == ih, "seeded full-model initialization does not reproduce")
            model.load_state_dict(initial["model"], strict=True)
            model.load_state_dict(final["model"], strict=True)
            parameters = sum(p.numel() for p in model.parameters())
            shapes = [tuple(p.shape) for p in model.parameters()]
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
    del model, initial
    require(parameters == saved_metrics["parameters"], "full-model parameter count mismatch")
    expected_updates = protocol["epochs"] * math.ceil(int(np.isin(data["unit"], expected_train).sum()) / protocol["batch"])
    require(final["epochs_completed"] == protocol["epochs"] and final["optimizer_updates"] ==
            saved_metrics["optimizer_updates"] == expected_updates, "epoch/update count mismatch")
    optimizer = final["optimizer"]
    require(len(optimizer["param_groups"]) == 1, "unexpected optimizer groups")
    group = optimizer["param_groups"][0]
    require(group["lr"] == protocol["learning_rate"] and group["weight_decay"] == protocol["weight_decay"], "optimizer hyperparameter mismatch")
    ids = group["params"]
    require(len(ids) == len(shapes) and len(set(ids)) == len(ids), "optimizer parameter mapping mismatch")
    require(len(optimizer["state"]) > 0 and set(optimizer["state"]).issubset(ids), "empty/foreign optimizer state")
    shape_by_id = dict(zip(ids, shapes))
    for pid, state in optimizer["state"].items():
        require(float(state["step"]) == expected_updates, "Adam per-parameter step mismatch")
        for key in ("exp_avg", "exp_avg_sq"):
            require(tuple(state[key].shape) == shape_by_id[pid] and bool(torch.isfinite(state[key]).all()), "invalid Adam moment tensor")
    trace = read_json(folder / "training_trace.json")
    require([r["epoch"] for r in trace] == [1] + list(range(20, protocol["epochs"] + 1, 20)), "epoch trace incomplete")
    require(all(all(math.isfinite(r[k]) for k in ("normalized_gaussian_nll", "gradient_norm_last", "seconds")) for r in trace), "nonfinite trace")
    require(all(a["seconds"] <= b["seconds"] for a, b in zip(trace, trace[1:])), "nonmonotone trace clock")
    seconds = saved_metrics["training_seconds"]
    require(seconds > 0 and trace[-1]["seconds"] <= seconds and seconds <= protocol["maximum_cell_seconds"] + 1., "invalid training duration")
    require(0 < saved_metrics["peak_allocated_MiB"] <= protocol["maximum_device_memory_MiB"], "device memory budget mismatch")
    require(saved_metrics["official_test_read"] is False and saved_metrics["execution_sha256"] == EXECUTION and
            saved_metrics["config_sha256"] == config["config_sha256"] and saved_metrics["checkpoint_sha256"] == files["checkpoint.pt"], "metrics provenance mismatch")
    for key in ("phase", "subset", "seed", "history", "points_per_flight", "outer_heldout_unit", "validation_unit"):
        require(saved_metrics[key] == config[key], "metrics/config identity mismatch: " + key)
    require(set(saved_metrics["scores"]) == set(views), "wrong evaluated view set")
    predictions, rows = {}, []
    for view in views:
        pred = read_prediction(folder / f"prediction_{view}.csv", data, config, view)
        predictions[view] = pred
        for target, y in (("raw", pred["y"]), ("clip125", np.minimum(pred["y"], 125.))):
            score = metrics(y, pred["prediction"])
            require(set(score) == set(saved_metrics["scores"][view][target]), "metric schema mismatch")
            for key, value in score.items():
                close(value, saved_metrics["scores"][view][target][key], "recomputed metric mismatch: " + key)
            rows.append({"cell": cell, "phase": phase, "subset": subset, "outer_engine": outer, "seed": seed,
                         "history": window[0], "points": window[1], "view": view, "target": target,
                         **score, "parameters": parameters, "training_seconds": seconds,
                         "optimizer_updates": expected_updates, "changed_state_tensors": changed,
                         "weight_delta_l2": math.sqrt(delta2), "config_sha256": config["config_sha256"],
                         "checkpoint_sha256": files["checkpoint.pt"], "initial_state_sha256": ih,
                         "final_state_sha256": fh, "success_sha256": digest(folder / "SUCCESS.json")})
    del final
    return {"config": config, "rows": rows, "predictions": predictions,
            "files_sha256": {**files, "SUCCESS.json": digest(folder / "SUCCESS.json")}, "folder": str(folder)}


def audit(runtime, cache, runs, author_file):
    """Return JSON evidence plus private runtime context for optional replay."""
    import numpy as np
    import torch
    torch.set_num_threads(4)
    runner, protocol, source_evidence = load_runtime(runtime, author_file)
    runs = Path(runs).resolve()
    launch = read_json(runs / "LAUNCH.json")
    require(launch["protocol"] == protocol and launch["execution_sha256"] == EXECUTION, "wrong launch identity")
    # Snapshot lists before validating: cells which finish later enter the next snapshot.
    selections = {p.stem: read_json(p) for p in sorted((runs / "selections").glob("*.json"))}
    seals = sorted((runs / "cells").glob("*/SUCCESS.json"))
    started_cells = sorted(p.name for p in (runs / "cells").iterdir() if p.is_dir())
    data_cache = {s: load_development(cache, s) for s in protocol["subsets"]}
    verified, errors = {}, []
    for seal in seals:
        try:
            config = read_json(seal.parent / "config.json")
            require(config["subset"] in data_cache, "unregistered subset")
            data, provenance, units = data_cache[config["subset"]]
            item = verify_cell(seal.parent, runner, protocol, data, provenance, units, Path(author_file))
            verified[config["cell"]] = item
        except Exception as exc:
            errors.append({"cell": seal.parent.name, "error": type(exc).__name__ + ": " + str(exc)})
    selection_rows, missing = [], []
    for subset, (_, _, units) in data_cache.items():
        for outer in units:
            inner = units[(units.index(outer) + 1) % len(units)]
            inner_train = [u for u in units if u not in (outer, inner)]
            expected_inner = [f"{subset}_inner_out{outer}_val{inner}_h{h}_p{p}_s{protocol['inner_selection_seed']}" for h, p in protocol["window_catalog"]]
            missing.extend(cell for cell in expected_inner if cell not in verified)
            key = f"{subset}_out{outer}"
            selection = selections.get(key)
            if selection is None:
                missing.extend(f"{subset}_outer_out{outer}_seed{s}" for s in protocol["seeds"])
                if any(v["config"]["phase"] == "outer" and v["config"]["subset"] == subset and v["config"]["outer_heldout_unit"] == outer for v in verified.values()):
                    errors.append({"selection": key, "error": "outer exists without a snapshot-visible selection"})
                continue
            try:
                require(all(c in verified for c in expected_inner), "selection lacks all five validated inner cells")
                choices = [{"window": w, "rmse": verified[c]["rows"][0]["rmse"], "cell": c,
                            "config_sha256": verified[c]["config"]["config_sha256"]}
                           for w, c in zip(protocol["window_catalog"], expected_inner)]
                chosen = min(choices, key=lambda c: (c["rmse"], math.prod(c["window"]), *c["window"]))
                require(selection == {"outer_heldout_unit": outer, "inner_validation_unit": inner,
                        "inner_training_units": inner_train, "choices": choices,
                        "selected_window": chosen["window"], "selection_metric": "inner_raw_RMSE"}, "incorrect nested window selection")
                h, p = chosen["window"]
                for seed in protocol["seeds"]:
                    cell = f"{subset}_outer_out{outer}_val{outer}_h{h}_p{p}_s{seed}"
                    if cell not in verified:
                        missing.append(cell)
                outer_candidates = [v for v in verified.values() if v["config"]["phase"] == "outer" and
                                    v["config"]["subset"] == subset and v["config"]["outer_heldout_unit"] == outer]
                require(all([v["config"]["history"], v["config"]["points_per_flight"]] == chosen["window"] for v in outer_candidates), "outer window differs from inner choice")
                selection_rows.append({"subset": subset, "outer_engine": outer, "inner_engine": inner,
                                       "history": h, "points": p, "inner_rmse": chosen["rmse"],
                                       "selected_cell": chosen["cell"]})
            except Exception as exc:
                errors.append({"selection": key, "error": type(exc).__name__ + ": " + str(exc)})
    require(set(selections).issubset({f"{s}_out{u}" for s, (_, _, us) in data_cache.items() for u in us}), "foreign selection file")
    inner_items = [v for v in verified.values() if v["config"]["phase"] == "inner"]
    outer_items = [v for v in verified.values() if v["config"]["phase"] == "outer"]
    rows = [row for item in verified.values() for row in item["rows"]]
    summaries, seed_summaries = [], []
    for subset in protocol["subsets"]:
        for view in VIEWS:
            for target in ("raw", "clip125"):
                complete_seeds = []
                for seed in protocol["seeds"]:
                    selected = [r for r in rows if r["phase"] == "outer" and r["subset"] == subset and
                                r["seed"] == seed and r["view"] == view and r["target"] == target]
                    complete = len(selected) == 6 and len({r["outer_engine"] for r in selected}) == 6
                    record = {"subset": subset, "view": view, "target": target, "seed": seed,
                              "validated_engines": len(selected), "status": "COMPLETE" if complete else "PARTIAL"}
                    if complete:
                        count = sum(r["n"] for r in selected)
                        record.update(engine_equal_rmse=float(np.mean([r["rmse"] for r in selected])),
                                      flight_pooled_rmse=float(np.sqrt(sum(r["n"] * r["rmse"] ** 2 for r in selected) / count)),
                                      nasa_sum=sum(r["nasa_sum"] for r in selected),
                                      nasa_mean=sum(r["nasa_sum"] for r in selected) / count)
                        complete_seeds.append(record)
                    seed_summaries.append(record)
                summary = {"subset": subset, "view": view, "target": target,
                           "complete_seeds": len(complete_seeds), "status": "COMPLETE" if len(complete_seeds) == 3 else "PARTIAL"}
                if len(complete_seeds) == 3:
                    for metric in ("engine_equal_rmse", "flight_pooled_rmse", "nasa_sum", "nasa_mean"):
                        values = [r[metric] for r in complete_seeds]
                        summary[metric + "_mean"] = float(np.mean(values))
                        summary[metric + "_sd"] = float(np.std(values, ddof=1))
                summaries.append(summary)
    complete_marker = read_json(runs / "REQUESTED_SUBSETS_COMPLETE.json") if (runs / "REQUESTED_SUBSETS_COMPLETE.json").exists() else None
    failure_marker = read_json(runs / "FAILED.json") if (runs / "FAILED.json").exists() else None
    marker_ok = bool(complete_marker and complete_marker["subsets"] == protocol["subsets"] and complete_marker["execution_sha256"] == EXECUTION)
    full = not errors and not missing and len(inner_items) == 120 and len(outer_items) == 72 and len(selection_rows) == 24 and marker_ok and failure_marker is None
    result = {"title": "DEVELOPMENT_LOEO — complete author BayesRUL-HNN migration", "status": "FAIL" if errors or failure_marker else "PASS" if full else "PARTIAL",
              "audit_visible_sealed_cells": "FAIL" if errors else "PASS", "full_gate": full,
              "snapshot_utc": datetime.now(timezone.utc).isoformat(), "runs": str(runs), "source": source_evidence,
              "collector_sha256": digest(__file__), "official_test_read": False, "training_launched": False,
              "forward_executed": False, "expected": {"inner": 120, "outer": 72, "total": 192, "selections": 24},
              "counts": {"inner": len(inner_items), "outer": len(outer_items), "total": len(verified), "selections": len(selection_rows),
                         "unsealed_started_cells": len(set(started_cells) - {p.parent.name for p in seals})},
              "unsealed_cells": sorted(set(started_cells) - {p.parent.name for p in seals}),
              "missing": missing, "errors": errors, "selections": selection_rows,
              "case_rows": rows, "seed_summaries": seed_summaries, "summaries": summaries,
              "sealed_files": {cell: item["files_sha256"] for cell, item in verified.items()},
              "training_seconds": {"inner": sum(v["rows"][0]["training_seconds"] for v in inner_items),
                                   "outer": sum(v["rows"][0]["training_seconds"] for v in outer_items)},
              "completion_marker": complete_marker, "failure_marker": failure_marker,
              "limitations": ["Development engines only; no official TEST or paper-native protocol reproduction.",
                              "Inner HPO scores and 72 outer evaluations are never pooled.",
                              "Seed first averages six heldout-engine RMSE values; SD is across three seed means.",
                              "Six folds share training engines; seeds are not independent engines; no significance claim.",
                              "Training seconds exclude inference/IO; incomplete snapshots are not final performance claims.",
                              "Collector checks CPU model construction/state, not forward; replay is a separate explicit operation."]}
    return result, {"runner": runner, "protocol": protocol, "data": data_cache, "verified": verified, "author_file": Path(author_file)}


def new_output(output, protected):
    output = Path(output).resolve()
    require(not output.exists(), "output already exists; refusing overwrite")
    for path in protected:
        p = Path(path).resolve()
        require(not output.is_relative_to(p) and not p.is_relative_to(output), "output overlaps a protected input")
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_csv(path, rows):
    with open(path, "x", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)


def write_result(output, result):
    (output / "COLLECTION.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for phase in ("inner", "outer"):
        write_csv(output / f"{phase}_cases.csv", [r for r in result["case_rows"] if r["phase"] == phase])
    for name in ("selections", "seed_summaries", "summaries"):
        write_csv(output / (name + ".csv"), result[name])
    lines = ["# DEVELOPMENT_LOEO — full BayesRUL-HNN migration", "", f"Status: **{result['status']}**; full 192-training gate: **{result['full_gate']}**.", "",
             f"Validated: {result['counts']['inner']}/120 inner and {result['counts']['outer']}/72 outer; {result['counts']['selections']}/24 nested selections.", "",
             "No training, model forward, or official TEST was executed by this collector.", "",
             "|Subset|View|Target|Status|Engine-equal RMSE mean ± seed SD|", "|---|---|---|---|---|"]
    for r in result["summaries"]:
        value = f"{r['engine_equal_rmse_mean']:.6f} ± {r['engine_equal_rmse_sd']:.6f}" if r["status"] == "COMPLETE" else "unavailable (incomplete 6-engine × 3-seed matrix)"
        lines.append(f"|{r['subset']}|{r['view']}|{r['target']}|{r['status']}|{value}|")
    lines.extend(["", *["- " + s for s in result["limitations"]], "", "Errors: " + str(len(result["errors"])) + "; see COLLECTION.json."])
    (output / "COLLECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "cache", "runs", "author-file", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(not args.output.exists(), "output already exists; refusing overwrite")
    result, _ = audit(args.runtime, args.cache, args.runs, args.author_file)
    output = new_output(args.output, (args.runtime, args.cache, args.runs, args.author_file.parents[3]))
    write_result(output, result)
    print(json.dumps({"status": result["status"], "full_gate": result["full_gate"], "counts": result["counts"], "errors": result["errors"], "output": str(output)}))
    return 1 if result["status"] == "FAIL" else 2 if args.require_complete and not result["full_gate"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
