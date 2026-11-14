"""Collect and optionally replay the frozen D3 INNER screen, never outer/TEST.

Implemented by the producer tool author; this is inspectable verification, not
independent scientific review. Default executes CPU checkpoint inspection only.
Only --replay executes prediction. Outputs are exclusive-create and separate.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
COMMIT = "7bb98787459c3bf2b50fc64b84c5ec720fde243b"
EXECUTION = "d5f117e60734d99a02a56d7d3d5f50961ecaaedd24ab0c13ced57c24dfe80ced"
MANIFEST_SHA = "12a391378e1c5624f7818855220aa72038d02a886afab7d204c8118230d6929c"
CHECKS_SHA = "2d57d9e8840b706ff0b3c7fbcec74f5a05b6943febbe953c68fa66cc01986aff"
BASE_EXECUTION = "0d5ab920d31ec11767bb38f9ed717ce21f31ece3c6d981de7453cbd672d08009"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.05)  # Producer exclusive-create can be briefly mid-write.


def dump(path, value):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write("\n")


def close(a, b, label, atol=1e-9, rtol=1e-9):
    import numpy as np
    require(np.shape(a) == np.shape(b) and bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)), label)


def compare(a, b, label):
    """Compare independently recomputed nested evidence without rounding gates."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), label + " keys")
        for key in a:
            compare(a[key], b[key], label + "/" + str(key))
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), label + " length")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, label + "/" + str(i))
    elif isinstance(a, float):
        close(a, b, label)
    else:
        require(a == b, label)


def safe_file(root, relative):
    p = (root / relative).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file() and not (root / relative).is_symlink(), "unsafe/missing file: " + relative)
    return p


def verify_seal(root, seal_name="SUCCESS.json"):
    seal = read(root / seal_name)
    files = seal["files_sha256"]
    actual = {str(p.relative_to(root)).replace('\\', '/') for p in root.rglob("*") if p.is_file()}
    require(actual == set(files) | {seal_name}, "unsealed/missing artifact files: " + str(root))
    for relative, digest in files.items():
        require(sha(safe_file(root, relative)) == digest, "sealed artifact changed: " + relative)
    return seal


def load_runtime(runtime):
    runtime = Path(runtime).resolve()
    manifest_path = runtime / "GIT_SOURCE_MANIFEST.json"
    require(sha(manifest_path) == MANIFEST_SHA, "wrong frozen source manifest bytes")
    manifest = read(manifest_path)
    require(manifest["commit"] == COMMIT, "wrong producer commit")
    paths = [e["path"] for e in manifest["files"]]
    require(len(paths) == len(set(paths)) == 13, "wrong source manifest inventory")
    for entry in manifest["files"]:
        p = safe_file(runtime, entry["path"]); content = p.read_bytes()
        require(len(content) == entry["bytes"] and hashlib.sha256(content).hexdigest() == entry["sha256"], "changed source: " + entry["path"])
        require(hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() == entry["git_blob_sha1"], "source Git blob mismatch")
    for key in tuple(sys.modules):
        if key == "paper_b" or key.startswith("paper_b."):
            del sys.modules[key]
    sys.path.insert(0, str(runtime))
    producer = importlib.import_module("paper_b.ncmapss_conditioning_20261114.run")
    frozen, protocol, model_protocol = producer.load_frozen()
    for key, module in tuple(sys.modules.items()):
        if key == "paper_b" or key.startswith("paper_b."):
            file = getattr(module, "__file__", None)
            locations = [Path(file)] if file else [Path(x) for x in getattr(module, "__path__", [])]
            require(locations and all(p.resolve().is_relative_to(runtime) for p in locations), "mixed runtime import: " + key)
    require(producer.execution_hash() == EXECUTION and frozen.execution_hash() == BASE_EXECUTION, "execution identity mismatch")
    require(sha(runtime / "paper_b/ncmapss_conditioning_20261114/checks.py") == CHECKS_SHA, "wrong admission checks")
    return producer, frozen, protocol, model_protocol, manifest


def independent_data(cache, subset, protocol):
    import numpy as np
    manifest = read(cache / f"{subset}_dev_manifest.json")
    require(manifest["official_test_decoded"] is False and manifest["decoded_keys"] == ["A_dev", "W_dev", "X_s_dev", "Y_dev"], "non-development cache")
    path = cache / f"{subset}_dev.npz"
    require(sha(path) == manifest["cache_sha256"] == protocol["expected_cache_sha256"][subset], "wrong cache bytes")
    with np.load(path, allow_pickle=False) as z:
        unit = z["unit"]
        units = sorted(map(int, np.unique(unit)))
        require(len(units) == 6, "six development engines required")
        keep = unit != units[0]
        # Outer labels are never used in a fit, metric, bundle or prediction.
        data = {"unit": unit[keep], "raw": z["raw"][keep], "cycle": z["cycle"][keep], "y": z["y"][keep]}
    require(data["raw"].shape == (len(data["y"]), 512, 18), "wrong raw shape")
    train, val = units[2:], units[1]
    train_keep = np.isin(data["unit"], train)
    x = data["raw"][train_keep].astype(np.float64); y = np.minimum(data["y"][train_keep].astype(np.float64), 125.)
    scalers = {"raw_mu": x.mean((0, 1)).tolist(), "raw_sd": np.maximum(x.std((0, 1)), 1e-6).tolist(),
               "age_q95": max(1., float(np.percentile(data["cycle"][train_keep], 95))),
               "target_mu": float(y.mean()), "target_sd": max(1., float(y.std())), "fitted_units": train}
    standardized = ((data["raw"][train_keep].astype(np.float32) - np.asarray(scalers["raw_mu"], np.float32)) /
                    np.asarray(scalers["raw_sd"], np.float32)).astype(np.float64)
    w = standardized[..., :4]
    phi = np.stack([np.ones(w.shape[:-1])] + [w[..., i] for i in range(4)] +
                   [w[..., i] * w[..., j] for i in range(4) for j in range(i, 4)], -1).reshape(-1, 15)
    target = standardized[..., 4:].reshape(-1, 14)
    penalty = np.eye(15) * .001; penalty[0, 0] = 0
    coef = np.linalg.solve(phi.T @ phi / len(phi) + penalty, phi.T @ target / len(phi))
    scale = (target - phi @ coef).std(0)
    expected = {"coef": coef, "scale_unfloored": scale, "scale": np.maximum(scale, 1e-6),
                "training_flights": int(train_keep.sum()), "training_rows": len(phi)}
    return {"data": data, "manifest": manifest, "units": units, "train": train, "val": val,
            "outer": units[0], "scalers": scalers, "processor_numeric": expected}


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(canonical(list(a.shape))); h.update(a.tobytes())
    return h.hexdigest()


def scores(y, prediction):
    import numpy as np
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    require(y.shape == prediction.shape and len(y) and bool(np.isfinite(y).all() and np.isfinite(prediction).all()), "invalid metric inputs")
    e = prediction - y; nasa = np.expm1(np.where(e >= 0, e / 10., -e / 13.))
    result = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(abs(e).mean()), "nasa_sum": float(nasa.sum()),
              "nasa_mean": float(nasa.mean()), "bias": float(e.mean()), "n": len(y)}
    require(all(math.isfinite(v) for v in result.values()), "nonfinite metrics")
    return result


def validate_cell(folder, context, frozen, protocol, model_protocol):
    import numpy as np
    import torch
    seal = verify_seal(folder)
    required = {"CLAIM.json", "config.json", "scalers.json", "processor.json", "initial.pt", "checkpoint.pt", "step_trace.csv", "training_trace.json", "prediction.csv", "metrics.json"}
    require(set(seal["files_sha256"]) == required and seal["cell"] == folder.name, "wrong cell inventory")
    c, scaler, processor, m = (read(folder / n) for n in ("config.json", "scalers.json", "processor.json", "metrics.json"))
    s, family, seed = c["subset"], c["family"], c["seed"]
    require(s in protocol["subsets"] and family in protocol["families"] and seed in protocol["seeds"], "foreign cell")
    require(c["cell"] == folder.name == f"{s}_inner_screen_out{context['outer']}_val{context['val']}_{family}_s{seed}", "wrong cell name")
    require(c["phase"] == "inner_screen" and c["excluded_outer_unit"] == context["outer"] and c["validation_unit"] == context["val"] and c["train_units"] == context["train"], "wrong inner split")
    require(c["protocol"] == protocol and c["frozen_model_protocol"] == model_protocol and c["execution_sha256"] == EXECUTION and c["frozen_execution_sha256"] == BASE_EXECUTION, "wrong config/source")
    require(c["config_sha256"] == identity({k: v for k, v in c.items() if k != "config_sha256"}), "bad config digest")
    require(c["conditioning"] == protocol["families"][family]["conditioning"] and c["auxiliary_weight"] == protocol["families"][family]["auxiliary_weight"], "wrong factorial condition")
    require(c["cache_sha256"] == context["manifest"]["cache_sha256"] and c["source_sha256"] == context["manifest"]["source_sha256"], "wrong data provenance")
    compare(scaler, context["scalers"], "independent train-four scalers")
    require(identity(scaler) == c["scalers_sha256"] == processor["global_scalers_sha256"], "scaler binding mismatch")
    require(processor["processor_sha256"] == c["processor_sha256"] == identity({k: v for k, v in processor.items() if k != "processor_sha256"}), "processor digest mismatch")
    require(processor["fitted_units"] == context["train"] and processor["cache_identity"] == {k: c[k] for k in ("cache_sha256", "source_sha256")}, "processor fitting/provenance mismatch")
    expected = context["processor_numeric"]
    for key, expected_key in (("coefficients", "coef"), ("empirical_residual_scale", "scale"), ("empirical_residual_scale_unfloored", "scale_unfloored")):
        close(processor[key], expected[expected_key], "independent FP32-coordinate ridge/scale: " + key)
    require(processor["training_flights"] == expected["training_flights"] and processor["training_rows"] == expected["training_rows"] and
            processor["ridge"] == .001 and processor["floor"] == 1e-6 and processor["intercept_penalized"] is False and
            processor["fixed_estimated_coefficients"] == 210 and processor["fixed_empirical_scale_values"] == 14 and
            processor["additional_neural_trainable_parameters"] == 0 and processor["label_arrays_accessed"] is False, "wrong fixed processor specification")
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    final = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    for cp in (initial, final):
        require(cp["config"] == c and cp["scalers"] == scaler and cp["processor"] == processor, "checkpoint metadata mismatch")
    with torch.random.fork_rng(devices=[]):
        model = frozen.Model("raw", model_protocol, seed).cpu()
        ih = state_hash(model.state_dict())
        require(ih == state_hash(initial["model"]), "CPU native initialization mismatch")
        require(initial["pairing"]["encoder_initial_sha256"] == frozen.state_hash(model.encoder) and
                initial["pairing"]["head_initial_sha256"] == frozen.state_hash(model.head), "initial encoder/head mismatch")
        model.load_state_dict(final["model"], strict=True)
        count = sum(p.numel() for p in model.parameters()); shapes = [tuple(p.shape) for p in model.parameters()]
    fh = state_hash(final["model"])
    require(ih != fh and fh == final["final_state_sha256"] == m["final_state_sha256"] and ih == final["initial_state_sha256"] == m["initial_state_sha256"], "weight change/state hashes mismatch")
    require(initial["model"].keys() == final["model"].keys(), "state structure changed")
    changed = sum(not torch.equal(initial["model"][k], final["model"][k]) for k in initial["model"])
    require(changed > 0 and all(bool(torch.isfinite(v).all()) for cp in (initial, final) for v in cp["model"].values()), "nonfinite/unchanged weights")
    require(m["parameters"] == count, "parameter count mismatch")
    steps = 200 * math.ceil(expected["training_flights"] / 64)
    require(final["epochs_completed"] == m["epochs_completed"] == 200 and final["optimizer_updates"] == m["optimizer_updates"] == steps, "epoch/update mismatch")
    opt = final["optimizer"]; require(len(opt["param_groups"]) == 1 and opt["state"], "missing AdamW state")
    group = opt["param_groups"][0]; ids = group["params"]
    require(group["lr"] == .001 and group["weight_decay"] == .0001 and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8, "AdamW hyperparameters mismatch")
    require(len(ids) == len(shapes) == len(set(ids)) and set(opt["state"]).issubset(ids), "optimizer parameter mapping mismatch")
    mapping = dict(zip(ids, shapes))
    for pid, state in opt["state"].items():
        require(float(state["step"]) == steps, "AdamW step mismatch")
        for key in ("exp_avg", "exp_avg_sq"):
            require(tuple(state[key].shape) == mapping[pid] and bool(torch.isfinite(state[key]).all()), "invalid AdamW moment")
        require(bool((state["exp_avg_sq"] >= 0).all()), "negative AdamW squared moment")
    trace = read(folder / "training_trace.json")
    with open(folder / "step_trace.csv", newline="", encoding="utf-8") as f:
        step_rows = list(csv.DictReader(f))
    require(len(trace) == 200 and len(step_rows) == steps, "incomplete step/epoch trace")
    generator = torch.Generator(device="cpu").manual_seed(seed + 20000)
    oh, bh = hashlib.sha256(), hashlib.sha256(); cursor = 0; epoch_hashes = []
    last_seconds = -1.
    for epoch in range(200):
        order = torch.randperm(expected["training_flights"], generator=generator).numpy().astype(np.int64, copy=False)
        eh = hashlib.sha256(order.tobytes()).hexdigest(); epoch_hashes.append(eh)
        oh.update(canonical({"epoch": epoch + 1, "n": len(order)})); oh.update(order.tobytes()); bh.update(canonical([epoch + 1, epoch % 4]))
        tasks, auxes = [], []
        for batch, lo in enumerate(range(0, len(order), 64)):
            r = step_rows[cursor]; cursor += 1; q = order[lo:lo + 64]
            require([int(r[k]) for k in ("epoch", "batch", "update", "n", "block")] == [epoch + 1, batch, cursor, len(q), epoch % 4] and r["query_sha256"] == hashlib.sha256(q.tobytes()).hexdigest(), "step/order/block mismatch")
            task, aux, weighted, total, norm, seconds = (float(r[k]) for k in ("task", "auxiliary", "weighted_auxiliary", "total", "gradient_norm", "seconds"))
            require(all(math.isfinite(x) for x in (task, aux, weighted, total, norm, seconds)) and task >= 0 and aux >= 0 and norm >= 0 and seconds >= last_seconds, "invalid step values")
            last_seconds = seconds
            close(weighted, c["auxiliary_weight"] * aux, "weighted auxiliary", atol=1e-12, rtol=1e-12)
            close(total, task + c["auxiliary_weight"] * aux, "FP32 total loss", atol=2e-6, rtol=2e-6)
            tasks.append(task); auxes.append(aux)
        tr = trace[epoch]
        require(tr["epoch"] == epoch + 1 and tr["block"] == epoch % 4 and tr["updates"] == cursor and tr["order_sha256"] == eh, "epoch trace identity mismatch")
        close(tr["task"], float(np.mean(tasks)), "epoch task"); close(tr["auxiliary"], float(np.mean(auxes)), "epoch auxiliary")
        require(tr["seconds"] >= last_seconds, "invalid epoch clock")
    pairing = {**initial["pairing"], "optimizer_updates": steps, "epoch_order_sha256": oh.hexdigest(), "block_schedule_sha256": bh.hexdigest()}
    require(initial["pairing"]["initial_state_sha256"] == ih and initial["pairing"]["processor_sha256"] == c["processor_sha256"] and
            initial["pairing"]["scalers_sha256"] == c["scalers_sha256"], "initial pairing metadata mismatch")
    require(final["pairing"] == m["pairing"] == pairing and m["epoch_order_sha256s"] == epoch_hashes, "cumulative pairing mismatch")
    require(m["training_seconds"] >= trace[-1]["seconds"] > 0 and 0 < m["peak_allocated_MiB"] <= 12000, "time/memory evidence invalid")
    require(final["outer_evaluated"] is False and final["validation_evaluations_before_checkpoint"] == 0 and
            m["outer_evaluated"] is False and m["official_TEST_read"] is False and m["midtraining_validation_evaluations"] == 0 and m["final_inner_prediction_sweeps"] == 1, "evaluation boundary violation")
    require(m["cell"] == c["cell"] and m["phase"] == "inner_screen" and all(m[k] == c[k] for k in ("subset", "seed", "family", "config_sha256", "execution_sha256")) and
            m["checkpoint_sha256"] == seal["files_sha256"]["checkpoint.pt"], "metrics identity mismatch")
    with open(folder / "prediction.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f); require(reader.fieldnames == ["unit", "cycle", "y_true_raw", "y_pred", "y_pred_unbounded", "seed", "family"], "wrong prediction schema"); rows = list(reader)
    keep = context["data"]["unit"] == context["val"]
    require(len(rows) == int(keep.sum()), "wrong validation flight count")
    for r, cycle, y in zip(rows, context["data"]["cycle"][keep], context["data"]["y"][keep]):
        require(int(r["unit"]) == context["val"] and float(r["cycle"]) == float(cycle) and float(r["y_true_raw"]) == float(y) and int(r["seed"]) == seed and r["family"] == family, "inner prediction labels/identity mismatch")
    pred = np.asarray([float(r["y_pred"]) for r in rows]); unbounded = np.asarray([float(r["y_pred_unbounded"]) for r in rows])
    require(bool(np.array_equal(pred, np.maximum(unbounded, 0))), "postprocessing mismatch")
    result_scores = {"raw": scores(context["data"]["y"][keep], pred), "clip125": scores(np.minimum(context["data"]["y"][keep], 125.), pred)}
    compare(m["scores"], result_scores, "independent CSV metrics")
    require(m["fixed_processor_coefficients"] == 210 and m["fixed_empirical_scales"] == 14 and m["additional_neural_parameters"] == 0, "processor parameter accounting mismatch")
    result = {"cell": folder.name, "subset": s, "family": family, "seed": seed, "phase": "inner_screen", "scores": result_scores,
              "pairing": pairing, "parameters": count, "training_seconds": m["training_seconds"], "changed_state_tensors": changed,
              "checkpoint_sha256": m["checkpoint_sha256"], "success_sha256": sha(folder / "SUCCESS.json"), "processor_sha256": c["processor_sha256"]}
    result["sealed_files_sha256"] = seal["files_sha256"]
    del model, initial, final
    return result, {"config": c, "processor": processor, "scalers": scaler, "unbounded": unbounded, "prediction": pred, "folder": folder}


def independent_summary(results, protocol):
    import numpy as np
    expected = {(s, f, seed) for s in protocol["subsets"] for f in protocol["families"] for seed in protocol["seeds"]}
    require(len(results) == 48 and {(r["subset"], r["family"], r["seed"]) for r in results} == expected, "complete48 required")
    families, gain = [], {}
    for s in protocol["subsets"]:
        means = {}
        for f in protocol["families"]:
            selected = [r["scores"]["raw"] for r in results if r["subset"] == s and r["family"] == f]
            row = {"subset": s, "family": f}
            for metric, label in (("rmse", "rmse"), ("nasa_sum", "nasa_sum"), ("nasa_mean", "nasa_mean")):
                v = [r[metric] for r in selected]; row[label + "_mean"] = float(np.mean(v)); row[label + "_sd"] = float(np.std(v, ddof=1))
            families.append(row); means[f] = row["rmse_mean"]
        gain[s] = {a: 1 - means["resid_" + a] / means["raw_" + a] if means["raw_" + a] > 0 else None for a in ("aux1", "aux0")}
    identifiable = all(v is not None for g in gain.values() for v in g.values())
    same = [s for s in protocol["subsets"] if all(gain[s][a] is not None and gain[s][a] >= .02 for a in ("aux1", "aux0"))]
    no_worse = identifiable and all(v >= -.02 for g in gain.values() for v in g.values())
    return {"families": families, "gain_ratio_of_three_seed_means": gain, "same_qualifying_subsets_both_aux": same,
            "all_subset_aux_no_worse_than_2pct": no_worse, "gate": "UNIDENTIFIABLE" if not identifiable else "MET" if len(same) >= 3 and no_worse else "NOT_MET"}


def replay(item, context, producer, frozen, model_protocol, device, output):
    import numpy as np
    import torch
    c = item["config"]; path = item["folder"] / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = frozen.Model("raw", model_protocol, c["seed"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    require(frozen.state_hash(model) == checkpoint["final_state_sha256"], "replay checkpoint state mismatch")
    held = frozen.bundle(context["data"], [context["val"]], item["scalers"], model_protocol, device)
    held = producer.apply_bundle(held, item["processor"], c["conditioning"])
    actual, _ = frozen.predict(model, held, item["scalers"], 64)
    require(bool(np.isfinite(actual).all()), "nonfinite replay prediction")
    saved = item["unbounded"]; error = float(np.max(np.abs(actual - saved)))
    bitwise = actual.dtype == saved.dtype and actual.tobytes() == saved.tobytes()
    with open(output / (c["cell"] + "_replay.csv"), "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["unit", "cycle", "saved_unbounded", "replayed_unbounded", "absolute_difference"])
        writer.writerows(zip(held["unit"], held["cycle"], saved, actual, abs(actual - saved)))
    result = {"cell": c["cell"], "device": device, "checkpoint_sha256": sha(path), "unbounded_max_absolute_difference": error,
              "bitwise_equal": bitwise, "absolute_tolerance": 1e-5, "within_tolerance": error <= 1e-5,
              "clipped_max_absolute_difference": float(np.max(abs(np.maximum(actual, 0) - item["prediction"]))),
              "optimizer_updates": 0, "outer_bundle_constructed": False}
    del model, held, checkpoint
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "cache", "runs", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--replay", action="store_true"); p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    output = args.output.resolve()
    require(not output.exists(), "output already exists; refusing overwrite")
    for protected in (args.runtime, args.cache, args.runs):
        protected = protected.resolve()
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), "output overlaps protected inputs")
    require(args.limit is None or args.limit > 0, "limit must be positive")
    output.mkdir(parents=True, exist_ok=False)
    report = {"title": "D3 DEVELOPMENT INNER_SCREEN audit", "tool_role": "producer-implemented tool; independent review required",
              "audit_source_sha256": sha(__file__), "execution_sha256": EXECUTION, "status": "PARTIAL", "full_gate": False,
              "snapshot_utc": datetime.now(timezone.utc).isoformat(), "errors": [], "cells": [], "replay": [],
              "official_TEST_read": False, "outer_evaluated": False, "optimizer_updates": 0,
              "replay_requested": args.replay, "limit": args.limit}
    report.update(producer_commit=COMMIT, frozen_execution_sha256=BASE_EXECUTION,
                  runtime=str(args.runtime.resolve()), cache=str(args.cache.resolve()), runs=str(args.runs.resolve()),
                  numeric_tolerances={"independent_scaler_coefficient_metrics_atol": 1e-9,
                                      "independent_scaler_coefficient_metrics_rtol": 1e-9,
                                      "FP32_total_loss_atol": 2e-6, "FP32_total_loss_rtol": 2e-6,
                                      "replay_unbounded_absolute_only": 1e-5})
    try:
        producer, frozen, protocol, model_protocol, manifest = load_runtime(args.runtime)
        import torch
        torch.set_num_threads(4)
        report["runtime_manifest_sha256"] = MANIFEST_SHA
        report["software"] = {"torch": torch.__version__, "cuda": torch.version.cuda}
        launch = read(args.runs / "LAUNCH.json")
        require(launch["protocol"] == protocol and launch["execution_sha256"] == EXECUTION, "wrong launch")
        require(launch["production_files_sha256"] == {n: sha(args.runtime / "paper_b/ncmapss_conditioning_20261114" / n) for n in producer.PRODUCTION} and launch["frozen_files_sha256"] == producer.FROZEN_HASHES, "launch source binding mismatch")
        contexts = {s: independent_data(args.cache, s, protocol) for s in protocol["subsets"]}
        snapshot = sorted((args.runs / "cells").glob("*/SUCCESS.json"))
        selected = snapshot[:args.limit] if args.limit else snapshot
        report["visible_sealed_cells"] = len(snapshot)
        report["unsealed_cells"] = sorted(d.name for d in (args.runs / "cells").iterdir() if d.is_dir() and not (d / "SUCCESS.json").exists())
        if args.replay:
            if args.device.startswith("cuda"):
                producer.configure_device(args.device, protocol)
            else:
                require(args.device == "cpu", "unregistered replay device")
                torch.use_deterministic_algorithms(True)
        details = {}
        for seal in selected:
            try:
                config = read(seal.parent / "config.json")
                require(config["subset"] in contexts, "foreign subset")
                record, item = validate_cell(seal.parent, contexts[config["subset"]], frozen, protocol, model_protocol)
                require(record["cell"] not in details, "duplicate cell")
                report["cells"].append(record); details[record["cell"]] = item
                if args.replay:
                    rr = replay(item, contexts[config["subset"]], producer, frozen, model_protocol, args.device, output)
                    report["replay"].append(rr)
                    require(rr["within_tolerance"], "replay absolute difference exceeds1e-5")
            except Exception as exc:
                report["errors"].append({"cell": seal.parent.name, "type": type(exc).__name__, "message": str(exc)})
        expected = {(s, f, seed) for s in protocol["subsets"] for f in protocol["families"] for seed in protocol["seeds"]}
        actual = {(r["subset"], r["family"], r["seed"]) for r in report["cells"]}
        require(len(actual) == len(report["cells"]) and actual.issubset(expected), "duplicate/foreign matrix cells")
        pairing_verified = []
        for s in protocol["subsets"]:
            processor_hashes = {r["processor_sha256"] for r in report["cells"] if r["subset"] == s}
            require(len(processor_hashes) <= 1, "processor differs across subset families/seeds")
            for seed in protocol["seeds"]:
                group = [r for r in report["cells"] if r["subset"] == s and r["seed"] == seed]
                require(all(r["pairing"] == group[0]["pairing"] for r in group), "four-family pairing mismatch")
                if len(group) == 4:
                    pair_path = args.runs / f"PAIRING_{s}_s{seed}.json"
                    if pair_path.exists():
                        pair = read(pair_path)
                        require(pair == {"status": "PASS", "families": list(protocol["families"]), "pairing": group[0]["pairing"]}, "producer pairing seal mismatch")
                        pairing_verified.append([s, seed])
                    # The producer may still be between its last cell seal and
                    # the pairing/completion writes. Such a snapshot is PARTIAL.
        report["pairing_groups_verified"] = pairing_verified
        report["missing_cells"] = [list(x) for x in sorted(expected - actual)]
        complete = (actual == expected and len(snapshot) == 48 and len(pairing_verified) == 12
                    and (args.runs / "SUCCESS.json").exists() and not report["errors"])
        if complete:
            global_seal = verify_seal(args.runs)
            require(global_seal["training_runs"] == 48 and global_seal["execution_sha256"] == EXECUTION, "wrong global completion seal")
            summary = independent_summary(report["cells"], protocol)
            saved_summary = read(args.runs / "SUMMARY.json")
            for key, value in summary.items():
                compare(value, saved_summary[key], "producer summary/" + key)
            require(saved_summary["status"] == "COMPLETE" and saved_summary["training_runs"] == 48 and
                    0 < saved_summary["elapsed_shared_gpu_wall_seconds"] <= 28800, "invalid completion/time budget")
            per_seed = [{"subset": r["subset"], "family": r["family"], "seed": r["seed"], **r["scores"]["raw"]} for r in report["cells"]]
            key = lambda r: (r["subset"], r["seed"], r["family"])
            compare(sorted(per_seed, key=key), sorted(saved_summary["per_seed"], key=key), "all48 per-seed summary")
            report.update(status="PASS", full_gate=True, summary=summary, global_seal_sha256=sha(args.runs / "SUCCESS.json"))
        if (args.runs / "FAILED.json").exists():
            report["producer_failure"] = read(args.runs / "FAILED.json")
            report["errors"].append({"scope": "producer", "message": "producer FAILED marker exists"})
    except Exception as exc:
        report["errors"].append({"scope": "global", "type": type(exc).__name__, "message": str(exc)})
    if report["errors"]:
        report.update(status="FAIL", full_gate=False)
    dump(output / "AUDIT.json", report)
    rows = [{k: r[k] for k in ("cell", "subset", "family", "seed", "parameters", "training_seconds", "checkpoint_sha256", "processor_sha256")} | r["scores"]["raw"] for r in report["cells"]]
    with open(output / "inner_cases.csv", "x", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ["# D3 DEVELOPMENT INNER_SCREEN", "", f"Audit status: **{report['status']}**; full48 gate evaluable: **{report['full_gate']}**.", "",
             f"Validated cells: {len(report['cells'])}; replayed: {len(report['replay'])}; errors: {len(report['errors'])}.", "",
             "Producer-implemented tool; independent review is separate. Partial snapshots cannot pass the48-cell screening gate. No outer/TEST evaluation or optimizer update is performed."]
    if report.get("summary") and report["full_gate"]:
        lines += ["", "Classical conditioning screening gate: **" + report["summary"]["gate"] + "**.",
                  "This establishes neither novelty nor outer confirmation. One inner engine per subset and overlapping development evidence limit inference."]
    (output / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "full_gate": report["full_gate"], "validated_cells": len(report["cells"]), "replayed_cells": len(report["replay"]), "errors": report["errors"], "output": str(output)}))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
