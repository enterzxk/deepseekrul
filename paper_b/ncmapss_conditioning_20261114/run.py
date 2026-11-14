"""48 fixed paired inner-screen trainings; classical conditioning, no outer/TEST.

No training starts on import or --help. A matching zero-optimizer hardware
admission JSON is mandatory. Frozen full Model/bundle helpers are byte-verified.
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
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from .preprocess import apply_bundle, canonical, fit_processor, identity, require

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "ncmapss_response"
FROZEN_EXECUTION = "0d5ab920d31ec11767bb38f9ed717ce21f31ece3c6d981de7453cbd672d08009"
FROZEN_HASHES = {
    "__init__.py": "8a27d37328ce836569918cccaf5cb1dd43a8d7c69cc37032458c8391ea2f7c86",
    "models.py": "8688d5f64c9366eb2e61f8835b99434058fcd0054df6363d5ea74ca426034fc9",
    "prepare.py": "cc7a9fcb6257b0622734cd4f04691229f63508396ca3fb07c206a4e0ca680d62",
    "run.py": "4284f7e1f963fe5315aadb8a0e16a52455acfd7847e03a6bdf35e47ebb61d071",
    "PROTOCOL.json": "e0524b827022745593ad70eeefd544de3aa06b0497f4457ba63f0c796bd95726"}
PRODUCTION = ("__init__.py", "preprocess.py", "run.py", "PROTOCOL.json")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write("\n")


def verify_frozen():
    h = hashlib.sha256()
    for name, expected in FROZEN_HASHES.items():
        content = (FROZEN / name).read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected, "changed frozen production source: " + name)
        h.update(name.encode()); h.update(content)
    require(h.hexdigest() == FROZEN_EXECUTION, "wrong frozen execution")
    protocol = read(HERE / "PROTOCOL.json")
    require(protocol["frozen_files_sha256"] == FROZEN_HASHES and protocol["frozen_execution_sha256"] == FROZEN_EXECUTION,
            "new protocol identifies different frozen code")
    require([protocol[k] for k in ("epochs", "batch", "learning_rate", "weight_decay", "gradient_clip", "span", "points_per_flight")]
            == [200, 64, .001, .0001, 1., 70, 512], "controller constants differ from fixed protocol")
    require(protocol["budget"] == {"one_process_shared_gpu_wall_seconds": 28800, "maximum_device_memory_MiB": 12000,
                                  "parallel_subset_workers": False}, "controller budget differs from fixed protocol")
    return protocol


def execution_hash():
    verify_frozen()
    h = hashlib.sha256()
    for name in PRODUCTION:
        h.update(name.encode()); h.update((HERE / name).read_bytes())
    h.update(FROZEN_EXECUTION.encode())
    return h.hexdigest()


def load_frozen():
    protocol = verify_frozen()
    module = importlib.import_module("paper_b.ncmapss_response.run")
    require(Path(module.__file__).resolve() == FROZEN / "run.py", "mixed checkout imported")
    require(module.execution_hash() == FROZEN_EXECUTION, "imported helpers differ")
    return module, protocol, read(FROZEN / "PROTOCOL.json")


def configure_device(device, protocol):
    import torch
    require(str(device).startswith("cuda") and torch.cuda.is_available(), "production/admission requires CUDA; no CPU training fallback")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.9, protocol["budget"]["maximum_device_memory_MiB"] * 1024**2 / total), device)


def prepare_subset(cache, subset, frozen, protocol, model_protocol, device):
    """Discard outer rows before fitting scalers/processor or constructing bundles."""
    import numpy as np
    data, provenance = frozen.load_cache(cache, subset)
    require(provenance["cache_sha256"] == protocol["expected_cache_sha256"][subset], "unregistered cache")
    units = sorted(map(int, np.unique(data["unit"])))
    require(len(units) == 6, "six development engines required")
    outer, validation = units[:2]
    train = units[2:]
    eligible = data["unit"] != outer
    screen = {key: value[eligible] for key, value in data.items()}
    del data
    require(outer not in set(map(int, screen["unit"])), "outer was not excluded")
    scalers = frozen.fit_scalers(screen, train, 125.)
    cache_identity = {key: provenance[key] for key in ("cache_sha256", "source_sha256")}
    processor = fit_processor(screen, train, scalers, cache_identity)
    base = frozen.bundle(screen, train, scalers, model_protocol, device)
    require(set(map(int, base["unit"])) == set(train), "training bundle includes a nontraining engine")
    return {"data": screen, "provenance": provenance, "outer": outer, "validation": validation,
            "train": train, "scalers": scalers, "processor": processor, "base": base}


def budget(deadline):
    require(time.monotonic() < deadline, "fixed eight-hour shared GPU wall budget exhausted")


def fit_cell(prepared, subset, seed, family, root, frozen, protocol, model_protocol, device, deadline, paired_reference=None):
    import numpy as np
    import torch
    budget(deadline)
    spec = protocol["families"][family]
    cell = f"{subset}_inner_screen_out{prepared['outer']}_val{prepared['validation']}_{family}_s{seed}"
    folder = root / cell; folder.mkdir(exist_ok=False)
    scalers, processor = prepared["scalers"], prepared["processor"]
    config = {"cell": cell, "phase": "inner_screen", "subset": subset, "seed": seed, "family": family,
              "conditioning": spec["conditioning"], "auxiliary_weight": spec["auxiliary_weight"],
              "excluded_outer_unit": prepared["outer"], "validation_unit": prepared["validation"],
              "train_units": prepared["train"], "protocol": protocol, "frozen_model_protocol": model_protocol,
              "execution_sha256": execution_hash(), "frozen_execution_sha256": FROZEN_EXECUTION,
              "cache_sha256": prepared["provenance"]["cache_sha256"], "source_sha256": prepared["provenance"]["source_sha256"],
              "processor_sha256": processor["processor_sha256"], "scalers_sha256": identity(scalers)}
    config["config_sha256"] = identity(config)
    dump(folder / "CLAIM.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})
    dump(folder / "config.json", config); dump(folder / "scalers.json", scalers); dump(folder / "processor.json", processor)
    training = apply_bundle(prepared["base"], processor, spec["conditioning"])
    frozen.seed_everything(seed)
    model = frozen.Model("raw", model_protocol, seed).to(device)
    initial = frozen.state_hash(model)
    pairing = {"initial_state_sha256": initial, "encoder_initial_sha256": frozen.state_hash(model.encoder),
               "head_initial_sha256": frozen.state_hash(model.head), "processor_sha256": processor["processor_sha256"],
               "scalers_sha256": identity(scalers)}
    if paired_reference is not None:
        require(all(value == paired_reference["pairing"][key] for key, value in pairing.items()),
                "four-family initialization/scaler/processor differs before training")
    parameters = sum(p.numel() for p in model.parameters())
    torch.save({"model": model.state_dict(), "config": config, "scalers": scalers, "processor": processor,
                "pairing": pairing}, folder / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator(device="cpu").manual_seed(seed + 20000)
    frozen.seed_everything(seed + 30000)
    started = time.monotonic(); updates = 0; order_hash = hashlib.sha256(); block_hash = hashlib.sha256(); trace = []
    torch.cuda.reset_peak_memory_stats(device)
    with open(folder / "step_trace.csv", "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "update", "n", "block", "query_sha256", "task", "auxiliary", "weighted_auxiliary", "total", "gradient_norm", "seconds"])
        for epoch in range(200):
            model.train(); tasks = []; auxiliaries = []; block = epoch % 4
            order = torch.randperm(len(training["y"]), generator=generator).numpy().astype(np.int64, copy=False)
            encoded_order = order.tobytes()
            epoch_hash = hashlib.sha256(encoded_order).hexdigest()
            if paired_reference is not None:
                require(epoch_hash == paired_reference["epoch_order_sha256s"][epoch], "paired epoch order changed before optimizer updates")
            order_hash.update(canonical({"epoch": epoch + 1, "n": len(order)})); order_hash.update(encoded_order)
            block_hash.update(canonical([epoch + 1, block]))
            for batch_no, lo in enumerate(range(0, len(order), 64)):
                budget(deadline)
                selected = order[lo:lo + 64]
                query = torch.as_tensor(selected, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=False):
                    pred, aux, _ = model(training, query, block=block)
                    task = (pred - training["target"][query]).square().mean()
                    loss = task + spec["auxiliary_weight"] * aux  # aux0 retains identical forward/RNG path.
                require(bool(torch.isfinite(loss)), "nonfinite paired-screen loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step(); updates += 1
                require(torch.cuda.max_memory_allocated(device) / 1024**2 <= 12000, "fixed GPU memory budget exceeded")
                t, a, total = float(task.detach()), float(aux.detach()), float(loss.detach())
                tasks.append(t); auxiliaries.append(a)
                writer.writerow([epoch + 1, batch_no, updates, len(selected), block,
                                 hashlib.sha256(selected.tobytes()).hexdigest(), t, a, spec["auxiliary_weight"] * a,
                                 total, float(norm), time.monotonic() - started])
            trace.append({"epoch": epoch + 1, "block": block, "order_sha256": epoch_hash,
                          "updates": updates, "task": float(np.mean(tasks)), "auxiliary": float(np.mean(auxiliaries)),
                          "seconds": time.monotonic() - started})
            f.flush()
    torch.cuda.synchronize(device)
    seconds = time.monotonic() - started
    final = frozen.state_hash(model)
    expected_updates = 200 * math.ceil(len(training["y"]) / 64)
    require(updates == expected_updates and final != initial, "missing updates/unchanged weights")
    require(optimizer.state_dict()["state"] and all(float(state["step"]) == updates for state in optimizer.state_dict()["state"].values()), "optimizer step state mismatch")
    pairing.update(optimizer_updates=updates, epoch_order_sha256=order_hash.hexdigest(), block_schedule_sha256=block_hash.hexdigest())
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                "scalers": scalers, "processor": processor, "pairing": pairing, "epochs_completed": 200,
                "optimizer_updates": updates, "initial_state_sha256": initial, "final_state_sha256": final,
                "outer_evaluated": False, "validation_evaluations_before_checkpoint": 0}, folder / "checkpoint.pt")
    dump(folder / "training_trace.json", trace)
    # The only real validation bundle/prediction is constructed after final fitting.
    heldout = frozen.bundle(prepared["data"], [prepared["validation"]], scalers, model_protocol, device)
    require(set(map(int, heldout["unit"])) == {prepared["validation"]}, "wrong inner validation engine")
    heldout = apply_bundle(heldout, processor, spec["conditioning"])
    unbounded, _ = frozen.predict(model, heldout, scalers, 64)
    prediction = np.maximum(unbounded, 0.)
    require(bool(np.isfinite(prediction).all()), "nonfinite final inner prediction")
    scores = {"raw": frozen.metrics(heldout["y"], prediction),
              "clip125": frozen.metrics(np.minimum(heldout["y"], 125.), prediction)}
    require(torch.cuda.max_memory_allocated(device) / 1024**2 <= 12000, "final validation exceeds fixed GPU memory budget")
    with open(folder / "prediction.csv", "x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["unit", "cycle", "y_true_raw", "y_pred", "y_pred_unbounded", "seed", "family"])
        writer.writerows(zip(heldout["unit"], heldout["cycle"], heldout["y"], prediction, unbounded,
                             [seed] * len(prediction), [family] * len(prediction)))
    result = {"cell": cell, "phase": "inner_screen", "subset": subset, "seed": seed, "family": family,
              "config_sha256": config["config_sha256"], "execution_sha256": config["execution_sha256"], "pairing": pairing,
              "epoch_order_sha256s": [r["order_sha256"] for r in trace],
              "parameters": parameters, "fixed_processor_coefficients": 210, "fixed_empirical_scales": 14,
              "additional_neural_parameters": 0, "epochs_completed": 200, "optimizer_updates": updates,
              "initial_state_sha256": initial, "final_state_sha256": final, "training_seconds": seconds,
              "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 1024**2,
              "checkpoint_sha256": digest(folder / "checkpoint.pt"), "scores": scores,
              "final_inner_prediction_sweeps": 1, "midtraining_validation_evaluations": 0,
              "official_TEST_read": False, "outer_evaluated": False}
    budget(deadline)
    dump(folder / "metrics.json", result)
    dump(folder / "SUCCESS.json", {"cell": cell, "files_sha256": {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    print(json.dumps({"completed_cell": cell, "training_seconds": seconds, "optimizer_updates": updates}), flush=True)
    del model, optimizer, training, heldout
    torch.cuda.empty_cache()
    return result


def summarize(results, protocol):
    import numpy as np
    require(len(results) == 48 and len({(r["subset"], r["seed"], r["family"]) for r in results}) == 48, "full48 required before screening gate")
    rows, gains = [], {}
    for subset in protocol["subsets"]:
        gains[subset] = {}
        for family in protocol["families"]:
            selected = [r for r in results if r["subset"] == subset and r["family"] == family]
            require(sorted(r["seed"] for r in selected) == protocol["seeds"], "incomplete three-seed family")
            values = [r["scores"]["raw"]["rmse"] for r in selected]
            row = {"subset": subset, "family": family, "rmse_mean": float(np.mean(values)), "rmse_sd": float(np.std(values, ddof=1))}
            for key in ("nasa_sum", "nasa_mean"):
                v = [r["scores"]["raw"][key] for r in selected]
                row[key + "_mean"] = float(np.mean(v)); row[key + "_sd"] = float(np.std(v, ddof=1))
            rows.append(row)
        by_family = {r["family"]: r for r in rows if r["subset"] == subset}
        for suffix in ("aux1", "aux0"):
            raw = by_family["raw_" + suffix]["rmse_mean"]
            gains[subset][suffix] = 1. - by_family["resid_" + suffix]["rmse_mean"] / raw if raw > 0 else None
    identifiable = all(v is not None for s in gains.values() for v in s.values())
    same = [s for s in protocol["subsets"] if all(gains[s][a] is not None and gains[s][a] >= .02 for a in ("aux1", "aux0"))]
    no_worse = identifiable and all(v >= -.02 for s in gains.values() for v in s.values())
    return {"status": "COMPLETE", "phase": "inner_screen", "training_runs": 48, "families": rows,
            "per_seed": [{"subset": r["subset"], "family": r["family"], "seed": r["seed"],
                          **r["scores"]["raw"]} for r in results],
            "gain_ratio_of_three_seed_means": gains, "same_qualifying_subsets_both_aux": same,
            "all_subset_aux_no_worse_than_2pct": no_worse, "gate": "UNIDENTIFIABLE" if not identifiable else "MET" if len(same) >= 3 and no_worse else "NOT_MET",
            "interpretation": protocol["purpose"], "no_new_module_or_outer_confirmation": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "output", "preflight"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    require(not args.output.exists(), "output exists; do not overwrite or silently resume")
    frozen, protocol, model_protocol = load_frozen()
    evidence = read(args.preflight)
    require(evidence.get("status") == "PASS" and evidence.get("optimizer_updates") == 0 and evidence.get("training_admission") is True
            and evidence.get("execution_sha256") == execution_hash() and evidence.get("device") == args.device,
            "missing matching zero-optimizer hardware admission")
    configure_device(args.device, protocol)
    import torch
    hardware = evidence.get("hardware", {})
    records = hardware.get("records", [])
    expected = {(s, f) for s in protocol["subsets"] for f in protocol["families"]}
    require(hardware.get("status") == "PASS" and len(records) == 16 and
            {(r["subset"], r["family"]) for r in records} == expected and
            all(r["optimizer_updates"] == 0 and 0 < r["peak_allocated_MiB"] <= 12000 for r in records) and
            evidence.get("checks_source_sha256") == digest(HERE/"checks.py"),
            "incomplete or changed hardware admission evidence")
    require(hardware.get("device_name") == torch.cuda.get_device_name(args.device) and
            hardware.get("torch") == torch.__version__ and hardware.get("cuda") == torch.version.cuda and
            0 < hardware.get("projected_48_training_hours_with_25pct_margin", float("inf")) < 8,
            "hardware/runtime identity or measured wall budget differs from admission")
    args.output.mkdir(parents=True, exist_ok=False); cells = args.output / "cells"; cells.mkdir()
    started = time.monotonic(); deadline = started + 28800
    dump(args.output / "LAUNCH.json", {"protocol": protocol, "execution_sha256": execution_hash(), "pid": os.getpid(),
         "started_utc": datetime.now(timezone.utc).isoformat(), "preflight_sha256": digest(args.preflight),
         "production_files_sha256": {n: digest(HERE / n) for n in PRODUCTION}, "frozen_files_sha256": FROZEN_HASHES})
    results = []
    try:
        for subset in protocol["subsets"]:
            prepared = prepare_subset(args.cache, subset, frozen, protocol, model_protocol, args.device)
            for seed in protocol["seeds"]:
                paired = []
                for family in protocol["families"]:
                    result = fit_cell(prepared, subset, seed, family, cells, frozen, protocol, model_protocol, args.device, deadline,
                                      paired_reference=paired[0] if paired else None)
                    paired.append(result); results.append(result)
                require(all(r["pairing"] == paired[0]["pairing"] for r in paired), "four-family initial/order/block/update/scaler/processor pairing failed")
                dump(args.output / f"PAIRING_{subset}_s{seed}.json", {"status": "PASS", "families": list(protocol["families"]), "pairing": paired[0]["pairing"]})
            del prepared
        summary = summarize(results, protocol); summary["elapsed_shared_gpu_wall_seconds"] = time.monotonic() - started
        budget(deadline)
        dump(args.output / "SUMMARY.json", summary)
        dump(args.output / "SUCCESS.json", {"training_runs": 48, "execution_sha256": execution_hash(),
             "files_sha256": {str(f.relative_to(args.output)).replace('\\', '/'): digest(f) for f in sorted(args.output.rglob("*")) if f.is_file()}})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        dump(args.output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc), "completed_cells": len(results),
             "elapsed_seconds": time.monotonic() - started, "full_gate": False, "execution_sha256": execution_hash(),
             "failure_branch": protocol["failure_branch"]})
        raise


if __name__ == "__main__":
    main()
