"""Software contracts. Default: synthetic NumPy only, no real data/Torch/training.

Root may explicitly request --admission-cache for actual GPU forward/backward
contracts with ZERO optimizer updates. Only that matching evidence admits run.py.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
from .preprocess import (apply_bundle, apply_numpy, fit_processor, global_standardize,
                         identity, require, restore_numpy, verify_processor)
from .run import (configure_device, digest, dump, execution_hash, load_frozen,
                  prepare_subset, summarize, verify_frozen)


def numpy_contracts():
    protocol = verify_frozen()
    rng = np.random.default_rng(641)
    raw = rng.normal(size=(12, 512, 18)).astype(np.float32)
    unit = np.repeat(np.arange(1, 7), 2)
    train = [3, 4, 5, 6]
    class Guard(dict):
        def __getitem__(self, key):
            require(key in ("raw", "unit"), "processor accessed forbidden data: " + key)
            return super().__getitem__(key)
    data = Guard(raw=raw, unit=unit, y=np.full(12, np.nan))
    def scalers(x):
        selected = x[np.isin(unit, train)].astype(np.float64)
        return {"raw_mu": selected.mean((0, 1)).tolist(), "raw_sd": np.maximum(selected.std((0, 1)), 1e-6).tolist(),
                "fitted_units": train, "target_mu": 0., "target_sd": 1., "age_q95": 4.}
    scale = scalers(raw)
    np_state, py_state = copy.deepcopy(np.random.get_state()), copy.deepcopy(random.getstate())
    processor = fit_processor(data, train, scale)
    standardized = global_standardize(raw, scale)
    independent_coordinates = (raw.astype(np.float32) - np.asarray(scale["raw_mu"], np.float32)) / np.asarray(scale["raw_sd"], np.float32)
    require(standardized.dtype == np.float32 and np.array_equal(standardized, independent_coordinates), "frozen FP32 coordinate order differs")
    conditioned = apply_numpy(standardized, processor)
    restored = restore_numpy(conditioned, processor)
    np_after = np.random.get_state()
    require(np_state[0] == np_after[0] and np.array_equal(np_state[1], np_after[1]) and
            np_state[2:] == np_after[2:] and py_state == random.getstate(), "fit/apply consumed RNG")
    altered = raw.copy(); altered[unit <= 2] = 1e8
    require(processor == fit_processor(Guard(raw=altered, unit=unit, y=np.arange(12)), train, scalers(altered)),
            "excluded outer/validation values changed fitted processor")
    require(np.array_equal(conditioned[..., :4], standardized[..., :4]), "W changed")
    error = float(np.max(np.abs(restored - standardized.astype(np.float64))))
    require(error <= 1e-5, "fixed conditioning inversion failed")
    initial_processor_hash = identity(processor)
    for family, spec in protocol["families"].items():
        output = apply_numpy(standardized, processor, spec["conditioning"])
        require(identity(processor) == initial_processor_hash, "family mutated shared processor")
        if spec["conditioning"] == "raw":
            require(np.array_equal(output, standardized), "raw family input changed")
    constant = np.zeros_like(raw)
    cp = fit_processor(Guard(raw=constant, unit=unit), train, scalers(constant))
    coef, residual_scale = verify_processor(cp)
    require(bool(np.isfinite(coef).all()) and bool((residual_scale == 1e-6).all()), "constant-input ridge/floor failure")
    damaged = copy.deepcopy(processor); damaged["coefficients"][0][0] += 1
    try:
        verify_processor(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("tampered processor accepted")
    # Explicitly synthetic gate fixtures, never retained as experiment cells.
    fixture = [{"subset": s, "family": f, "seed": seed,
                "scores": {"raw": {"rmse": 9.7 if f.startswith("resid") else 10., "nasa_sum": 1., "nasa_mean": 1.}}}
               for s in protocol["subsets"] for f in protocol["families"] for seed in protocol["seeds"]]
    require(summarize(fixture, protocol)["gate"] == "MET", "fixed full-matrix gate fixture failed")
    for r in fixture:
        if (r["subset"], r["family"]) in (("DS07", "resid_aux1"), ("DS06", "resid_aux0")):
            r["scores"]["raw"]["rmse"] = 10.
    require(summarize(fixture, protocol)["gate"] == "NOT_MET", "different qualifying subsets were incorrectly merged")
    try:
        summarize(fixture[:-1], protocol)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial matrix entered screening gate")
    return {"status": "PASS", "synthetic_only": True, "real_data_read": False,
            "checks": ["frozen_five_files", "fit_reads_only_raw_unit", "excluded_val_and_outer_invariance",
                       "FP32_bundle_coordinate_fit", "identical_processor_all_four_families", "raw_identity",
                       "W_byte_identity", "inverse_recovery", "no_NumPy_or_Python_RNG_consumption",
                       "constant_ridge_scale_finite", "processor_hash_tamper_rejected",
                       "same_subset_intersection_gate", "partial_matrix_gate_rejected"],
            "inverse_max_error_standardized_units": error, "fixed_coefficients": 210, "empirical_scales": 14,
            "Torch_equivalence_and_metadata_identity": "NOT_RUN", "initial_and_batch_pairing": "enforced_in_training_controller_not_executed"}


def hardware_admission(cache, device):
    """Root-only explicit admission: full train bundles, original model, no optimizer."""
    import torch
    started = time.monotonic()
    frozen, protocol, model_protocol = load_frozen()
    configure_device(device, protocol)
    records = []
    for subset in protocol["subsets"]:
        prepared = prepare_subset(cache, subset, frozen, protocol, model_protocol, device)
        initial_reference = None
        torch.cuda.reset_peak_memory_stats(device)
        for family, spec in protocol["families"].items():
            cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
            training = apply_bundle(prepared["base"], prepared["processor"], spec["conditioning"])
            require(torch.equal(cpu_rng, torch.random.get_rng_state()) and
                    torch.equal(cuda_rng, torch.cuda.get_rng_state(device)), "processor application consumed Torch RNG")
            expected = apply_numpy(prepared["base"]["raw"].cpu().numpy(), prepared["processor"], spec["conditioning"])
            difference = float(np.max(np.abs(training["raw"].cpu().numpy().astype(np.float64) - expected.astype(np.float64))))
            require(difference <= 2e-5, "Torch/NumPy coordinate mismatch")
            require(all(training[k] is value for k, value in prepared["base"].items() if k != "raw"), "age/coverage/window/label metadata changed")
            frozen.seed_everything(101)
            model = frozen.Model("raw", model_protocol, 101).to(device)
            initial = frozen.state_hash(model)
            reference = {"full": initial, "encoder": frozen.state_hash(model.encoder), "head": frozen.state_hash(model.head)}
            if initial_reference is None:
                initial_reference = reference
            require(reference == initial_reference, "four-family initial state mismatch")
            frozen.seed_everything(30101)
            query = torch.arange(min(64, len(training["y"])), dtype=torch.long, device=device)
            model.train()
            torch.cuda.synchronize(device)
            batch_started = time.monotonic()
            with torch.autocast(device_type="cuda", enabled=False):
                prediction, aux, _ = model(training, query, block=0)
                task = (prediction - training["target"][query]).square().mean()
                loss = task + spec["auxiliary_weight"] * aux
            require(bool(torch.isfinite(loss)), "nonfinite admission loss")
            loss.backward()
            torch.cuda.synchronize(device)
            batch_seconds = time.monotonic() - batch_started
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            require(grads and all(bool(torch.isfinite(g).all()) for g in grads) and any(bool(g.ne(0).any()) for g in grads), "invalid admission gradients")
            require(frozen.state_hash(model) == initial, "weights changed without optimizer in admission")
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            require(peak <= 12000, "admission exceeds memory budget")
            records.append({"subset": subset, "family": family, "initial": reference,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "processor_sha256": prepared["processor"]["processor_sha256"],
                            "scalers_sha256": identity(prepared["scalers"]), "Torch_NumPy_max_difference": difference,
                            "peak_allocated_MiB": peak, "gradient_tensors": len(grads),
                            "batch_forward_backward_seconds": batch_seconds,
                            "training_batches_per_epoch": (len(training["y"])+63)//64,
                            "outer_bundle_constructed": False, "optimizer_updates": 0})
            del model, training
            torch.cuda.empty_cache()
        del prepared
    hours = sum(r["batch_forward_backward_seconds"]*r["training_batches_per_epoch"]*200*3 for r in records)*1.25/3600
    require(hours < protocol["budget"]["one_process_shared_gpu_wall_seconds"]/3600,
            "measured complete 48-cell projection exceeds fixed budget")
    return {"status": "PASS", "records": records, "optimizer_updates": 0,
            "projected_48_training_hours_with_25pct_margin": hours,
            "elapsed_seconds": time.monotonic() - started,
            "torch": torch.__version__, "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(device)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--admission-cache", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if args.output:
        require(not args.output.exists(), "check evidence already exists")
    result = {"status": "PASS", "optimizer_updates": 0, "training_admission": False,
              "execution_sha256": execution_hash(), "checks_source_sha256": digest(__file__),
              "numpy_contracts": numpy_contracts(), "hardware": {"status": "NOT_RUN"}}
    if args.admission_cache:
        result["hardware"] = hardware_admission(args.admission_cache, args.device)
        result.update(training_admission=True, device=args.device)
    if args.output:
        dump(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
