"""Full-author AIDGN migration with nested engine window selection and no TEST.

The original positive output head requires fixed physical target scaling.
All five source windows are attempted; there is no surrogate network.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from paper_b.ncmapss_response.run import (bundle, canonical, execution_hash as shared_execution,
                                         fit_scalers, load_cache, metrics, seed_everything, state_hash)
from paper_b.ncmapss_response.prepare import digest, dump_new
from .aidgn_adapter import AIDGNAdapter, pack_window

HERE = Path(__file__).parent


def execution_hash(author_file):
    h = hashlib.sha256()
    for name in ("__init__.py", "aidgn_adapter.py", "run_aidgn.py", "PROTOCOL.json"):
        h.update(name.encode()); h.update((HERE/name).read_bytes())
    h.update(shared_execution().encode()); h.update(digest(author_file).encode())
    return h.hexdigest()


def input_scalers(data, units):
    scalers = fit_scalers(data, units, 125.)
    physical = data["raw"][np.isin(data["unit"], units)].astype(np.float64)
    lower = physical.min(axis=(0, 1)); upper = physical.max(axis=(0, 1))
    ranges = upper-lower
    constant = ranges < 10*np.finfo(np.float64).eps
    handled = np.where(constant, 1., ranges)
    scalers["raw_mu"] = (lower+handled/2).tolist()
    scalers["raw_sd"] = (handled/2).tolist()
    scalers["physical_transform"] = "training-only MinMax[-1,1]; raw_mu is offset, raw_sd is half handled range; constants map -1"
    scalers["constant_columns"] = np.flatnonzero(constant).tolist()
    # The original abs head cannot model negative z-scored labels. These fixed
    # coordinates preserve physical labels and keep the complete author head.
    scalers["target_mu"] = 0.; scalers["target_sd"] = 125.
    return scalers


def verify_completed(folder, config):
    if not (folder/"SUCCESS.json").exists():
        return False
    seal = json.loads((folder/"SUCCESS.json").read_text())
    for name, sha in seal["files_sha256"].items():
        if digest(folder/name) != sha:
            raise AssertionError("modified sealed baseline artifact")
    if json.loads((folder/"config.json").read_text()) != config:
        raise AssertionError("different baseline run identity")
    return True


def fit_one(data, provenance, train_units, validation_unit, subset, seed, window,
            phase, outer_unit, protocol, author_file, root, device):
    history, points = window
    cell = f"{subset}_{phase}_out{outer_unit}_val{validation_unit}_h{history}_p{points}_s{seed}"
    folder = root/cell
    config = {"protocol": protocol, "cell": cell, "phase": phase, "subset": subset,
              "seed": seed, "history": history, "points_per_flight": points,
              "train_units": list(map(int, train_units)), "validation_unit": int(validation_unit),
              "outer_heldout_unit": int(outer_unit), "source_sha256": provenance["source_sha256"],
              "cache_sha256": provenance["cache_sha256"], "author_file_sha256": digest(author_file),
              "execution_sha256": execution_hash(author_file)}
    config["config_sha256"] = hashlib.sha256(canonical(config)).hexdigest()
    if verify_completed(folder, config):
        return json.loads((folder/"metrics.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    dump_new(folder/"CLAIM.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})
    dump_new(folder/"config.json", config)
    scalers = input_scalers(data, train_units); dump_new(folder/"scalers.json", scalers)
    training = bundle(data, train_units, scalers,
                      {"span": 70, "prior_lag_flights": 5, "support_radius": .75, "label_cap": 125.}, device)
    seed_everything(seed)
    model = AIDGNAdapter(author_file, history=history, points=points).to(device)
    initial = state_hash(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    torch.save({"model": model.state_dict(), "config": config, "scalers": scalers,
                "initial_state_sha256": initial, "author_provenance": model.provenance}, folder/"initial.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rates"][subset], weight_decay=0.)
    generator = torch.Generator(device="cpu").manual_seed(seed+20000)
    seed_everything(seed+30000)
    steps = 0; trace = []; started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(protocol["epochs"]):
        model.train(); losses = []
        order = torch.randperm(len(training["y"]), generator=generator).tolist()
        for start in range(0, len(order), protocol["batch"]):
            query = torch.tensor(order[start:start+protocol["batch"]], dtype=torch.long, device=device)
            x, age = pack_window(training, query, history, points)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, age)
            loss = (pred-training["target"][query]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite author-model loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step(); steps += 1; losses.append(float(loss.detach()))
            if torch.cuda.max_memory_allocated(device)/1024**2 > protocol["maximum_device_memory_MiB"]:
                raise MemoryError("author-model device budget exceeded")
        if epoch == 0 or (epoch+1) % 20 == 0:
            trace.append({"epoch": epoch+1, "normalized_mse": float(np.mean(losses)),
                          "gradient_norm_last": float(norm), "seconds": time.monotonic()-started})
        if time.monotonic()-started > protocol["maximum_cell_seconds"]:
            raise TimeoutError("author-model single-cell budget exceeded")
    torch.cuda.synchronize(device)
    seconds = time.monotonic()-started; final = state_hash(model)
    if final == initial:
        raise AssertionError("unchanged author weights after claimed training")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "scalers": scalers, "author_provenance": model.provenance,
                "epochs_completed": protocol["epochs"], "optimizer_updates": steps,
                "initial_state_sha256": initial, "final_state_sha256": final}, folder/"checkpoint.pt")
    dump_new(folder/"training_trace.json", trace)
    model.eval(); views = ("clean",) if phase == "inner" else ("clean", "density", "unpaired")
    results = {}
    for view in views:
        heldout = bundle(data, [validation_unit], scalers,
                         {"span": 70, "prior_lag_flights": 5, "support_radius": .75, "label_cap": 125.}, device, view)
        predictions = []
        with torch.no_grad():
            for lo in range(0, len(heldout["y"]), protocol["batch"]):
                q = torch.arange(lo, min(lo+protocol["batch"], len(heldout["y"])), device=device)
                x, a = pack_window(heldout, q, history, points)
                predictions.append(model(x, a).cpu().numpy())
        raw_prediction = np.concatenate(predictions).astype(np.float64)*125.
        prediction = np.maximum(raw_prediction, 0.)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("nonfinite author-model prediction")
        results[view] = {"raw": metrics(heldout["y"], prediction),
                         "clip125": metrics(np.minimum(heldout["y"], 125.), prediction)}
        with open(folder/f"prediction_{view}.csv", "x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f); writer.writerow(["unit", "cycle", "y_true_raw", "y_pred", "y_pred_unbounded", "seed", "view"])
            writer.writerows(zip(heldout["unit"], heldout["cycle"], heldout["y"], prediction,
                                 raw_prediction, [seed]*len(prediction), [view]*len(prediction)))
    result = {"cell": cell, "phase": phase, "subset": subset, "seed": seed, "history": history,
              "points_per_flight": points, "outer_heldout_unit": outer_unit, "validation_unit": validation_unit,
              "training_seconds": seconds, "parameters": parameter_count, "optimizer_updates": steps,
              "initial_state_sha256": initial, "final_state_sha256": final,
              "peak_allocated_MiB": torch.cuda.max_memory_allocated(device)/1024**2,
              "config_sha256": config["config_sha256"], "execution_sha256": config["execution_sha256"],
              "author_provenance": model.provenance, "checkpoint_sha256": digest(folder/"checkpoint.pt"),
              "official_test_read": False, "scores": results}
    dump_new(folder/"metrics.json", result)
    seal = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}
    dump_new(folder/"SUCCESS.json", {"cell": cell, "files_sha256": seal})
    print(json.dumps({"cell": cell, "parameters": parameter_count, "optimizer_updates": steps,
                      "training_seconds": seconds, "rmse": results["clean"]["raw"]["rmse"]}), flush=True)
    del model, optimizer, training, heldout
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--author-file", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True); p.add_argument("--subsets", nargs="+")
    p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    protocol = json.loads((HERE/"PROTOCOL.json").read_text())
    if shared_execution() != protocol["shared_data_execution_sha256"]:
        raise AssertionError("frozen data helpers changed")
    evidence = json.loads(args.preflight.read_text())
    if evidence["status"] != "PASS" or evidence["optimizer_updates"] != 0 or evidence["execution_sha256"] != execution_hash(args.author_file):
        raise AssertionError("missing exact-source hardware preflight")
    if args.subsets and not set(args.subsets).issubset(protocol["subsets"]):
        raise ValueError("unregistered subset")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    memory = torch.cuda.get_device_properties(args.device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.9, protocol["maximum_device_memory_MiB"]*1024**2/memory), args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    dump_new(args.output/"LAUNCH.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
                                       "protocol": protocol, "execution_sha256": execution_hash(args.author_file)})
    cells = args.output/"cells"; cells.mkdir(); selections = args.output/"selections"; selections.mkdir()
    start = time.monotonic()
    try:
        for subset in args.subsets or protocol["subsets"]:
            data, provenance = load_cache(args.cache, subset)
            units = sorted(set(map(int, data["unit"])))
            for pos, outer in enumerate(units):
                inner = units[(pos+1) % len(units)]
                inner_train = [u for u in units if u not in (outer, inner)]
                choices = []
                for window in protocol["window_catalog"]:
                    result = fit_one(data, provenance, inner_train, inner, subset, protocol["inner_selection_seed"],
                                     window, "inner", outer, protocol, args.author_file, cells, args.device)
                    choices.append({"window": window, "rmse": result["scores"]["clean"]["raw"]["rmse"],
                                    "cell": result["cell"], "config_sha256": result["config_sha256"]})
                chosen = min(choices, key=lambda c: (c["rmse"], c["window"][0]*c["window"][1], *c["window"]))
                dump_new(selections/f"{subset}_out{outer}.json", {"outer_heldout_unit": outer,
                    "inner_validation_unit": inner, "inner_training_units": inner_train,
                    "choices": choices, "selected_window": chosen["window"], "selection_metric": "inner_raw_RMSE"})
                outer_train = [u for u in units if u != outer]
                for seed in protocol["seeds"]:
                    fit_one(data, provenance, outer_train, outer, subset, seed, chosen["window"], "outer", outer,
                            protocol, args.author_file, cells, args.device)
                if time.monotonic()-start > protocol["maximum_shared_gpu_wall_hours"]*3600:
                    raise TimeoutError("whole public-model training budget exceeded")
        dump_new(args.output/"REQUESTED_SUBSETS_COMPLETE.json", {"subsets": args.subsets or protocol["subsets"],
                  "execution_sha256": execution_hash(args.author_file), "elapsed_seconds": time.monotonic()-start})
    except BaseException as e:
        dump_new(args.output/"FAILED.json", {"type": type(e).__name__, "message": str(e),
                                            "execution_sha256": execution_hash(args.author_file)})
        raise


if __name__ == "__main__":
    main()
