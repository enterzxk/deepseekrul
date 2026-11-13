"""Reload sealed checkpoints and independently regenerate development predictions.

Run this file directly with --runtime pointing to the immutable training tree.
The audit never changes a run, performs an optimizer update, or opens HDF5/TEST.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    import numpy as np
    import torch
    from paper_b.ncmapss_response.run import (Model, bundle, execution_hash, fit_scalers,
                                             load_cache, predict, state_hash)
    from paper_b.ncmapss_response.prepare import digest, dump_new

    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    protocol = json.loads((args.runtime/"paper_b/ncmapss_response/PROTOCOL.json").read_text())
    execution = execution_hash()
    manifest = json.loads((args.runtime/"GIT_SOURCE_MANIFEST.json").read_text())
    if not all(digest(args.runtime/name) == sha for name, sha in manifest["files_sha256"].items()):
        raise AssertionError("training runtime was modified")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = sorted(args.runs.glob("*/SUCCESS.json"))
    if args.limit:
        selected = selected[:args.limit]
    rows = []; cache = {}; started = time.monotonic()
    for seal_path in selected:
        folder = seal_path.parent
        seal = json.loads(seal_path.read_text())
        if not all(digest(folder/name) == sha for name, sha in seal["files_sha256"].items()):
            raise AssertionError(f"modified sealed artifacts {folder.name}")
        config = json.loads((folder/"config.json").read_text())
        if config["execution_sha256"] != execution or config["protocol"] != protocol:
            raise AssertionError("different registered training execution")
        subset = folder.name.split("_")[0]
        if subset not in cache:
            cache[subset] = load_cache(args.cache, subset)
        data, provenance = cache[subset]
        if (config["source_sha256"] != provenance["source_sha256"] or
                config["cache_sha256"] != provenance["cache_sha256"]):
            raise AssertionError("different source data")
        expected_scaler = fit_scalers(data, config["train_units"], protocol["label_cap"])
        checkpoint = torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["scalers"] != expected_scaler:
            raise AssertionError("checkpoint identity/scaler mismatch")
        model = Model(config["mode"], protocol, config["seed"]).to("cuda:0")
        model.load_state_dict(checkpoint["model"], strict=True)
        if state_hash(model) != checkpoint["final_state_sha256"]:
            raise AssertionError("checkpoint tensor hash mismatch")
        maximum = 0.; exact = True
        for view in ("clean", "density", "unpaired"):
            heldout = bundle(data, [config["held_out_unit"]], expected_scaler, protocol, "cuda:0", view)
            unbounded, _ = predict(model, heldout, expected_scaler, protocol["batch"])
            with open(folder/f"prediction_{view}.csv", newline="", encoding="utf-8") as f:
                records = list(csv.DictReader(f))
            saved = np.array([float(r["y_pred_unbounded"]) for r in records], np.float64)
            if saved.shape != unbounded.shape:
                raise AssertionError("prediction query shape mismatch")
            difference = float(np.max(np.abs(saved-unbounded)))
            maximum = max(maximum, difference); exact = exact and np.array_equal(saved, unbounded)
            if difference > 1e-5:
                raise AssertionError(f"checkpoint-prediction mismatch {folder.name}/{view}: {difference}")
        row = {"cell": folder.name, "three_views_replayed": True,
               "bitwise_equal": bool(exact), "maximum_absolute_difference_cycles": maximum,
               "checkpoint_sha256": digest(folder/"checkpoint.pt"), "seal_sha256": digest(seal_path)}
        dump_new(args.output/(folder.name+".json"), row); rows.append(row)
        print(json.dumps(row), flush=True)
        del model, checkpoint, heldout
        torch.cuda.empty_cache()
    expected = protocol["training_runs"]
    report = {"status": "PASS" if len(rows) == expected else "PARTIAL",
              "n_replayed": len(rows), "expected": expected, "rows": rows,
              "execution_sha256": execution, "source_git_commit": manifest["git_commit"],
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "optimizer_updates": 0, "official_test_read": False,
              "tolerance_cycles": 1e-5, "seconds": time.monotonic()-started}
    dump_new(args.output/"REPLAY_AUDIT.json", report)


if __name__ == "__main__":
    main()
