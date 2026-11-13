"""Measure a training-fold production batch without an optimizer or heldout forward."""
import argparse
import json
from pathlib import Path
import time

import torch

from .prepare import dump_new
from .run import Model, bundle, execution_hash, fit_scalers, load_cache


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    protocol = json.loads((Path(__file__).parent/"PROTOCOL.json").read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data, manifest = load_cache(args.cache, "DS02")
    units = sorted(map(int, set(data["unit"])))
    training_units = units[1:]
    scaler = fit_scalers(data, training_units, protocol["label_cap"])
    training = bundle(data, training_units, scaler, protocol, "cuda:0")
    n_batches = (len(training["y"])+protocol["batch"]-1)//protocol["batch"]
    query = torch.arange(protocol["batch"], device="cuda:0")
    rows = []
    for mode in protocol["modes"]:
        model = Model(mode, protocol, 101).cuda().train()
        elapsed = []
        torch.cuda.reset_peak_memory_stats()
        for repetition in range(4):
            model.zero_grad(set_to_none=True); torch.cuda.synchronize(); start = time.monotonic()
            pred, aux, _ = model(training, query, block=repetition)
            loss = (pred-training["target"][query]).square().mean()+protocol["auxiliary_weight"]*aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"], error_if_nonfinite=True)
            torch.cuda.synchronize(); elapsed.append(time.monotonic()-start)
        row = {"mode": mode, "median_batch_seconds": sorted(elapsed[1:])[1],
               "peak_MiB": torch.cuda.max_memory_allocated()/1024**2,
               "parameters": sum(p.numel() for p in model.parameters()),
               "estimated_cell_seconds": sorted(elapsed[1:])[1]*n_batches*protocol["epochs"]}
        rows.append(row); print(json.dumps(row), flush=True)
        del model; torch.cuda.empty_cache()
    estimated = sum(r["estimated_cell_seconds"] for r in rows) * 72 / 3600
    result = {"status": "PASS", "rows": rows, "execution_sha256": execution_hash(),
              "optimizer_updates": 0, "heldout_forward": False, "official_test_read": False,
              "projected_matrix_hours": estimated, "projection_125pct_hours": estimated*1.25,
              "training_units": training_units, "n_training_flights": len(training["y"]),
              "admission": estimated*1.25 < protocol["maximum_gpu_training_hours"] and
                           max(r["peak_MiB"] for r in rows) < protocol["maximum_device_memory_MiB"]}
    dump_new(args.output, result)
    print(json.dumps(result), flush=True)
    if not result["admission"]:
        raise RuntimeError("frozen matrix exceeds hardware evidence budget")


if __name__ == "__main__":
    main()
