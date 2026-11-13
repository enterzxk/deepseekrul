"""Full-author software equivalence and B64 hardware admission; no optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from paper_b.ncmapss_response.prepare import dump_new
from .aidgn_adapter import AIDGNAdapter, pack_window
from .adapter_checks import run_checks
from .run_aidgn import execution_hash, input_scalers


def preprocessing_contracts():
    rng = np.random.default_rng(612)
    unit = np.repeat(np.arange(1, 7), 4)
    raw = rng.normal(size=(24, 512, 18)).astype(np.float32)
    raw[:, :, 0] = 7.
    data = {"raw": raw, "unit": unit, "cycle": np.tile(np.arange(1, 5), 6),
            "y": np.linspace(0, 180, 24).astype(np.float32)}
    for units in ([1, 2, 3, 4], [1, 2, 3, 4, 5]):
        scalers = input_scalers(data, units)
        keep = np.isin(unit, units)
        changed = {key: value.copy() for key, value in data.items()}
        changed["raw"][~keep] += 10000
        changed["y"][~keep] += 10000; changed["cycle"][~keep] += 10000
        if scalers != input_scalers(changed, units):
            raise AssertionError("excluded-engine preprocessing leakage")
        offset = np.asarray(scalers["raw_mu"]); scale = np.asarray(scalers["raw_sd"])
        transformed = (data["raw"][keep].astype(np.float64)-offset)/scale
        if not np.allclose(transformed[:, :, 1:].min((0, 1)), -1, atol=1e-12) or not np.allclose(transformed[:, :, 1:].max((0, 1)), 1, atol=1e-12):
            raise AssertionError("author [-1,1] normalization not preserved")
        if not np.array_equal(transformed[:, :, 0], -np.ones_like(transformed[:, :, 0])):
            raise AssertionError("constant columns must map to -1")
        beyond = offset+3*scale
        if not np.all((beyond-offset)/scale > 1):
            raise AssertionError("heldout extrapolation was clipped")
        target = (np.minimum(data["y"], 125)-scalers["target_mu"])/scalers["target_sd"]
        if not np.array_equal(target, np.minimum(data["y"], 125)/125) or target.min() < 0:
            raise AssertionError("positive author-head target scaling is wrong")
        if not np.array_equal(np.array([0., .25, .5, 1.])*scalers["target_sd"]+scalers["target_mu"], np.array([0., 31.25, 62.5, 125.])):
            raise AssertionError("physical RUL inverse scaling is wrong")
    return {"inner4_outer5_excluded_engine_invariance": True, "author_minmax_range": [-1, 1],
            "constant_columns_map_minus_one": True, "heldout_values_not_clipped": True,
            "positive_cap125_div125_and_inverse": True, "optimizer_updates": 0}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--author-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    protocol = json.loads((Path(__file__).parent/"PROTOCOL.json").read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.9, protocol["maximum_device_memory_MiB"]*1024**2/memory))
    preprocessing = preprocessing_contracts()
    software = run_checks(args.author_file, device="cuda:0")
    f = 80
    raw = torch.randn(f, 512, 18, device="cuda:0")
    age = torch.arange(1, f+1, device="cuda:0").float()/f
    windows = torch.full((f, 70), -1, device="cuda:0", dtype=torch.long)
    for q in range(f):
        sequence = torch.arange(max(0, q-69), q+1, device="cuda:0")
        windows[q, -len(sequence):] = sequence
    data = {"raw": raw, "age": age, "windows": windows}
    query = torch.arange(16, 80, device="cuda:0")
    rows = []
    for history, points in protocol["window_catalog"]:
        model = AIDGNAdapter(args.author_file, history, points).cuda().train()
        times = []; torch.cuda.reset_peak_memory_stats()
        for repetition in range(4):
            model.zero_grad(set_to_none=True); torch.cuda.synchronize(); started = time.monotonic()
            x, a = pack_window(data, query, history, points)
            prediction = model(x, a)
            loss = (prediction-.4).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            torch.cuda.synchronize(); times.append(time.monotonic()-started)
        row = {"history": history, "points": points, "sequence_len": history*points,
               "batch": 64, "median_batch_seconds": sorted(times[1:])[1],
               "parameters": sum(p.numel() for p in model.parameters()),
               "peak_MiB": torch.cuda.max_memory_allocated()/1024**2,
               "last_gradient_norm": float(norm)}
        rows.append(row); print(json.dumps(row), flush=True)
        del model, x, a, prediction, loss
        torch.cuda.empty_cache()
    # Conservative 6 training batches in inner4 and 8 in outer5, 200 passes.
    inner_seconds = 24*sum(r["median_batch_seconds"]*6*200 for r in rows)
    outer_seconds = 72*max(r["median_batch_seconds"] for r in rows)*8*200
    expected_hours = (inner_seconds+outer_seconds)*1.25/3600
    admission = expected_hours < protocol["maximum_shared_gpu_wall_hours"] and max(r["peak_MiB"] for r in rows) < protocol["maximum_device_memory_MiB"]
    result = {"status": "PASS" if admission else "FAILED", "admission": admission,
              "execution_sha256": execution_hash(args.author_file), "software": software, "preprocessing": preprocessing,
              "rows": rows, "projected_serial_hours_with_25pct_margin": expected_hours,
              "optimizer_updates": 0, "NASA_data_or_labels_read": False, "official_test_read": False,
              "interpretation": "real original-model forward/backward; no training or benefit claim"}
    dump_new(args.output, result)
    print(json.dumps(result), flush=True)
    if not admission:
        raise RuntimeError("full-author window catalog exceeds budget; do not silently drop a window")


if __name__ == "__main__":
    main()
