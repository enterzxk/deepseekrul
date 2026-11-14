"""Original initialized HNN equivalence and hardware admission; zero updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from paper_b.ncmapss_response.prepare import dump_new
from .adapter_checks import check_network, check_packing, check_source
from .bayes_adapter import BayesHNNAdapter, pack_window
from .run_bayes import execution_hash, input_scalers, verify_schema_contract


def preprocessing_contracts():
    rng = np.random.default_rng(612)
    units = np.repeat(np.arange(1, 7), 4)
    data = {"raw": rng.normal(size=(24, 512, 18)).astype(np.float32),
            "unit": units, "cycle": np.tile(np.arange(1, 5), 6),
            "y": np.linspace(0, 180, 24).astype(np.float32)}
    data["raw"][:, :, 0] = 7.
    for train in ([1, 2, 3, 4], [1, 2, 3, 4, 5]):
        scalers = input_scalers(data, train)
        keep = np.isin(units, train)
        altered = {k: v.copy() for k, v in data.items()}
        for name in ("raw", "y", "cycle"):
            altered[name][~keep] += 10000
        assert scalers == input_scalers(altered, train), "excluded engine affected fitted scaler"
        mu = data["raw"][keep].astype(np.float64).mean((0, 1))
        sd = np.maximum(data["raw"][keep].astype(np.float64).std((0, 1)), 1e-6)
        assert np.array_equal(mu, np.asarray(scalers["raw_mu"]))
        assert np.array_equal(sd, np.asarray(scalers["raw_sd"]))
        assert scalers["target_mu"] == 0 and scalers["target_sd"] == 125
        assert scalers["age_q95"] == max(1., float(np.percentile(data["cycle"][keep], 95)))
        y = np.minimum(data["y"], 125)/125
        assert y.min() >= 0 and y.max() <= 1
        assert np.array_equal(np.array([0., .25, 1.])*125, [0., 31.25, 125.])
    return {"inner4_outer5_excluded_engine_invariance": True,
            "global_training_only_mean_std": True, "constant_std_floor": 1e-6,
            "target_and_sigma_inverse_factor": 125, "optimizer_updates": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(__file__).with_name("PROTOCOL.json").read_text())
    schema = verify_schema_contract()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.9, protocol["maximum_device_memory_MiB"]*1024**2/memory))
    preprocessing = preprocessing_contracts()
    software = {"source": check_source(args.author_file), "packing": check_packing("cuda:0"),
                "network": check_network(args.author_file, "cuda:0")}
    n = 80
    raw = torch.randn(n, 512, 18, device="cuda:0")
    age = torch.arange(1, n+1, device="cuda:0").float()/n
    windows = torch.full((n, 70), -1, dtype=torch.long, device="cuda:0")
    for q in range(n):
        seq = torch.arange(max(0, q-69), q+1, device="cuda:0")
        windows[q, -len(seq):] = seq
    data = {"raw": raw, "age": age, "windows": windows}
    query = torch.arange(16, 80, device="cuda:0")
    rows = []
    for history, points in protocol["window_catalog"]:
        model = BayesHNNAdapter(args.author_file, history, points, device="cuda:0").train()
        times = []
        torch.cuda.reset_peak_memory_stats()
        for repetition in range(4):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            x, a = pack_window(data, query, history, points)
            a.requires_grad_(True)
            out = model(x, a)
            assert bool(torch.isfinite(out["mu_norm"]).all() & torch.isfinite(out["sigma_norm"]).all())
            assert bool((out["mu_norm"] > 0).all() & (out["sigma_norm"] > 0).all())
            loss = model.nll(out["mu_norm"], torch.full((64,), .4, device="cuda:0"), out["sigma_norm"])
            assert bool(torch.isfinite(loss))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            assert a.grad is not None and bool(torch.isfinite(a.grad).all()) and bool(a.grad[:, -1].ne(0).any())
            assert not bool(a.grad[:, :-1].ne(0).any()), "only current observed age is supplied"
            assert bool(model.author_model.last.weight.grad[0].ne(0).any())
            assert bool(model.author_model.last.weight.grad[1].ne(0).any())
            torch.cuda.synchronize()
            times.append(time.monotonic()-started)
        row = {"history": history, "points": points, "author_sequence_len": history*points+1,
               "batch": 64, "parameters": model.parameter_count,
               "median_batch_seconds": sorted(times[1:])[1],
               "peak_MiB": torch.cuda.max_memory_allocated()/1024**2,
               "last_gradient_norm": float(norm), "last_nll": float(loss.detach()),
               "both_heads_and_current_age_gradients": True}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, x, a, out, loss
        torch.cuda.empty_cache()
    inner = 24*sum(r["median_batch_seconds"]*6*200 for r in rows)
    outer = 72*max(r["median_batch_seconds"] for r in rows)*8*200
    hours = (inner+outer)*1.25/3600
    admitted = hours < protocol["maximum_shared_gpu_wall_hours"] and max(r["peak_MiB"] for r in rows) < protocol["maximum_device_memory_MiB"]
    result = {"status": "PASS" if admitted else "FAILED", "admission": admitted,
              "execution_sha256": execution_hash(args.author_file), "schema": schema,
              "software": software, "preprocessing": preprocessing, "rows": rows,
              "projected_serial_hours_with_25pct_margin": hours,
              "optimizer_updates": 0, "NASA_data_or_labels_read": False, "official_test_read": False,
              "interpretation": "real complete initialized HNN forward/backward; no training or gain claim"}
    dump_new(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("software", "rows")}), flush=True)
    if not admitted:
        raise RuntimeError("full HNN catalog exceeds preregistered hardware budget")


if __name__ == "__main__":
    main()
