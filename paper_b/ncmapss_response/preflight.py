"""Numerical and information-boundary checks, with zero optimizer updates."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .model_checks import run_checks
from .prepare import digest, dump_new
from .run import Model, bundle, execution_hash, fit_scalers, run_cell


def check_history(protocol, device):
    rng = np.random.default_rng(721)
    data = {"raw": rng.normal(size=(80, 512, 18)).astype(np.float32),
            "unit": np.ones(80, np.int64), "cycle": np.arange(1, 81),
            "y": np.arange(79, -1, -1).astype(np.float32)}
    scaler = fit_scalers(data, [1], protocol["label_cap"])
    clean = bundle(data, [1], scaler, protocol, device)
    queries = torch.tensor([68, 70, 74, 79], device=device)
    checks = []
    for mode in ("raw", "fe", "delta", "supported", "common"):
        model = Model(mode, protocol, 101).to(device).eval()
        with torch.no_grad():
            baseline, _, _ = model(clean, queries)
            for qpos, t in enumerate(queries.tolist()):
                low = max(0, t - 74); high = max(0, t - 69)
                if high == low:
                    continue
                for what in ("W", "Xs", "both"):
                    changed = {k: v.copy() for k, v in data.items()}
                    columns = slice(0, 4) if what == "W" else (slice(4, 18) if what == "Xs" else slice(0, 18))
                    changed["raw"][low:high, :, columns] += 7.0
                    altered = bundle(changed, [1], scaler, protocol, device)
                    predicted, _, _ = model(altered, queries)
                    if not torch.equal(baseline[qpos], predicted[qpos]):
                        raise AssertionError(f"outside-history influence: {mode}, q={t}, {what}, {baseline[qpos]} vs {predicted[qpos]}")
                    checks.append(f"{mode}:t{t}:{what}:outside70_unchanged")
            # Within the visible window, the model must still be sensitive.
            changed = {k: v.copy() for k, v in data.items()}
            changed["raw"][-1, :, 4:] += 2.0
            altered = bundle(changed, [1], scaler, protocol, device)
            predicted, _, _ = model(altered, queries)
            if torch.equal(baseline[-1], predicted[-1]):
                raise AssertionError("history check passed by an insensitive constant model")
        model.train()
        pred, aux, _ = model(clean, queries, block=0)
        (pred.square().mean() + protocol["auxiliary_weight"] * aux).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise AssertionError(f"nonfinite production-FP32 gradients: {mode}")
        checks.append(f"{mode}:production_FP32_backward_finite_zero_optimizer_updates")
        del model
    w_model = Model("w_only", protocol, 101).to(device).eval()
    changed = {k: v.copy() for k, v in data.items()}
    changed["raw"][..., 4:] += 8.0
    altered = bundle(changed, [1], scaler, protocol, device)
    with torch.no_grad():
        p0, a0, _ = w_model(clean, queries, block=0)
        p1, a1, _ = w_model(altered, queries, block=0)
    if not torch.equal(p0, p1) or float(a0) != 0 or float(a1) != 0:
        raise AssertionError("W-only reference is using sensor observations or auxiliary labels")
    checks.append("W_only_ignores_all_Xs_values_and_has_no_sensor_auxiliary_supervision")
    return checks


def check_seal(protocol):
    # No model construction/forward/training occurs on the verified-skip path.
    with tempfile.TemporaryDirectory(prefix="nc_response_seal_") as t:
        root = Path(t); cell = "DS02_raw_u2_s101"; folder = root/cell; folder.mkdir()
        manifest = {"subset": "DS02", "source_sha256": "source", "cache_sha256": "cache"}
        config = {"execution_sha256": execution_hash(), "protocol": protocol, "cell": cell,
                  "source_sha256": "source", "cache_sha256": "cache"}
        dump_new(folder/"config.json", config)
        (folder/"evidence.bin").write_bytes(b"original")
        dump_new(folder/"SUCCESS.json", {"files_sha256": {"config.json": digest(folder/"config.json"),
                                                           "evidence.bin": digest(folder/"evidence.bin")}})
        run_cell(None, manifest, protocol, "raw", 2, 101, root, "cpu")
        (folder/"evidence.bin").write_bytes(b"tampered")
        try:
            run_cell(None, manifest, protocol, "raw", 2, 101, root, "cpu")
        except AssertionError:
            pass
        else:
            raise AssertionError("tampered seal accepted")
    return ["valid_resume_seal_accepted", "tampered_resume_seal_rejected"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    protocol = json.loads((Path(__file__).parent/"PROTOCOL.json").read_text())
    started = time.monotonic()
    cpu = run_checks()
    # run_checks intentionally chooses CPU as default; reset before CUDA probes.
    torch.set_default_device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    history = check_history(protocol, args.device)
    seals = check_seal(protocol)
    checks = cpu + history + seals
    result = {"status": "PASS", "checks": checks, "n_checks": len(checks),
              "execution_sha256": execution_hash(), "optimizer_updates": 0,
              "official_test_read": False, "seconds": time.monotonic()-started,
              "torch": torch.__version__, "device": str(torch.cuda.get_device_name(args.device))}
    dump_new(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
