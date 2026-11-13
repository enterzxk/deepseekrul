"""Immutable, development-only paired experiment controller.

This file never opens HDF5 or imports the previous N-CMAPSS training pipeline.
All model choices and the fixed stopping rule are registered before any fit.
"""
from __future__ import annotations

import argparse
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
from torch import nn

from .models import ResponseEncoder
from .prepare import digest, dump_new

HERE = Path(__file__).parent


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(canonical(list(a.shape)))
        h.update(a.tobytes())
    return h.hexdigest()


def execution_hash():
    paths = [HERE / name for name in ("__init__.py", "models.py", "prepare.py", "run.py", "PROTOCOL.json")]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y, pred):
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    error = pred - y
    score = np.expm1(np.where(error >= 0, error / 10.0, -error / 13.0))
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "nasa_sum": float(score.sum()), "nasa_mean": float(score.mean()),
            "bias": float(error.mean()), "n": len(y)}


def load_cache(folder, subset):
    folder = Path(folder)
    manifest = json.loads((folder / f"{subset}_dev_manifest.json").read_text())
    path = folder / f"{subset}_dev.npz"
    if digest(path) != manifest["cache_sha256"]:
        raise AssertionError("cache hash mismatch")
    if manifest["official_test_decoded"] or manifest["decoded_keys"] != ["A_dev", "W_dev", "X_s_dev", "Y_dev"]:
        raise AssertionError("invalid development-only data boundary")
    with np.load(path, allow_pickle=False) as f:
        # Source indices and original flight duration are audit fields only.
        data = {k: f[k] for k in ("raw", "y", "unit", "cycle")}
    if data["raw"].shape != (len(data["y"]), 512, 18):
        raise AssertionError("unexpected shared information contract")
    if len(np.unique(data["unit"])) != 6:
        raise AssertionError("this pre-registration requires six development engines")
    return data, manifest


def fit_scalers(data, train_units, cap):
    keep = np.isin(data["unit"], train_units)
    raw = data["raw"][keep].astype(np.float64)
    y = np.minimum(data["y"][keep].astype(np.float64), cap)
    return {"raw_mu": raw.mean(axis=(0, 1)).tolist(),
            "raw_sd": np.maximum(raw.std(axis=(0, 1)), 1e-6).tolist(),
            "age_q95": max(1.0, float(np.percentile(data["cycle"][keep], 95))),
            "target_mu": float(y.mean()), "target_sd": max(1.0, float(y.std())),
            "fitted_units": list(map(int, train_units))}


def bundle(data, units, scalers, protocol, device, view="clean"):
    keep = np.isin(data["unit"], units)
    original = data["raw"][keep].astype(np.float32).copy()
    unit, cycle, y = (data[k][keep].copy() for k in ("unit", "cycle", "y"))
    if view == "density":
        # Same paired observations, intentionally changed W sampling density.
        # This is a robustness intervention, not a physical counterfactual label.
        for i in range(len(original)):
            altitude = original[i, :, 0].astype(np.float64)
            rank = np.argsort(np.argsort(altitude, kind="stable"), kind="stable")
            weights = 0.25 + 1.5 * rank / max(1, len(rank) - 1)
            cdf = np.cumsum(weights) / weights.sum()
            idx = np.searchsorted(cdf, (np.arange(512) + 0.5) / 512)
            original[i] = original[i, idx]
    elif view == "unpaired":
        # Preserve both empirical marginals while breaking their synchronization.
        for i in range(len(original)):
            rng = np.random.default_rng(17011 + int(unit[i]) * 1000 + int(cycle[i]))
            original[i, :, :4] = original[i, rng.permutation(512), :4]
    elif view != "clean":
        raise ValueError(view)
    raw = torch.as_tensor((original - np.asarray(scalers["raw_mu"], np.float32)) /
                          np.asarray(scalers["raw_sd"], np.float32), device=device)
    prior = np.zeros(len(y), np.int64)
    windows = np.full((len(y), protocol["span"]), -1, np.int64)
    for u in units:
        ix = np.flatnonzero(unit == u)
        if not np.all(np.diff(cycle[ix]) > 0):
            raise AssertionError("flight sequence must be strictly increasing")
        for pos, i in enumerate(ix):
            prior[i] = ix[max(0, pos - protocol["prior_lag_flights"])]
            seq = ix[max(0, pos - protocol["span"] + 1):pos + 1]
            windows[i, -len(seq):] = seq
            if cycle[prior[i]] > cycle[i] or unit[prior[i]] != unit[i]:
                raise AssertionError("noncausal or cross-engine reference")
    prior_t = torch.as_tensor(prior, dtype=torch.long, device=device)
    # Physical W-only support is the same visible information in every arm.
    mc, mr = [], []
    with torch.no_grad():
        for i, ref in enumerate(prior):
            if i == ref:
                mc.append(torch.zeros(512, dtype=torch.bool, device=device))
                mr.append(torch.zeros(512, dtype=torch.bool, device=device))
            else:
                distances = torch.cdist(raw[i, :, :4], raw[ref, :, :4])
                mc.append(distances.min(1).values <= protocol["support_radius"])
                mr.append(distances.min(0).values <= protocol["support_radius"])
    return {"raw": raw, "prior": prior_t, "support_current": torch.stack(mc),
            "support_ref": torch.stack(mr),
            "age": torch.as_tensor(cycle / scalers["age_q95"], dtype=torch.float32, device=device),
            "windows": torch.as_tensor(windows, dtype=torch.long, device=device),
            "target": torch.as_tensor((np.minimum(y, protocol["label_cap"]) - scalers["target_mu"]) /
                                      scalers["target_sd"], dtype=torch.float32, device=device),
            "y": y, "unit": unit, "cycle": cycle}


class SequenceHead(nn.Module):
    def __init__(self, protocol):
        super().__init__()
        d, span = protocol["hidden"], protocol["span"]
        self.input = nn.Linear(d + 1, d)
        pe = torch.zeros(span, d)
        position = torch.arange(span).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * scale); pe[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, protocol["heads"], dim_feedforward=2*d,
                                           dropout=protocol["dropout"], activation="gelu", batch_first=True)
        self.body = nn.TransformerEncoder(layer, protocol["layers"], enable_nested_tensor=False)
        self.output = nn.Linear(d, 1)

    def forward(self, encoded, data, query):
        win = data["windows"][query]; valid = win >= 0
        indices = win.clamp_min(0)
        window_start = win.masked_fill(~valid, torch.iinfo(win.dtype).max).amin(dim=1)
        reference_visible = valid & (data["prior"][indices] >= window_start[:, None])
        emb = torch.where(reference_visible.unsqueeze(-1), encoded["embedding"][indices],
                          encoded["embedding_without_reference"][indices])
        x = torch.cat((emb, data["age"][indices].unsqueeze(-1)), dim=-1)
        x = self.input(x) + self.pe.unsqueeze(0)
        # Every query window ends at the prediction flight. No later flight exists in it.
        x = self.body(x, src_key_padding_mask=~valid)
        return self.output(x[:, -1]).squeeze(-1)


class Model(nn.Module):
    def __init__(self, mode, protocol, seed):
        super().__init__()
        self.mode = mode
        if mode == "age":
            seed_everything(seed + 10000)
            self.age_net = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            seed_everything(seed)
            self.encoder = ResponseEncoder("raw" if mode == "w_only" else mode)
            # Identical temporal initialization in all paired neural arms.
            seed_everything(seed + 10000)
            self.head = SequenceHead(protocol)

    def forward(self, data, query, block=None):
        if self.mode == "age":
            pred = self.age_net(data["age"][query, None]).squeeze(-1)
            return pred, pred.new_zeros(()), {}
        observed = data["raw"]
        if self.mode == "w_only":
            observed = torch.cat((observed[..., :4], torch.zeros_like(observed[..., 4:])), -1)
        encoded = self.encoder(observed, data["prior"], data["support_current"], data["support_ref"],
                               block=None if self.mode == "w_only" else block)
        aux = encoded["reconstruction_per_flight"][query].mean() if block is not None and self.mode != "w_only" else encoded["embedding"].new_zeros(())
        return self.head(encoded, data, query), aux, encoded.get("diagnostics", {})


def predict(model, data, scalers, batch):
    model.eval(); preds = []; diagnostic = None
    # Encoder is flight-local plus an explicitly causal reference; batching queries is safe.
    with torch.no_grad():
        for start in range(0, len(data["y"]), batch):
            query = torch.arange(start, min(start + batch, len(data["y"])), device=data["raw"].device)
            p, _, diagnostic = model(data, query, block=None)
            preds.append(p.detach().cpu().numpy())
    pred = np.concatenate(preds).astype(np.float64) * scalers["target_sd"] + scalers["target_mu"]
    return pred, diagnostic


def json_diagnostics(d):
    if isinstance(d, torch.Tensor):
        a = d.detach().cpu().float().numpy()
        result = {"shape": list(a.shape), "mean": float(a.mean()), "minimum": float(a.min()), "maximum": float(a.max()),
                  "nonzero_fraction_1e6": float(np.mean(np.abs(a) > 1e-6))}
        if d.dtype in (torch.int32, torch.int64, torch.bool):
            values, counts = np.unique(a.astype(np.int64), return_counts=True)
            result["counts"] = {str(int(v)): int(n) for v, n in zip(values, counts)}
        return result
    if isinstance(d, dict):
        return {k: json_diagnostics(v) for k, v in d.items()}
    if isinstance(d, (float, int, str, bool)) or d is None:
        return d
    return str(d)


def run_cell(data, manifest, protocol, mode, held_out, seed, output, device):
    cell = f"{manifest['subset']}_{mode}_u{held_out}_s{seed}"
    folder = output / cell
    if (folder / "SUCCESS.json").exists():
        seal = json.loads((folder / "SUCCESS.json").read_text())
        for name, expected in seal["files_sha256"].items():
            if digest(folder / name) != expected:
                raise AssertionError(f"sealed artifact was modified: {cell}/{name}")
        previous = json.loads((folder / "config.json").read_text())
        if not (previous["execution_sha256"] == execution_hash() and previous["protocol"] == protocol
                and previous["cell"] == cell and previous["source_sha256"] == manifest["source_sha256"]
                and previous["cache_sha256"] == manifest["cache_sha256"]):
            raise AssertionError(f"refusing to reuse a different experiment identity: {cell}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    dump_new(folder / "CLAIM.json", {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})
    train_units = sorted(int(u) for u in np.unique(data["unit"]) if u != held_out)
    config = {"protocol": protocol, "cell": cell, "seed": seed, "mode": mode,
              "train_units": train_units, "held_out_unit": held_out,
              "source_sha256": manifest["source_sha256"], "cache_sha256": manifest["cache_sha256"],
              "execution_sha256": execution_hash()}
    config["config_sha256"] = hashlib.sha256(canonical(config)).hexdigest()
    dump_new(folder / "config.json", config)
    scalers = fit_scalers(data, train_units, protocol["label_cap"])
    dump_new(folder / "scalers.json", scalers)
    training = bundle(data, train_units, scalers, protocol, device)
    seed_everything(seed)
    model = Model(mode, protocol, seed).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_parameters = model.encoder.parameter_report() if mode != "age" else None
    if mode == "w_only":
        inactive = sum(p.numel() for module in (model.encoder.response_body, model.encoder.background,
                                               model.encoder.basis, model.encoder.aux_context_head)
                       for p in module.parameters())
        inactive += model.encoder.raw_encoder[0].weight[:, 4:].numel()
        inactive += model.encoder.embedding.weight[:, 64:72].numel()
        encoder_parameters = {"total_parameters": sum(p.numel() for p in model.encoder.parameters()),
                              "inactive_by_design": inactive,
                              "structurally_effective_parameters": sum(p.numel() for p in model.encoder.parameters())-inactive,
                              "scope": "W-only MSE reference; no sensor auxiliary task; zero Xs input columns excluded"}
    init_hash = state_hash(model)
    torch.save({"model": model.state_dict(), "config": config, "scalers": scalers,
                "initial_state_sha256": init_hash}, folder / "initial.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(seed + 20000)
    seed_everything(seed + 30000)
    start_time = time.monotonic(); updates = 0; trace = []
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(protocol["epochs"]):
        model.train(); losses = []; auxiliaries = []
        order = torch.randperm(len(training["y"]), generator=generator).tolist()
        for start in range(0, len(order), protocol["batch"]):
            query = torch.tensor(order[start:start+protocol["batch"]], device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            pred, aux, _ = model(training, query, block=epoch % 4)
            task = torch.mean((pred - training["target"][query]) ** 2)
            loss = task + protocol["auxiliary_weight"] * aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss {cell}, epoch {epoch+1}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"], error_if_nonfinite=True)
            optimizer.step(); updates += 1
            if torch.cuda.max_memory_allocated(device)/1024**2 > protocol["maximum_device_memory_MiB"]:
                raise MemoryError("pre-registered device-memory budget exceeded")
            losses.append(float(task.detach())); auxiliaries.append(float(aux.detach()))
        if epoch == 0 or (epoch+1) % 20 == 0 or epoch+1 == protocol["epochs"]:
            trace.append({"epoch": epoch+1, "task": float(np.mean(losses)), "auxiliary": float(np.mean(auxiliaries)),
                          "gradient_norm_last": float(norm), "elapsed_seconds": time.monotonic()-start_time})
        if time.monotonic() - start_time > 3600:
            raise TimeoutError("single-cell one-hour evidence budget exhausted")
    torch.cuda.synchronize(device)
    training_seconds = time.monotonic() - start_time
    final_hash = state_hash(model)
    if final_hash == init_hash:
        raise AssertionError("weights did not change during claimed training")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "scalers": scalers, "epochs_completed": protocol["epochs"],
                "optimizer_updates": updates, "initial_state_sha256": init_hash,
                "final_state_sha256": final_hash}, folder / "checkpoint.pt")
    dump_new(folder / "training_trace.json", trace)
    scores = {}; predictions = {}
    for view in ("clean", "density", "unpaired"):
        heldout = bundle(data, [held_out], scalers, protocol, device, view=view)
        unbounded, diagnostic = predict(model, heldout, scalers, protocol["batch"])
        pred = np.maximum(unbounded, 0.0)
        if not np.isfinite(pred).all():
            raise FloatingPointError("nonfinite held-out prediction")
        predictions[view] = pred
        scores[view] = {"raw": metrics(heldout["y"], pred),
                        "clip125": metrics(np.minimum(heldout["y"], protocol["label_cap"]), pred),
                        "diagnostics": json_diagnostics(diagnostic)}
        with open(folder / f"prediction_{view}.csv", "x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f); writer.writerow(["unit", "cycle", "y_true_raw", "y_pred", "y_pred_unbounded", "seed", "view"])
            writer.writerows(zip(heldout["unit"], heldout["cycle"], heldout["y"], pred, unbounded,
                                 [seed]*len(pred), [view]*len(pred)))
    for view in ("density", "unpaired"):
        scores[view]["mean_absolute_prediction_drift"] = float(np.mean(np.abs(predictions[view]-predictions["clean"])))
    result = {"cell": cell, "subset": manifest["subset"], "mode": mode, "held_out_unit": held_out,
              "seed": seed, "config_sha256": config["config_sha256"], "execution_sha256": config["execution_sha256"],
              "source_sha256": manifest["source_sha256"], "parameters": params,
              "encoder_parameter_report": encoder_parameters,
              "structurally_effective_parameters": params-(encoder_parameters["inactive_by_design"] if encoder_parameters else 0),
              "auxiliary_disabled_for_reference": mode in ("age", "w_only"),
              "training_seconds": training_seconds, "optimizer_updates": updates,
              "initial_state_sha256": init_hash, "final_state_sha256": final_hash,
              "peak_allocated_MiB": torch.cuda.max_memory_allocated(device)/1024**2,
              "scores": scores, "official_test_read": False, "checkpoint_sha256": digest(folder/"checkpoint.pt")}
    dump_new(folder / "metrics.json", result)
    files = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}
    dump_new(folder / "SUCCESS.json", {"cell": cell, "files_sha256": files})
    print(json.dumps({k: result[k] for k in ("cell", "training_seconds", "parameters", "optimizer_updates", "peak_allocated_MiB")} |
                     {"rmse": scores["clean"]["raw"]["rmse"]}), flush=True)
    del model, optimizer, training
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--subsets", nargs="+", default=None)
    p.add_argument("--modes", nargs="+", default=None); p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--held-out", nargs="+", type=int, default=None)
    args = p.parse_args()
    protocol = json.loads((HERE/"PROTOCOL.json").read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not args.device.startswith("cuda"):
        raise ValueError("formal real-training matrix is CUDA FP32")
    for selected, allowed in ((args.subsets, protocol["subsets"]), (args.modes, protocol["modes"]),
                               (args.seeds, protocol["seeds"])):
        if selected is not None and not set(selected).issubset(allowed):
            raise ValueError("requested grid is outside the pre-registration")
    total_memory = torch.cuda.get_device_properties(args.device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.9, protocol["maximum_device_memory_MiB"]*1024**2/total_memory), args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        for subset in args.subsets or protocol["subsets"]:
            data, manifest = load_cache(args.cache, subset)
            if args.held_out is not None and not set(args.held_out).issubset(set(map(int, data["unit"]))):
                raise ValueError("held-out identifier is not a development engine")
            for mode in args.modes or protocol["modes"]:
                for seed in args.seeds or protocol["seeds"]:
                    for u in args.held_out or sorted(map(int, np.unique(data["unit"]))):
                        run_cell(data, manifest, protocol, mode, u, seed, args.output, args.device)
                        if time.monotonic()-start > protocol["maximum_gpu_training_hours"]*3600:
                            raise TimeoutError("pre-registered process-hour budget reached")
        request = vars(args) | {"cache": str(args.cache), "output": str(args.output)}
        request_id = hashlib.sha256(canonical(request)).hexdigest()[:12]
        dump_new(args.output/f"GRID_{request_id}_COMPLETE.json", {"execution_sha256": execution_hash(), "arguments": request,
                                                               "elapsed_seconds": time.monotonic()-start})
        if args.subsets is None and args.modes is None and args.seeds is None and args.held_out is None:
            dump_new(args.output/"ALL_TRAINING_COMPLETE.json", {"execution_sha256": execution_hash(), "training_runs": protocol["training_runs"]})
    except BaseException as e:
        name = f"FAILURE_{int(time.time())}_{os.getpid()}.json"
        dump_new(args.output/name, {"type": type(e).__name__, "message": str(e), "execution_sha256": execution_hash()})
        raise


if __name__ == "__main__":
    main()
