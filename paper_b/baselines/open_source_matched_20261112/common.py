"""Frozen-source identity, common FP32 condition and independent selection evidence."""
from __future__ import annotations
import builtins
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from ..open_source.common import config_hash, digest, predictions, research_task_start, source_record, write_json
from ..open_source.common import seed_everything as original_seed
from .reference.training.metrics import summarize, per_engine_rmse

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
MODELS = PROTOCOL["models"]
REFERENCES = ["dspt_v0_cycle", "dspt_train_defaults"]


def verify_sources():
    record = json.loads((ROOT / "EXECUTION_SOURCE_MANIFEST.json").read_text())
    for item in record["files"]:
        assert digest(PROJECT / item["path"]) == item["sha256"], item["path"]
    return digest(ROOT / "EXECUTION_SOURCE_MANIFEST.json")


def verify_training_data(cfg, data_dir):
    assert Path(data_dir).resolve() == Path(cfg["data_dir"]).resolve(), "frozen data directory changed"
    expected = json.loads((ROOT / "DATASET_EXPECTED_SHA256.json").read_text())
    for subset, recorded in cfg["training_data_sha256"].items():
        name = f"train_{subset}.txt"
        assert digest(Path(data_dir) / name) == recorded == expected[name], name


def seed_everything(seed):
    original_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def backend_flags():
    return {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}


@contextlib.contextmanager
def fp32_autocast():
    """Disable autocast in the original DSPT trainer/predictor without editing it."""
    original = torch.autocast
    def disabled(*args, **kwargs):
        kwargs["enabled"] = False
        return original(*args, **kwargs)
    torch.autocast = disabled
    try:
        yield
    finally:
        torch.autocast = original


class Guard:
    def __init__(self, phase, unlocked=False):
        self.record = {"phase": phase, "official_test_unlocked": unlocked,
                       "official_test_reads": [], "violations": [], "nonfinite_replacements": 0}
    def __enter__(self):
        self.open, self.io_open, self.stdout = builtins.open, io.open, sys.stdout
        def wrap(original):
            def opened(file, *args, **kwargs):
                if isinstance(file, (str, bytes, Path)):
                    name = Path(file.decode() if isinstance(file, bytes) else file).name
                    if name.startswith(("test_FD00", "RUL_FD00")):
                        if not self.record["official_test_unlocked"]:
                            self.record["violations"].append(str(file))
                            raise PermissionError("official TEST is locked during training")
                        self.record["official_test_reads"].append({"path": str(file), "unix": time.time()})
                return original(file, *args, **kwargs)
            return opened
        guard = self
        class Stdout:
            def write(self, text):
                if "Invalid output detected" in text:
                    guard.record["nonfinite_replacements"] += 1
                    raise FloatingPointError("upstream nonfinite prediction replacement rejected")
                return guard.stdout.write(text)
            def flush(self): return guard.stdout.flush()
            def __getattr__(self, name): return getattr(guard.stdout, name)
        builtins.open, io.open, sys.stdout = wrap(self.open), wrap(self.io_open), Stdout()
        return self
    def __exit__(self, *args):
        builtins.open, io.open, sys.stdout = self.open, self.io_open, self.stdout


def metrics(y, p):
    y, p = np.asarray(y).reshape(-1), np.asarray(p).reshape(-1)
    assert len(y) == len(p) and np.isfinite(y).all() and np.isfinite(p).all()
    return {**summarize(p, y), "n": len(y)}


def validation_record(data, p):
    y, p, unit = np.asarray(data["y"], float), np.asarray(p, float), np.asarray(data["unit"])
    parts = []
    for u in np.unique(unit):
        errors = p[unit == u] - y[unit == u]
        parts.append({"unit": int(u), "n": len(errors), "sum_squared_error": math.fsum(float(e)**2 for e in errors)})
    engine_mean = math.fsum(math.sqrt(r["sum_squared_error"] / r["n"]) for r in parts) / len(parts)
    return {"val_engine_mean_rmse": engine_mean, "val_pooled_rmse": metrics(y, p)["rmse"],
            "engine_squared_errors": parts}


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def configuration(model, subset, seed, data_dir):
    assert model in MODELS and subset in PROTOCOL["subsets"] and seed in PROTOCOL["seeds"]
    fd = PROTOCOL["subsets"].index(subset)
    cfg = {key: PROTOCOL[key] for key in ["split_seed", "fold", "val_frac", "cuts", "rul_cap", "n_cond", "epochs", "patience", "min_delta", "precision"]}
    cfg.update(protocol="matched_20261112", model=model, subset=subset, seed=seed,
               data_dir=str(Path(data_dir).resolve()),
               window=PROTOCOL["window_choices"][model][fd], window_tuning_trials=0,
               execution_manifest_sha256=verify_sources(), protocol_sha256=digest(ROOT / "PROTOCOL.json"),
               snapshot_manifest_sha256=digest(ROOT / "SNAPSHOT_MANIFEST.json"),
               training_windows="all_cycle_left_padded", checkpoint_selection="val_engine_mean_rmse",
               tf32=False, training_data_sha256={subset: digest(Path(data_dir) / f"train_{subset}.txt")})
    if model in REFERENCES:
        from .reference.training.train import DEFAULTS
        native = {**DEFAULTS, **PROTOCOL["reference_configs"][model], "seed": seed, "subset": subset,
                  "tag": model, "data_dir": str(Path(data_dir).resolve()), "span": cfg["window"]}
        cfg.update(native_config=native, source={"model": model, "snapshot_manifest_sha256": cfg["snapshot_manifest_sha256"]},
                   batch=native["batch"], lr=native["lr"], weight_decay=native["weight_decay"])
    else:
        cfg.update(source=source_record(model), batch={"aidgn":128,"two_phase":256,"dual_mixer":204,"ffve":128,"fcstgnn":100}[model],
                   lr=(0.005 if subset=="FD001" else 0.001) if model=="ffve" else (0.0001 if model=="two_phase" and subset in ["FD002","FD004"] else 0.001),
                   weight_decay=0.01 if model=="ffve" else 0.0)
        if model == "two_phase":
            pair = {"FD001":"FD003","FD003":"FD001","FD002":"FD004","FD004":"FD002"}[subset]
            cfg.update(paired_pretraining_subset=pair, pretraining_epochs=20 if subset in ["FD001","FD003"] else 100,
                       seed_policy="whole-pipeline FT-before-PRE; no phase reseed",
                       extra_data_condition="paired subset training engines; both validation sets excluded")
            cfg["training_data_sha256"][pair] = digest(Path(data_dir) / f"train_{pair}.txt")
        if model == "ffve":
            cfg.update(ffve_blocks=2 if subset in ["FD002","FD004"] else 1, ffve_long_residual=subset=="FD004")
    verify_training_data(cfg, data_dir)
    return cfg
