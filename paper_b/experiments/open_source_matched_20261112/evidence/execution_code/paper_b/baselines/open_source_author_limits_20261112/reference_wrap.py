"""Call the unchanged frozen DSPT trainer under the preregistered FP32 wrapper."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import torch
from .common import ROOT, fp32_autocast, predictions, state_hash, validation_record, write_json
from .reference.training import train as native
from .reference.training import data as native_data


def train_reference(cfg, data_dir, out):
    for function in [native.train_one, native._build, native.predict, native.prepare_train, native_data._prepare_split]:
        assert Path(function.__code__.co_filename).resolve().is_relative_to(ROOT / "reference")
    assert native.build_dspt.__module__.startswith(__package__ + ".reference.")
    original = native.predict
    trace, captured = [], {}
    def traced(net, data, *args, **kwargs):
        p = original(net, data, *args, **kwargs)
        trace.append({**validation_record(data, p), "state_content_sha256": state_hash(net.state_dict())})
        captured.update(net=net, data=data, prediction=p)
        return p
    native.predict = traced
    try:
        with fp32_autocast():
            m = native.train_one(cfg["native_config"], device="cuda:0", out_root=out / "native")
    finally:
        native.predict = original
    source = out / "native" / cfg["model"] / cfg["subset"] / f"s{cfg['seed']}"
    history = json.loads((source / "history.json").read_text())
    assert len(trace) == len(history) + 1
    best, best_ep = float("inf"), 0
    for row, check in zip(history, trace):
        assert abs(row["val_rmse"] - check["val_engine_mean_rmse"]) < 1e-8
        if check["val_engine_mean_rmse"] < best - cfg["min_delta"]:
            best, best_ep = check["val_engine_mean_rmse"], row["epoch"]
        check.update(epoch=row["epoch"], selected_epoch_so_far=best_ep)
    assert best_ep == m["best_ep"] and abs(best - m["val_engine_mean_rmse"]) < 1e-8
    net, va, pv = captured["net"], captured["data"], captured["prediction"]
    assert state_hash(net.state_dict()) == trace[best_ep-1]["state_content_sha256"]
    write_json(out / "history.json", trace[:-1])
    write_json(out / "native_validation_metrics.json", m)
    shutil.copy2(source / "config.json", out / "native_config.json")
    shutil.copy2(source / "model.pt", out / "checkpoint.pt")
    predictions(out / "validation_predictions.csv", va["y"], pv, va["unit"], va["cut"])
    data = native_data.prepare_train(cfg["subset"], Path(data_dir), span=cfg["window"],
                                    cuts=tuple(cfg["cuts"]), split_seed=cfg["split_seed"], fold=cfg["fold"],
                                    val_frac=cfg["val_frac"], rul_cap=cfg["rul_cap"], n_cond=cfg["n_cond"])
    assert len(data["train"]["y"]) == len(data["tr"])
    return net, data, va, pv, trace[:-1], best_ep


def predict_reference(net, data, cfg):
    with fp32_autocast():
        return native.predict(net, data, torch.device("cuda:0"), cfg["native_config"]["mode"])


def load_reference(cfg, checkpoint):
    net = native._build(cfg["native_config"]).cuda()
    net.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    return net
