"""Complete upstream methods under the frozen Paper B holdout protocol."""
from __future__ import annotations
import argparse
import builtins
import io
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .adapters import UpstreamAdapter, module
from .common import ROOT, PROJECT, config_hash, digest, metrics, predictions, research_task_start, seed_everything, source_record, write_json
from .data import prepare


def configuration(model, subset, seed):
    window = {"aidgn": 50, "two_phase": 45, "dual_mixer": 30, "ffve": 30,
              "fcstgnn": 30 if subset == "FD001" else 50}[model]
    cfg = {"protocol": "unified", "model": model, "subset": subset, "seed": seed,
           "window": window, "split_seed": 7, "fold": 0, "val_frac": 0.2, "rul_cap": 125,
           "cuts": [0.4, 0.55, 0.7, 0.85, 0.95], "source": source_record(model),
           "window_search": [window], "window_tuning_trials": 0, "precision": "float32",
           "epochs": {"aidgn": 300, "two_phase": 200 if subset in ("FD001", "FD003") else 600,
                      "dual_mixer": 100, "ffve": 100000, "fcstgnn": 41}[model],
           "patience": 10 if model == "ffve" else 12,
           "batch": {"aidgn": 128, "two_phase": 256, "dual_mixer": 204, "ffve": 128, "fcstgnn": 100}[model],
           "lr": (0.005 if subset == "FD001" else 0.001) if model == "ffve" else (0.0001 if model == "two_phase" and subset in ("FD002", "FD004") else 0.001),
           "weight_decay": 0.01 if model == "ffve" else 0.0,
           "code_sha256": {name: digest(ROOT / name) for name in ["unified.py", "adapters.py", "data.py", "common.py"]}}
    if model == "two_phase":
        cfg["paired_pretraining_subset"] = {"FD001": "FD003", "FD003": "FD001", "FD002": "FD004", "FD004": "FD002"}[subset]
        cfg["pretraining_epochs"] = 20 if subset in ("FD001", "FD003") else 100
        cfg["pretraining_data"] = "80% training engines in both paired subsets; no validation/test engines"
    if model == "ffve":
        cfg["ffve_blocks"] = 2 if subset in ("FD002", "FD004") else 1
        cfg["ffve_long_residual"] = subset == "FD004"
    return cfg


def predict(net, data, seed, batch=256):
    net.eval()
    outputs = []
    # FFVE uses the actual variational sample, never a latent-mean shortcut.
    # Fixed evaluation seed gives repeatable single-draw validation/test.
    with torch.random.fork_rng(devices=[0]), torch.no_grad():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for start in range(0, len(data["y"]), batch):
            x = torch.from_numpy(data["x"][start:start + batch]).cuda()
            age = torch.from_numpy(data["age"][start:start + batch]).cuda()
            outputs.append(net(x, age).cpu().numpy())
    return np.concatenate(outputs)


def pretrain_two_phase(net, cfg, data, data_dir, out):
    paired = prepare(cfg["paired_pretraining_subset"], data_dir, cfg)
    write_json(out / "paired_split.json", {k: paired[k] for k in ["train_units", "val_units", "scaler"]})
    x = np.concatenate([data["train"]["x"], paired["train"]["x"]])
    y = np.concatenate([data["train"]["y"], paired["train"]["y"]])
    domain = np.concatenate([np.ones(len(data["train"]["y"]), np.int64), np.zeros(len(paired["train"]["y"]), np.int64)])
    # Upstream corruption is fixed once before training, with random seed 529.
    import random
    rng = random.Random(529)
    mask_p, random_p = (0.6, 0.3) if cfg["pretraining_epochs"] == 20 else (0.3, 0.6)
    for window in x:
        value = rng.random()
        if value < mask_p:
            window[-1] = 0
        elif value < mask_p + random_p:
            window[-1] = [rng.random() for _ in range(x.shape[-1])]
            window[-1, 1] = 0
    layers, dropout = (1, 0.1) if cfg["pretraining_epochs"] == 20 else (2, 0.0)
    pre = net.pre_module.Transformer_pre(17, 16, 45, 32, 32, layers, 2, dropout).cuda()
    opt = torch.optim.Adam(pre.parameters(), lr=0.001)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(domain))
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    history = []
    for epoch in range(1, cfg["pretraining_epochs"] + 1):
        pre.train()
        loss_sum = 0.0
        for bx, by, bd in loader:
            bx, by, bd = bx.cuda(), by.cuda(), bd.cuda()
            opt.zero_grad(set_to_none=True)
            regression, classification = pre(bx)
            loss = torch.sqrt(torch.mean((regression.reshape(-1) - by) ** 2)) + torch.nn.functional.cross_entropy(classification, bd)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach())
        history.append({"epoch": epoch, "loss": loss_sum / len(loader)})
        print(f"PRETRAIN ep={epoch} loss={history[-1]['loss']:.6f}", flush=True)
    torch.save(pre.state_dict(), out / "pretrain_checkpoint.pt")
    write_json(out / "pretrain_history.json", history)
    transfer = net.net.encoder.load_state_dict(pre.encoder.state_dict(), strict=True)
    assert not transfer.missing_keys and not transfer.unexpected_keys
    for parameter in net.net.encoder.parameters():
        parameter.requires_grad_(False)
    del pre
    torch.cuda.empty_cache()


def run(model, subset, seed, data_dir, out_root):
    cfg = configuration(model, subset, seed)
    cfg["training_data_sha256"] = {subset: digest(Path(data_dir) / f"train_{subset}.txt")}
    if model == "two_phase":
        paired = cfg["paired_pretraining_subset"]
        cfg["training_data_sha256"][paired] = digest(Path(data_dir) / f"train_{paired}.txt")
    out = Path(out_root).resolve() / "unified" / model / subset / f"s{seed}"
    research_task_start(out, cfg)
    started = time.time()
    seed_everything(seed)
    original_open, original_io_open = builtins.open, io.open
    access = {"official_test_unlocked": False, "official_test_reads": [], "violations": []}

    def protected_open(delegate):
        def opened(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                name = Path(file.decode() if isinstance(file, bytes) else file).name
                if name.startswith(("test_FD00", "RUL_FD00")):
                    if not access["official_test_unlocked"]:
                        access["violations"].append(str(file))
                        raise PermissionError("official test access before checkpoint freeze")
                    access["official_test_reads"].append({"path": str(file), "unix": time.time()})
            return delegate(file, *args, **kwargs)
        return opened

    builtins.open, io.open = protected_open(original_open), protected_open(original_io_open)
    try:
        data = prepare(subset, Path(data_dir), cfg)
        write_json(out / "split.json", {k: data[k] for k in ["train_units", "val_units"]})
        write_json(out / "scaler.json", data["scaler"])
        net = UpstreamAdapter(cfg).cuda()
        write_json(out / "environment.json", {"python": sys.version, "torch": torch.__version__,
                                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
        if model == "two_phase":
            pretrain_two_phase(net, cfg, data, Path(data_dir), out)
        tr, va = data["train"], data["val"]
        if model == "dual_mixer":
            cmapss = module("dual_mixer", "dataset/cmapss.py", "upstream_cmapss")
            dataset = cmapss.Cmapss(tr["x"], tr["unit"], tr["y"] / 125.0)
            cmapss.CmapssGaussianNegativeSampler(dataset, neg_num=5, thresh=0.5, std=0.3)
        else:
            dataset = TensorDataset(torch.from_numpy(tr["x"]), torch.from_numpy(tr["y"]), torch.from_numpy(tr["age"]))
        loader = DataLoader(dataset, batch_size=cfg["batch"], shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                     lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = None
        if model == "ffve":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        if model == "aidgn":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10)
        best, best_epoch, history = float("inf"), 0, []
        for epoch in range(1, cfg["epochs"] + 1):
            net.train()
            total = 0.0
            for batch in loader:
                x, y = batch[0].cuda().float(), batch[1].cuda().float()
                age = batch[2].cuda() if model != "dual_mixer" else None
                optimizer.zero_grad(set_to_none=True)
                loss = net.loss(x, y, age)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            pv = predict(net, va, seed + 100000)
            score = metrics(va["y"], pv)["rmse"]
            row = {"epoch": epoch, "train_loss": total / len(loader), "val_rmse": score,
                   "elapsed_s": time.time() - started, "lr": optimizer.param_groups[0]["lr"]}
            history.append(row)
            print("TRAIN", json.dumps(row), flush=True)
            if score < best:
                best, best_epoch = score, epoch
                torch.save(net.state_dict(), out / "checkpoint.pt")
            if scheduler is not None:
                scheduler.step(score / net.output_scale ** 2) if model == "aidgn" else scheduler.step()
            if model != "fcstgnn" and epoch - best_epoch >= cfg["patience"]:
                break
        import pandas as pd
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        net.load_state_dict(torch.load(out / "checkpoint.pt", map_location="cuda", weights_only=True))
        pv = predict(net, va, seed + 100000)
        predictions(out / "validation_predictions.csv", va["y"], pv, va["unit"], va["cut"])
        write_json(out / "validation_metrics.json", metrics(va["y"], pv))
        elapsed = time.time() - started
        freeze = {"frozen": True, "selection": "minimum pooled engine-cut validation RMSE",
                  "best_epoch": best_epoch, "config_sha256": config_hash(cfg),
                  "checkpoint_sha256": digest(out / "checkpoint.pt"),
                  "source": cfg["source"], "created_unix": time.time(), "test_evaluations": 1}
        write_json(out / "FREEZE.json", freeze)
        access["official_test_unlocked"] = True
        # The only official test opening on this execution path occurs here,
        # after the exact checkpoint and configuration have been frozen.
        test = prepare(subset, Path(data_dir), cfg, official_test=True)["test"]
        assert len(test["y"]) == {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}[subset]
        pt = predict(net, test, seed + 200000)
        predictions(out / "test_predictions.csv", test["y"], pt, test["unit"], test["cut"])
        result = {**metrics(test["y"], pt), "protocol": "unified", "model": model,
                  "subset": subset, "seed": seed, "best_epoch": best_epoch,
                  "epochs_run": len(history), "validation_rmse": best,
                  "training_time_s": elapsed, "params": sum(p.numel() for p in net.parameters()),
                  "trainable_params": sum(p.numel() for p in net.parameters() if p.requires_grad),
                  "config_sha256": config_hash(cfg), "checkpoint_sha256": digest(out / "checkpoint.pt"),
                  "n_train_windows": len(tr["y"]), "n_validation_cuts": len(va["y"]),
                  "evaluation_unit": "official_test_engine", "evaluation_seed": seed + 200000}
        result["official_test_data_sha256"] = {name: digest(Path(data_dir) / f"{name}_{subset}.txt") for name in ["test", "RUL"]}
        write_json(out / "metrics.json", result)
        (out / "RUNNING.json").rename(out / "COMPLETED.json")
        print("UNIFIED_DONE", json.dumps(result), flush=True)
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (out / "RUNNING.json").rename(out / "FAILED.json")
        raise
    finally:
        builtins.open, io.open = original_open, original_io_open
        write_json(out / "data_access_audit.json", access)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    run(a.model, a.subset, a.seed, a.data_dir, a.out_root)
