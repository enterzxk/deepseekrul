"""One full-budget job. No early stopping and no official-test file access."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import pandas as pd
import torch

from .data import prepare_fold, load_cache, save_cache, sha256
from .model import ARMS, ContrastPE, training_loss
from .metrics import engine_statistics, summarize

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INPUT_KEYS = ("history", "cycles", "conditions", "observed", "as_of", "extras", "prior")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def code_fingerprint():
    h = hashlib.sha256()
    for p in sorted((PACKAGE_ROOT/"tim_contrast").rglob("*.py")):
        h.update(str(p.relative_to(PACKAGE_ROOT)).replace("\\", "/").encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    h.update((PACKAGE_ROOT/"protocol.json").read_bytes())
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(arm, seed, use_prior):
    # Seed BEFORE constructing both the reference and candidate. Copy shared
    # tensors so all equal-shaped PE parameters are paired across architectures.
    seed_all(seed)
    canonical = ContrastPE("pe50", use_prior=use_prior)
    shared = canonical.state_dict()
    seed_all(seed)
    model = ContrastPE(arm, use_prior=use_prior)
    state = model.state_dict()
    for name, value in shared.items():
        if name in state and state[name].shape == value.shape:
            state[name] = value.clone()
    model.load_state_dict(state)
    if model.window == 100:
        with torch.no_grad():
            model.body.temporal_pe[:, -50:].copy_(canonical.body.temporal_pe)
    seed_all(seed)
    return model


def write_json(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix+".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(p)


def run(args):
    protocol = json.loads((PACKAGE_ROOT/"protocol.json").read_text(encoding="utf-8"))
    # Epoch count is intentionally not a CLI hyperparameter.
    epochs, batch_size = protocol["epochs"], protocol["batch_size"]
    if args.nasa_weight not in (0.0, protocol["rescue_nasa_weight"]):
        raise ValueError("Only the two preregistered losses are allowed")
    use_prior = args.prior == "full"
    label = args.arm + ("_risk" if args.nasa_weight else "")
    dest = Path(args.out)/args.subset/f"{label}_f{args.fold}_s{args.seed}_{args.prior}"
    dest.mkdir(parents=True, exist_ok=True)
    fingerprint = code_fingerprint()
    raw_sha = sha256(Path(args.data_root)/f"train_{args.subset}.txt")
    if (dest/"done.json").exists():
        done = json.loads((dest/"done.json").read_text(encoding="utf-8"))
        if done["code_sha256"] != fingerprint or done["train_sha256"] != raw_sha:
            raise RuntimeError("Completed run belongs to different code/data; use a new output directory")
        for name in ("predictions.csv", "engine_metrics.csv", "metrics.json", "checkpoint.pt"):
            if not (dest/name).is_file() or sha256(dest/name) != done["artifacts"][name]:
                raise RuntimeError(f"Completed artifact is missing or changed: {dest/name}")
        return
    lock = dest/"RUNNING.json"
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": utc_now()}, f)
    started, started_utc = time.monotonic(), utc_now()
    try:
        device = torch.device(args.device)
        cache_key = hashlib.sha256((fingerprint+raw_sha+args.subset+str(args.fold)+args.prior).encode()).hexdigest()[:20]
        cache = Path(args.cache)/f"{args.subset}_f{args.fold}_{cache_key}.npz"
        if not cache.is_file():
            a, b, meta = prepare_fold(args.data_root, args.subset, args.fold, use_prior)
            save_cache(cache, a, b, meta)
        a, b, meta = load_cache(cache)
        if meta["train_sha256"] != raw_sha:
            raise RuntimeError("Cache/raw-data mismatch")
        train_select = a["as_of"] >= (100 if args.arm == "w100_late" else 50)
        ta = {k: torch.as_tensor(v[train_select], device=device) for k, v in a.items()}
        tb = {k: torch.as_tensor(v, device=device) for k, v in b.items()}
        model = build_model(args.arm, args.seed, use_prior).to(device)
        seed_all(args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["lr"],
                                      weight_decay=protocol["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                               eta_min=protocol["min_lr"])
        generator = torch.Generator(device=device).manual_seed(args.seed)
        n = len(ta["target"])
        epoch_records = []
        for epoch in range(1, epochs+1):
            model.train()
            loss_sum = 0.0
            for index in torch.randperm(n, device=device, generator=generator).split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                pred = model(*(ta[k][index] for k in INPUT_KEYS))
                loss = training_loss(pred, ta["target"][index], args.nasa_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch}; no queries dropped")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                loss_sum += loss.detach().item()*len(index)
            scheduler.step()
            epoch_records.append({"epoch": epoch, "loss": loss_sum/n})
            if epoch % 10 == 0 or epoch == epochs:
                write_json(dest/"progress.json", {"epoch": epoch, "epochs": epochs,
                           "loss": loss_sum/n, "seconds": time.monotonic()-started})
                print(f"{args.subset} {label} f{args.fold} s{args.seed}: {epoch}/{epochs}", flush=True)
        model.eval()
        with torch.no_grad():
            predictions = torch.cat([model(*(tb[k][index] for k in INPUT_KEYS))
                           for index in torch.arange(len(b["target"]), device=device).split(batch_size)])
        result = pd.DataFrame({"unit": b["unit"], "end_cycle": b["as_of"],
                               "y_true": b["target"], "y_pred": predictions.cpu().numpy(),
                               "fold": args.fold, "seed": args.seed})
        stats = engine_statistics(result)
        result.to_csv(dest/"predictions.csv", index=False)
        stats.to_csv(dest/"engine_metrics.csv")
        torch.save({"state_dict": model.cpu().state_dict(), "arm": args.arm,
                    "use_prior": use_prior, "seed": args.seed, "epoch": epochs,
                    "code_sha256": fingerprint, "data_metadata": meta}, dest/"checkpoint.pt")
        metrics = {**summarize(stats), **meta, "arm": args.arm, "label": label,
                   "nasa_weight": args.nasa_weight, "seed": args.seed, "epochs": epochs,
                   "n_training_queries": n, "parameters": sum(p.numel() for p in model.parameters()),
                   "elapsed_seconds": time.monotonic()-started, "started": started_utc,
                   "completed": utc_now(), "code_sha256": fingerprint,
                   "prediction_clipping": False, "epoch_selection": "fixed_final",
                   "torch_version": torch.__version__, "train_history": epoch_records}
        write_json(dest/"metrics.json", metrics)
        write_json(dest/"done.json", {"code_sha256": fingerprint, "train_sha256": raw_sha,
                   "completed": utc_now(), "artifacts": {name: sha256(dest/name) for name in
                   ("predictions.csv", "engine_metrics.csv", "metrics.json", "checkpoint.pt")}})
        print(f"DONE {args.subset} {label} f{args.fold} s{args.seed}: RMSE={metrics['rmse']:.4f}", flush=True)
    except Exception as exc:
        write_json(dest/"failed.json", {"type": type(exc).__name__, "message": str(exc),
                                      "time": utc_now(), "code_sha256": fingerprint})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--cache", default="cache")
    p.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), required=True)
    p.add_argument("--fold", type=int, choices=range(5), required=True)
    p.add_argument("--seed", type=int, choices=(101, 202, 303), default=101)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--prior", choices=("full", "none"), default="full")
    p.add_argument("--nasa-weight", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())


if __name__ == "__main__":
    main()
