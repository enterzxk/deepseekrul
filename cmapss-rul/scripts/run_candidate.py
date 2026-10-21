"""Stage-2/3 candidate tournament runner (validation-only, no test access).

Trains one candidate configuration on the 80/20 development split of FD001 or
FD002 and writes validation metrics. The official test split is never built,
loaded or evaluated here - Stage 2/3 is strictly validation-only by design.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS,
    OperatingConditionScaler,
    SensorMinMaxScaler,
    build_piecewise_linear_rul,
    compute_rmse,
    load_cmapss_train,
)
from candidates.model import CandidateModel, ModelEMA, soft_nasa_loss  # noqa: E402
from baselines import build_baseline  # noqa: E402

CYCLE_SCALE = 250.0

CONFIGS = {
    "base": dict(use_baseline=False, use_ema_features=False, use_cycle=False,
                 use_graph=False, use_dual_head=False, use_model_ema=False,
                 cls_weight=0.0, nasa_weight=0.0),
    "B": dict(use_baseline=True, use_ema_features=False, use_cycle=False,
              use_graph=False, use_dual_head=False, use_model_ema=False,
              cls_weight=0.0, nasa_weight=0.0),
    "E": dict(use_baseline=False, use_ema_features=True, use_cycle=False,
              use_graph=False, use_dual_head=False, use_model_ema=False,
              cls_weight=0.0, nasa_weight=0.0),
    "C": dict(use_baseline=False, use_ema_features=False, use_cycle=True,
              use_graph=False, use_dual_head=False, use_model_ema=False,
              cls_weight=0.0, nasa_weight=0.0),
    "BE": dict(use_baseline=True, use_ema_features=True, use_cycle=False,
               use_graph=False, use_dual_head=False, use_model_ema=False,
               cls_weight=0.0, nasa_weight=0.0),
    "BC": dict(use_baseline=True, use_ema_features=False, use_cycle=True,
               use_graph=False, use_dual_head=False, use_model_ema=False,
               cls_weight=0.0, nasa_weight=0.0),
    "EC": dict(use_baseline=False, use_ema_features=True, use_cycle=True,
               use_graph=False, use_dual_head=False, use_model_ema=False,
               cls_weight=0.0, nasa_weight=0.0),
    "A": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
              use_graph=False, use_dual_head=False, use_model_ema=False,
              cls_weight=0.0, nasa_weight=0.0),
    "AB": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
               use_graph=True, use_dual_head=False, use_model_ema=False,
               cls_weight=0.0, nasa_weight=0.0),
    "ABC": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                use_graph=True, use_dual_head=True, use_model_ema=False,
                cls_weight=0.2, nasa_weight=0.05),
    "ABE": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                use_graph=True, use_dual_head=False, use_model_ema=True,
                cls_weight=0.0, nasa_weight=0.0),
    "ABCE": dict(use_baseline=True, use_ema_features=True, use_cycle=True,
                 use_graph=True, use_dual_head=True, use_model_ema=True,
                 cls_weight=0.2, nasa_weight=0.05),
    "tf": dict(baseline="transformer", use_baseline=False, use_ema_features=False,
               use_cycle=False, use_graph=False, use_dual_head=False,
               use_model_ema=False, cls_weight=0.0, nasa_weight=0.0),
}


class WindowSet(Dataset):
    def __init__(self, x, y, cond, extras):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.cond = torch.from_numpy(cond) if cond is not None else None
        self.extras = torch.from_numpy(extras)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        cond = self.cond[index] if self.cond is not None else torch.zeros(50, 3)
        return self.x[index], self.y[index], cond, self.extras[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_windows(frame, values, window, stride, max_rul, test_mode=False, cutoff=None):
    """Windows plus (per-sensor healthy baseline, fixed-scale cycle coordinate).

    Fallbacks, all computable from observations available online:
      * baseline uses the first min(5, len) observed cycles of that engine;
      * if an engine has fewer than 5 cycles the available ones are used, with
        a global early-life mean covering any engine without an early segment;
      * the cycle coordinate divides by a FIXED constant (250) and never by a
        test engine's true lifetime.
    """
    xs, ys, extras = [], [], []
    global_baseline = np.asarray(values, dtype=np.float32)[: 5 * 20].mean(axis=0)
    for _, group in frame.groupby("unit", sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        early = seq[: min(5, seq.shape[0])]
        baseline = early.mean(axis=0) if early.shape[0] > 0 else global_baseline
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        if cutoff is not None:
            # Official-test-style evaluation: only history up to the cut is
            # visible, one final window per engine.
            end = max(window, int(round(seq.shape[0] * cutoff)))
            visible = seq[:end]
            rul_at_end = build_piecewise_linear_rul(group, max_rul=max_rul)[end - 1]
            xs.append(np.ascontiguousarray(visible[-window:]))
            ys.append(float(rul_at_end))
            extras.append(np.concatenate([baseline, [min(cycles[end - 1] / CYCLE_SCALE, 1.0)]]))
        elif test_mode:
            if seq.shape[0] < window:
                pad = np.repeat(seq[:1], window - seq.shape[0], axis=0)
                seq = np.concatenate([pad, seq], axis=0)
            xs.append(np.ascontiguousarray(seq[-window:]))
            extras.append(np.concatenate([baseline, [min(cycles[-1] / CYCLE_SCALE, 1.0)]]))
            ys.append(0.0)
        else:
            rul = build_piecewise_linear_rul(group, max_rul=max_rul)
            start = 0
            for end in range(window - 1, seq.shape[0], stride):
                xs.append(np.ascontiguousarray(seq[end - window + 1 : end + 1]))
                ys.append(float(rul[end]))
                extras.append(np.concatenate([baseline, [min(cycles[end] / CYCLE_SCALE, 1.0)]]))
    return (np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32),
            np.stack(extras).astype(np.float32))


def prepare(subset: str, seed: int, window: int, max_rul: float, condition_aware: bool, n_cond: int):
    # Development path: training file only. The official test/RUL files are
    # never opened here (see dataset.load_cmapss_train).
    raw_train = load_cmapss_train(ROOT / "data" / "CMAPSS", subset)
    units = np.sort(raw_train["unit"].unique())
    rng = np.random.default_rng(seed + 20261020)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(units.size * 0.2)))
    val_units = sorted(int(x) for x in shuffled[:n_val])
    train_units = sorted(int(x) for x in shuffled[n_val:])
    train_frame = raw_train[raw_train.unit.isin(train_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    val_frame = raw_train[raw_train.unit.isin(val_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    cols = list(SELECTED_SENSOR_COLUMNS)
    if condition_aware:
        op = ["op_1", "op_2", "op_3"]
        mean = train_frame[op].to_numpy(dtype=np.float64).mean(axis=0)
        std = train_frame[op].to_numpy(dtype=np.float64).std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        cond_t = ((train_frame[op].to_numpy(np.float64) - mean) / std).astype(np.float32)
        cond_v = ((val_frame[op].to_numpy(np.float64) - mean) / std).astype(np.float32)
        scaler = OperatingConditionScaler(cols, n_clusters=n_cond, random_state=seed)
        tr_s, va_s = scaler.fit_transform(train_frame), scaler.transform(val_frame)
    else:
        cond_t = cond_v = None
        scaler = SensorMinMaxScaler(cols)
        tr_s = scaler.fit_transform(train_frame[cols].to_numpy(np.float64))
        va_s = scaler.transform(val_frame[cols].to_numpy(np.float64))
    xtr, ytr, etr = build_windows(train_frame, tr_s, window, 1, max_rul)
    xva, yva, eva = build_windows(val_frame, va_s, window, 1, max_rul)
    xpt, ypt, ept = build_windows(val_frame, va_s, window, 1, max_rul, cutoff=0.7)
    ctr = build_cond(train_frame, cond_t, window) if cond_t is not None else None
    cva = build_cond(val_frame, cond_v, window) if cond_v is not None else None
    cpt = build_cond_cut(val_frame, cond_v, window, 0.7) if cond_v is not None else None
    return (WindowSet(xtr, ytr, ctr, etr), WindowSet(xva, yva, cva, eva),
            train_units, val_units, WindowSet(xpt, ypt, cpt, ept))


def build_cond_cut(frame, values, window, cutoff):
    out = []
    for _, group in frame.groupby("unit", sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        end = max(window, int(round(seq.shape[0] * cutoff)))
        out.append(np.ascontiguousarray(seq[end - window : end]))
    return np.stack(out).astype(np.float32)


def build_cond(frame, values, window):
    out = []
    for _, group in frame.groupby("unit", sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        for end in range(window - 1, seq.shape[0]):
            out.append(seq[end - window + 1 : end + 1])
    return np.stack(out).astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device, baseline: bool = False):
    model.eval()
    preds, targets = [], []
    for x, y, cond, extras in loader:
        if baseline:
            out = model(x.to(device))
            pred = out
        else:
            out = model(x.to(device), cond=cond.to(device), extras=extras.to(device))
            pred = out["prediction"]
        preds.append(pred.reshape(-1).cpu().numpy())
        targets.append(y.numpy().reshape(-1))
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    return float(compute_rmse(t, p)), p, t


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--out", default="experiments/tournament")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = CONFIGS[args.variant]
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set, val_set, train_units, val_units, pseudo_set = prepare(
        args.subset, args.seed, args.window, 125.0,
        condition_aware=args.subset in {"FD002", "FD004"}, n_cond=6,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2)
    is_baseline = "baseline" in cfg
    if is_baseline:
        model = build_baseline(cfg["baseline"], num_sensors=14, d_model=88,
                               dropout=args.dropout).to(device)
    else:
        model = CandidateModel(
            num_sensors=14, window_size=args.window, d_model=args.d_model,
            num_layers=args.num_layers, dropout=args.dropout,
            use_baseline=cfg["use_baseline"], use_ema_features=cfg["use_ema_features"],
            use_cycle=cfg["use_cycle"], use_graph=cfg["use_graph"], use_dual_head=cfg["use_dual_head"],
        ).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)
    ema = ModelEMA(model, 0.995) if cfg["use_model_ema"] else None
    bce = nn.BCEWithLogitsLoss()

    best, best_epoch, best_state = float("inf"), -1, None
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y, cond, extras in train_loader:
            x, y, cond, extras = x.to(device), y.to(device), cond.to(device), extras.to(device)
            optimiser.zero_grad(set_to_none=True)
            if is_baseline:
                pred = model(x)
                out = {"prediction": pred}
            else:
                out = model(x, cond=cond, extras=extras)
                pred = out["prediction"]
            y = y.reshape(-1, 1)
            delta = pred - y
            loss = nn.functional.mse_loss(pred, y)
            late = torch.where(delta > 0, 2.0 * delta.square(), delta.square())
            loss = loss + 0.5 * late.mean()
            if cfg["cls_weight"] > 0 and "logit" in out:
                label = (y < 124.5).float()
                loss = loss + cfg["cls_weight"] * bce(out["logit"], label)
            if cfg["nasa_weight"] > 0:
                loss = loss + cfg["nasa_weight"] * soft_nasa_loss(delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if ema is not None:
                ema.update(model)
            total += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        if ema is not None:
            ema.store_and_apply(model)
        val_rmse, _, _ = evaluate(model, val_loader, device, baseline=is_baseline)
        # When EMA is active, the evaluated weights are the EMA weights, so the
        # selected checkpoint must be the EMA state too - never the raw model.
        eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema is not None:
            ema.restore(model)

        history.append({"epoch": epoch, "train_loss": total / max(seen, 1), "val_rmse": val_rmse})
        print(f"[{args.subset} {args.variant} s{args.seed}] ep{epoch:03d} loss={total/max(seen,1):.3f} val={val_rmse:.4f}", flush=True)
        if val_rmse < best - 1e-4:
            best, best_epoch = val_rmse, epoch
            best_state = eval_state
        elif epoch - best_epoch >= args.patience:
            break

    pt_loader = DataLoader(pseudo_set, batch_size=256, shuffle=False, num_workers=2)
    pt_rmse, _, _ = evaluate(model, pt_loader, device, baseline=is_baseline)

    out_dir = ROOT / args.out / args.subset / f"{args.variant}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({
        "subset": args.subset, "variant": args.variant, "seed": args.seed,
        "config": cfg, "params": params, "best_val_rmse": best,
        "best_epoch": best_epoch, "epochs_ran": len(history),
        "pseudo_test_rmse": pt_rmse,
        "pseudo_test_protocol": "validation engines cut at 70% of observed life, "
        "single final window, targets clipped at 125",
        "val_units": val_units, "train_units_count": len(train_units),
        "elapsed_seconds": time.time() - start,
        "split": "80/20 engine-level development split; test set never loaded",
    }, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"BEST {args.variant} {args.subset} s{args.seed} val={best:.4f} params={params:,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
