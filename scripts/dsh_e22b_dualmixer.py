"""E22b — Dual-Mixer under the project's unified protocol.

Protocol frozen in paper/E22B_DUALMIXER_PREREGISTRATION.md (sha in E22_LEDGER.md).

Kept from the original repository (fuen1590/PhmDeepLearningProjects):
  DualMLPMixer(hidden=32, layers=6, dropout=0), Adam lr=1e-3, MSE, batch 1024,
  100 epochs, early stop 5 on validation MSE, RUL target capped at 125.
Unified with this project:
  engine-level 5-fold split, 14 sensors with the subset-appropriate scaler, 50-cycle
  window, test = last observable window per engine (first-frame pad if <50 cycles),
  capped truth for the headline metric.

Usage:
  python dsh_e22b_dualmixer.py --subset FD001 --fold 2 --seed 2027
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = "/home/yndx/zxk/bearing-rul-pi"
REPO = f"{ROOT}/external/dualmixer_ref"
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{REPO}/models/RULPrediction")
sys.path.insert(0, REPO)  # ContrastiveModules imports `train.trainable`

import dsh_e21_test_all as e21  # frozen helpers: split_frames, fit_scaler, scale_frame
import run_candidate as rc  # frozen train-side window builders
from dataset import SELECTED_SENSOR_COLUMNS  # noqa: E402
from MLPMixer import DualMLPMixer  # noqa: E402  (original model class)

WINDOW = e21.WINDOW
MAX_RUL = e21.MAX_RUL
EPOCHS, BATCH, LR, PATIENCE = 100, 1024, 1e-3, 5


def windows_from_frame(scaled: np.ndarray, frame, cap=MAX_RUL):
    """Sliding 50-cycle windows per engine, stride 1, target = min(RUL, cap)."""
    xs, ys = [], []
    for _, g in frame.groupby("unit", sort=True):
        idx = g.index.to_numpy(np.int64)
        arr = scaled[idx]
        L = len(arr)
        for i in range(0, L - WINDOW + 1):
            xs.append(arr[i:i + WINDOW])
            ys.append(min(float(L - (i + WINDOW)), cap))
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32))


def last_window(scaled: np.ndarray, frame):
    xs, units = [], []
    order = np.argsort(frame["unit"].to_numpy())
    units_arr = frame["unit"].to_numpy()[order]
    pos = np.arange(len(frame))[order]
    bounds = np.flatnonzero(np.diff(units_arr)) + 1
    for idx in np.split(pos, bounds):
        arr = scaled[idx]
        units.append(int(units_arr[idx[0]]))
        if len(arr) < WINDOW:
            pad = np.repeat(arr[:1], WINDOW - len(arr), axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        xs.append(arr[-WINDOW:])
    return np.asarray(xs, dtype=np.float32), units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    raw, tr, va, _ = e21.split_frames(a.subset, a.fold)
    # E22b input convention (Amendment 1): per-channel z-score from the fold's training
    # engines. The min-max scaler that the candidate arms use leaves near-constant
    # channels as amplifier noise; with the mixer's two sigmoid gates per layer the
    # signal dies (val RMSE 43 vs 15 for the same model on z-scored input).
    cols = list(SELECTED_SENSOR_COLUMNS)
    mu = tr[cols].mean().to_numpy()
    sd = tr[cols].std().to_numpy().clip(1e-6)
    tr_s = (tr[cols].to_numpy() - mu) / sd
    va_s = (va[cols].to_numpy() - mu) / sd

    xtr, ytr = windows_from_frame(tr_s, tr)
    # validation follows the project's matched protocol (deterministic cutoff seed), so
    # the early-stopping signal is comparable with every other run in the project
    xva, yva, _, units_va = rc.build_windows_matched(va, va_s, WINDOW, MAX_RUL)
    # fixed-scale labels (y/125): the original repository trains on normalised labels;
    # with raw cycles a lr of 1e-3 gives an effective step ~4 orders of magnitude too
    # small for MSE. Fixed scaling (not per-engine) keeps test-time inversion exact.
    ytr = ytr / MAX_RUL
    yva = yva / MAX_RUL

    model = DualMLPMixer(window_size=WINDOW, in_features=len(SELECTED_SENSOR_COLUMNS),
                         hidden_dim=32, num_layers=6, dropout=0.0, device=device,
                         model_flag="dualmixer-e22b", label_norm=False, filter_size=0)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    xt = torch.from_numpy(xtr); yt = torch.from_numpy(ytr).unsqueeze(-1)
    xv = torch.from_numpy(xva).to(device); yv = torch.from_numpy(yva).to(device).unsqueeze(-1)
    n = len(xt)
    best, best_state, patience = float("inf"), None, 0
    history = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = xt[idx].to(device); yb = yt[idx].to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vp = model(xv)
            vloss = float(loss_fn(vp, yv))
        vrmse = float(torch.sqrt(torch.mean((vp - yv) ** 2))) * MAX_RUL
        history.append(dict(epoch=ep, train_mse=tot / n, val_mse=vloss, val_rmse=vrmse))
        if vloss < best - 1e-6:
            best, patience = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"early stop at epoch {ep}", flush=True)
                break
        print(f"[{a.subset} f{a.fold} s{a.seed}] ep{ep:03d} "
              f"train_mse={tot/n:.3f} val_rmse={vrmse:.3f}", flush=True)

    out_dir = f"{ROOT}/experiments/e22b/{a.subset}/f{a.fold}_s{a.seed}"
    os.makedirs(out_dir, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, f"{out_dir}/best_model.pt")
    json.dump(dict(subset=a.subset, fold=a.fold, seed=a.seed, num_params=n_params,
                   best_val_mse=best,
                   best_val_rmse=min(h["val_rmse"] for h in history),
                   epochs_ran=len(history), elapsed_seconds=time.time() - t0,
                   config=dict(window=WINDOW, hidden=32, layers=6, dropout=0.0, lr=LR,
                               batch=BATCH, epochs=EPOCHS, early_stop=PATIENCE,
                               label_cap=MAX_RUL, source="fuen1590/PhmDeepLearningProjects",
                               unified="engine-level fold, 14 sensors, 50-window"),
                   history=history,
                   finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds")),
              open(f"{out_dir}/metrics.json", "w"), indent=2)
    print(f"saved {out_dir} params={n_params}")


if __name__ == "__main__":
    main()
