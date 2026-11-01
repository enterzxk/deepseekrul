"""Train CET-RUL end to end on C-MAPSS development folds.

Unlike the earlier lines, the target is the **true failure time**: for a training engine
of lifetime L observed at cycle a, the remaining life is L - a - 1 cycles. The 125-cycle
cap is applied only when reporting benchmark RMSE, never inside the likelihood.

Usage:
  python cet_train.py --subset FD001 --fold 0 --seed 2027 --epochs 60
  python cet_train.py --subset FD001 --fold 0 --seed 2027 --ablation no_gate
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
import torch.nn.functional as F

ROOT = os.environ.get("RUL_ROOT", "/home/yndx/zxk/bearing-rul-pi")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")

import dsh_e21_test_all as e21                       # frozen fold split + scaler
from dataset import CMAPSS_COLUMNS
from cet.model import CETRUL, N_SENSORS, K_BINS

import pandas as pd

MAX_RUL_REPORT = 125.0      # benchmark convention only
STRIDE = 2                  # belief update every STRIDE cycles (speed)


def build_sequences(subset: str, fold: int, kind: str = "train"):
    """Per-engine sequences with true failure time; no future leakage in any input."""
    raw, tr, va, _ = e21.split_frames(subset, fold)
    frame = tr if kind == "train" else va
    scaler = e21.fit_scaler(subset, tr)          # always fit on TRAIN engines
    scaled = e21.scale_frame(subset, scaler, frame)

    seqs = []
    for unit, g in frame.groupby("unit", sort=True):
        idx = g.index.to_numpy()
        L = len(idx)
        if L < 20:
            continue
        xs = np.asarray(scaled[idx], dtype=np.float32)
        op = g[["op_1", "op_2", "op_3"]].to_numpy(np.float32) \
            if "op_1" in g.columns else np.zeros((L, 3), np.float32)
        age = g["cycle"].to_numpy(np.float32)
        mask = np.ones((L, N_SENSORS), np.float32)
        rul = (L - np.arange(1, L + 1)).astype(np.float32)      # remaining after this cycle
        seqs.append(dict(unit=int(unit), x=xs, op=op, age=age, mask=mask, rul=rul, L=L))
    return seqs


def collate(seqs, device):
    """Pad to the longest sequence in the batch; valid marks real cycles."""
    T = max(s["L"] for s in seqs)
    B = len(seqs)
    x = np.zeros((B, T, N_SENSORS), np.float32)
    op = np.zeros((B, T, 3), np.float32)
    age = np.zeros((B, T), np.float32)
    mask = np.zeros((B, T, N_SENSORS), np.float32)
    rul = np.zeros((B, T), np.float32)
    valid = np.zeros((B, T), np.float32)
    for i, s in enumerate(seqs):
        L = s["L"]
        x[i, :L] = s["x"]; op[i, :L] = s["op"]; age[i, :L] = s["age"]
        mask[i, :L] = s["mask"]; rul[i, :L] = s["rul"]; valid[i, :L] = 1.0
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(x), t(op), t(mask), t(age), t(rul), t(valid)


def make_weight(valid, rul, min_rul=0.0):
    """Loss weight: every valid cycle counts; cycles with RUL beyond the horizon are
    censored (handled in the NLL) rather than dropped."""
    return valid * (rul >= min_rul).float()


def evaluate(model, seqs, device, batch=32):
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            x, op, mask, age, rul, valid = collate(chunk, device)
            out = model(x, op, mask, age, valid)
            q = out["q"]
            pred = model.rul_mean(q)
            lo, hi = model.rul_interval(q)
            for b, s in enumerate(chunk):
                L = s["L"]
                v = valid[b].bool()
                rows.append(dict(
                    unit=s["unit"], L=L,
                    rul_true=s["rul"], rul_pred=pred[b][v].cpu().numpy(),
                    lo=lo[b][v].cpu().numpy(), hi=hi[b][v].cpu().numpy(),
                    gate=out["gate"][b][v].cpu().numpy(),
                    logE=out["logE"][b][v].cpu().numpy()))
    # aggregate metrics
    all_true = np.concatenate([r["rul_true"] for r in rows])
    all_pred = np.concatenate([r["rul_pred"] for r in rows])
    all_lo = np.concatenate([r["lo"] for r in rows])
    all_hi = np.concatenate([r["hi"] for r in rows])
    cap_true = np.minimum(all_true, MAX_RUL_REPORT)
    cap_pred = np.minimum(all_pred, MAX_RUL_REPORT)
    err = cap_pred - cap_true
    from math import sqrt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # last-cycle-only RMSE (benchmark convention: one prediction per engine)
    last_true = np.array([r["rul_true"][-1] for r in rows])
    last_pred = np.array([r["rul_pred"][-1] for r in rows])
    last_err = np.minimum(last_pred, 125) - np.minimum(last_true, 125)
    rmse_last = float(np.sqrt(np.mean(last_err ** 2)))
    cover = float(np.mean((all_lo <= all_true) & (all_true <= all_hi)))
    cover_last = float(np.mean([(r["lo"][-1] <= r["rul_true"][-1] <= r["hi"][-1]) for r in rows]))
    return dict(rmse_all=rmse, rmse_last=rmse_last, cover90_all=cover,
                cover90_last=cover_last, n_engines=len(rows),
                mean_gate=float(np.mean(np.concatenate([r["gate"] for r in rows]))),
                n_rows=len(all_true))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--ablation", default="full",
                    choices=["full", "no_gate", "gate_one", "no_update"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tr = build_sequences(a.subset, a.fold, "train")
    va = build_sequences(a.subset, a.fold, "val")
    print(f"[{a.subset} f{a.fold} s{a.seed}] train engines={len(tr)} val engines={len(va)} "
          f"T range {min(s['L'] for s in tr)}-{max(s['L'] for s in tr)}", flush=True)

    model = CETRUL(ablation=a.ablation).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params={n_par}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, best_state, bad, hist = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        order = np.random.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(order), a.batch):
            chunk = [tr[j] for j in order[i:i + a.batch]]
            x, op, mask, age, rul, valid = collate(chunk, device)
            out = model(x, op, mask, age, valid)
            w = make_weight(valid, rul)
            # subsample cycles to keep the sequential loop affordable
            if x.shape[1] > 120:
                sel = torch.zeros_like(valid)
                sel[:, ::STRIDE] = 1.0
                w = w * sel
            loss_nll = model.nll(out["q"], rul, w)
            loss_crps = model.crps(out["q"], rul, w)
            obs_nll = F.mse_loss(out["mu"] * mask, x * mask, reduction="none").mean()
            loss = loss_nll + 0.05 * loss_crps + 0.3 * obs_nll
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()
        met = evaluate(model, va, device)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), **met))
        flag = ""
        if met["rmse_last"] < best - 1e-4:
            best, bad = met["rmse_last"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            bad += 1
        if ep % 2 == 0 or ep == 1:
            print(f"ep{ep:03d} loss={tot/nb:.4f} val_rmse_all={met['rmse_all']:.2f} "
                  f"last={met['rmse_last']:.2f} cov90={met['cover90_all']:.2f} "
                  f"gate={met['mean_gate']:.3f}{flag}", flush=True)
        if bad >= a.patience:
            print(f"early stop at ep{ep}", flush=True)
            break

    out_dir = a.out or f"{ROOT}/experiments/cet/{a.subset}/f{a.fold}_s{a.seed}{a.tag}"
    os.makedirs(out_dir, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model": model.state_dict(),
                "config": dict(d=64, k_bins=K_BINS, alpha=0.05, ablation=a.ablation,
                               subset=a.subset, fold=a.fold, seed=a.seed)},
               f"{out_dir}/model.pt")
    final = evaluate(model, va, device)
    json.dump(dict(subset=a.subset, fold=a.fold, seed=a.seed, ablation=a.ablation,
                   num_params=n_par, epochs=len(hist), best_val_rmse_last=best,
                   final=final, history=hist, elapsed=time.time() - t0,
                   finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds")),
              open(f"{out_dir}/metrics.json", "w"), indent=2)
    print(f"saved {out_dir}  best_last={best:.3f}  final={json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
