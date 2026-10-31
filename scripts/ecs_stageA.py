"""ECS Stage A: train the age prior (closed form) and the bin likelihood (5 folds).

Usage: python ecs_stageA.py --subset FD002 --fold 2 --seed 2027
Outputs: experiments/ecs/{subset}/f{fold}_s{seed}/prior.json + like.pt + metrics.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

ROOT = "/home/yndx/zxk/bearing-rul-pi"
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/scripts")

import dsh_e21_test_all as e21
import run_candidate as rc
from ecs.prior import AgeConditionalPrior
from ecs.likelihood import BinLikelihood, rul_to_bin, log_prob_np, N_BINS

EPOCHS, PATIENCE, BATCH, LR = 80, 20, 1024, 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    raw, tr, va, _ = e21.split_frames(a.subset, a.fold)
    # ---- prior from training engines only ----
    lifetimes = tr.groupby("unit").size().to_numpy(float)
    prior = AgeConditionalPrior(lifetimes)

    # ---- training windows (frozen builder) ----
    scaler = e21.fit_scaler(a.subset, tr)
    tr_s = e21.scale_frame(a.subset, scaler, tr)
    va_s = e21.scale_frame(a.subset, scaler, va)
    xtr, ytr, etr = rc.build_windows(tr, tr_s, e21.WINDOW, 1, 125.0)
    xva, yva, eva, _ = rc.build_windows_matched(va, va_s, e21.WINDOW, 125.0)

    xt = torch.from_numpy(xtr); et = torch.from_numpy(etr)
    yt = torch.from_numpy(rul_to_bin(ytr)).long()
    xv = torch.from_numpy(xva).to(device); ev = torch.from_numpy(eva).to(device)
    yv_np = np.asarray(yva, dtype=float)

    model = BinLikelihood().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    n = len(xt); best, best_state, patience, hist = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            lg = model(xt[idx].to(device), extras=et[idx].to(device))
            loss = ce(lg, yt[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            lv = model(xv, extras=ev)
            vloss = float(ce(lv, torch.from_numpy(rul_to_bin(yv_np)).long().to(device)))
            # expected RUL = sum p_bin * centre  (for a dev-side sanity metric)
            pv = torch.softmax(lv, -1).cpu().numpy()
        exp_rul = (pv * np.linspace(125.0 / N_BINS / 2, 125.0 - 125.0 / N_BINS / 2, N_BINS)).sum(1)
        vrmse = float(np.sqrt(np.mean((exp_rul - yv_np) ** 2)))
        hist.append(dict(epoch=ep, train_ce=tot / n, val_ce=vloss, val_rmse=vrmse))
        if vloss < best - 1e-6:
            best, patience = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"early stop ep{ep}", flush=True); break
        if ep % 10 == 0:
            print(f"[{a.subset} f{a.fold} s{a.seed}] ep{ep:03d} train_ce={tot/n:.4f} "
                  f"val_rmse={vrmse:.2f}", flush=True)

    out = f"{ROOT}/experiments/ecs/{a.subset}/f{a.fold}_s{a.seed}"
    os.makedirs(out, exist_ok=True)
    model.load_state_dict(best_state); model.eval()
    torch.save({"model": best_state, "config": dict(n_bins=N_BINS, d_model=64,
               num_layers=1, dropout=0.15, subset=a.subset, fold=a.fold, seed=a.seed)},
               f"{out}/like.pt")
    json.dump(dict(subset=a.subset, fold=a.fold, seed=a.seed,
                   prior=dict(k=prior.k, lam=prior.lam, n_train_engines=int(len(lifetimes))),
                   best_val_ce=best,
                   best_val_rmse=min(h["val_rmse"] for h in hist),
                   epochs_ran=len(hist), elapsed=time.time() - t0,
                   num_params=int(sum(p.numel() for p in model.parameters())),
                   finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds")),
              open(f"{out}/metrics.json", "w"), indent=2)
    print(f"saved {out} best_val_rmse={min(h['val_rmse'] for h in hist):.2f}")


if __name__ == "__main__":
    main()
