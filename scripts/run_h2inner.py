"""H2 weight selection data: inner-CV predictions of PE and HGB.

For each (subset, outer fold): split outer-train engines with the SAME inner
4-fold convention as the lifetime prior (KFold(4, shuffle,
random_state=20261104+100+fold) over sorted units). For each inner split,
train PE (M0 config, identical budget) on inner-train and predict inner-val;
train the HGB RUL ensemble (run_h1 config) on inner-train windows and predict
inner-val. Saves both prediction sets so the mixing weight can be chosen
without ever touching outer-val.

GPU usage: 4 inner PE trainings per outer fold. Run after the Stage B queue
releases the GPUs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from run_ida import (  # noqa: E402
    WindowSet, evaluate, outer_folds, prep_fold, set_seed,
)
from dataset import compute_rmse  # noqa: E402
from run_h1 import window_rows, fit_scalers  # noqa: E402

OUTER_SEED = 20261104
HGB_SEEDS = (11, 22, 33, 44, 55)


def train_pe(train_set, val_set, device, seed, epochs=80, patience=20):
    set_seed(seed)
    model = PriorWrapperStub().to(device)
    loader = DataLoader(train_set, batch_size=256, shuffle=True,
                        num_workers=2, pin_memory=True)
    vloader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, cond, extras, prior in loader:
            x, y = x.to(device), y.to(device)
            cond, extras = cond.to(device), extras.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x, cond, extras, prior)["prediction"].reshape(-1, 1)
            y = y.reshape(-1, 1)
            delta = pred - y
            loss = nn.functional.mse_loss(pred, y)
            late = torch.where(delta > 0, 1.8 * delta.square(), delta.square())
            (loss + 0.5 * late.mean()).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        p, t = evaluate(model, vloader, device)
        val = float(compute_rmse(t, p))
        if val < best - 1e-4:
            best, best_state = val, {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
        elif epoch - best >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


class PriorWrapperStub(nn.Module):
    """M0 = body encode -> mean pool -> reg head (identical to run_ida's
    PriorWrapper with arm='M0', kept local to avoid touching run_ida)."""

    def __init__(self):
        super().__init__()
        from candidates.model import CandidateModel
        self.body = CandidateModel(
            num_sensors=14, window_size=50, d_model=64, num_layers=1,
            dropout=0.15, use_baseline=True, use_ema_features=True,
            use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)

    def forward(self, x, cond, extras, prior):
        pooled = self.body.encode(x, cond=cond, extras=extras)
        return {"prediction": self.body.reg_head(pooled.mean(dim=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--out", default="experiments/ida/gate1/h2inner")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    folds, df = outer_folds(args.subset)
    tr_u, _ = folds[args.fold]

    inner = KFold(n_splits=4, shuffle=True,
                  random_state=OUTER_SEED + 100 + args.fold)
    units_sorted = np.array(sorted(tr_u), dtype=np.int64)
    summary = []
    for k, (ii, ik) in enumerate(inner.split(units_sorted)):
        in_tr = set(int(u) for u in units_sorted[ii])
        in_va = set(int(u) for u in units_sorted[ik])
        # --- windows + scalers (fit on inner-train only) ---
        tr_df_all = df[df.unit.isin(in_tr)]
        va_df_all = df[df.unit.isin(in_va)]
        tr_df, va_df, tr_s, va_s = fit_scalers(tr_df_all, va_df_all,
                                               args.subset, args.fold)
        (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = prep_fold(
            df, args.subset, args.fold, in_tr, in_va, window=50)
        # PE predictions on inner-val
        train_set = WindowSet(xtr, ytr, ctr, etr,
                              np.zeros((len(xtr), 1), np.float32))
        val_set = WindowSet(xva, yva, cva, eva,
                            np.zeros((len(xva), 1), np.float32))
        model, best = train_pe(train_set, val_set, device, seed=101)
        p_pe, t_pe = evaluate(model, DataLoader(val_set, batch_size=512,
                                                num_workers=2), device)
        # HGB (H1 config) predictions on inner-val
        Xtr, Ytr, Mtr = window_rows(tr_df, tr_s)
        Xva, _, Mva_chk = window_rows(va_df, va_s)
        assert np.array_equal(Mva_chk, mva), "window meta mismatch"
        count = {}
        for u in Mtr[:, 0]:
            count[int(u)] = count.get(int(u), 0) + 1
        w = np.array([1.0 / count[int(u)] for u in Mtr[:, 0]])
        preds = np.zeros(len(Xva))
        for s in HGB_SEEDS:
            m = HistGradientBoostingRegressor(
                loss="squared_error", max_iter=200, learning_rate=0.05,
                max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=10.0,
                early_stopping=False, random_state=s)
            m.fit(Xtr, Ytr, sample_weight=w)
            preds += m.predict(Xva)
        preds /= len(HGB_SEEDS)
        preds = np.maximum(preds, 0.0)

        out = ROOT / args.out / args.subset
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "unit": mva[:, 0], "end_cycle": mva[:, 1], "y_true": t_pe,
            "pe_pred": p_pe, "hgb_pred": preds,
        }).to_csv(out / f"f{args.fold}_k{k}.csv", index=False)
        rm_hgb = float(compute_rmse(t_pe, preds))
        summary.append(dict(k=k, pe_rmse=best, hgb_rmse=rm_hgb))
        print(f"[h2inner {args.subset} f{args.fold} k{k}] "
              f"PE {best:.3f} HGB {rm_hgb:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
    (ROOT / args.out / args.subset / f"f{args.fold}_summary.json").write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
