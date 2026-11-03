"""H3 arm: PE + H1 (HGB) RUL prediction as an input feature.

Feature construction mirrors the lifetime prior exactly:
- train windows: OOF predictions -- outer-train engines are split into the
  SAME inner 4 folds as the prior (seed 20261104+100+fold); each engine's
  H1 feature comes from HGBs trained on the other 3 parts only (no in-bag
  optimism anywhere);
- outer-val windows: 5-seed HGB ensemble trained on the full outer-train
  (the run_h1 output, aligned by (unit, end_cycle)).

Injection: zero-init linear map of [h1_pred / 125] into the pooled context
(same site as M1), so H3 starts exactly at M0 and is parameter-matched with
M1 (+192). Any difference M1 vs H3 is purely "what information" (lifetime
scale vs direct RUL guess), under the same nested causal protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
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
GATE1_DIR = ROOT / "experiments" / "ida" / "gate1"


def hgb_ensemble_fit(Xtr, Ytr, weights):
    preds = np.zeros(len(Xtr))
    models = []
    for s in HGB_SEEDS:
        m = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=200, learning_rate=0.05,
            max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=10.0,
            early_stopping=False, random_state=s)
        m.fit(Xtr, Ytr, sample_weight=weights)
        models.append(m)
    return models


def oof_h1_features(df, subset, fold, tr_u):
    """OOF H1 RUL predictions for every outer-train window end."""
    inner = KFold(n_splits=4, shuffle=True,
                  random_state=OUTER_SEED + 100 + fold)
    units_sorted = np.array(sorted(tr_u), dtype=np.int64)
    parts = []
    for k, (ii, ik) in enumerate(inner.split(units_sorted)):
        fit_u = set(int(u) for u in units_sorted[ii])
        oof_u = set(int(u) for u in units_sorted[ik])
        tr_df, oof_df, tr_s, oof_s = fit_scalers(
            df[df.unit.isin(fit_u)], df[df.unit.isin(oof_u)], subset, fold)
        Xtr, Ytr, Mtr = window_rows(tr_df, tr_s)
        Xo, _, Mo = window_rows(oof_df, oof_s)
        count = {}
        for u in Mtr[:, 0]:
            count[int(u)] = count.get(int(u), 0) + 1
        w = np.array([1.0 / count[int(u)] for u in Mtr[:, 0]])
        preds = np.zeros(len(Xo))
        for m in hgb_ensemble_fit(Xtr, Ytr, w):
            preds += m.predict(Xo)
        parts.append(pd.DataFrame({
            "unit": Mo[:, 0], "end_cycle": Mo[:, 1],
            "h1_pred": np.maximum(preds / len(HGB_SEEDS), 0.0)}))
        print(f"[h3-oof {subset} f{fold} k{k}] {len(parts[-1])} windows",
              flush=True)
    return pd.concat(parts, ignore_index=True)


class H3Wrapper(nn.Module):
    """M0 body + zero-init projection of the H1 feature into the context."""

    def __init__(self):
        super().__init__()
        from candidates.model import CandidateModel
        self.body = CandidateModel(
            num_sensors=14, window_size=50, d_model=64, num_layers=1,
            dropout=0.15, use_baseline=True, use_ema_features=True,
            use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
        self.proj = nn.Linear(1, 64)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond, extras, prior):
        pooled = self.body.encode(x, cond=cond, extras=extras)
        return {"prediction": self.body.reg_head(
            pooled.mean(dim=1) + self.proj(prior))}


def lookup(meta, table):
    d = {(int(u), int(t)): float(p) for u, t, p in
         table[["unit", "end_cycle", "h1_pred"]].itertuples(index=False)}
    out = np.empty(len(meta), np.float32)
    for i, (u, t) in enumerate(meta):
        out[i] = d[(int(u), int(t))]          # KeyError = coverage bug
    return (out / 125.0).reshape(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", default="experiments/ida/gate1")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    folds, df = outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]

    oof_path = GATE1_DIR / "h3_oof" / f"{args.subset}_f{args.fold}_oof.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
    else:
        oof = oof_h1_features(df, args.subset, args.fold, tr_u)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof.to_csv(oof_path, index=False)

    h1_val = pd.read_csv(
        ROOT / "experiments/ida/gate0" / args.subset /
        f"H1_f{args.fold}_s0" / "val_predictions.csv")[
            ["unit", "end_cycle", "y_pred"]].rename(columns={"y_pred": "h1_pred"})

    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = prep_fold(
        df, args.subset, args.fold, tr_u, va_u, window=50)
    ptr = lookup(mtr, oof)
    pva = lookup(mva, h1_val)

    train_set = WindowSet(xtr, ytr, ctr, etr, ptr)
    val_set = WindowSet(xva, yva, cva, eva, pva)
    loader = DataLoader(train_set, batch_size=256, shuffle=True,
                        num_workers=2, pin_memory=True)
    vloader = DataLoader(val_set, batch_size=512, num_workers=2)

    model = H3Wrapper().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # zero-init equivalence assertion
    model.eval()
    with torch.no_grad():
        xb, yb, cb, eb, pb = next(iter(vloader))
        o1 = model(xb[:8].to(device), cb[:8].to(device), eb[:8].to(device),
                   pb[:8].to(device))
        b = model.body
        o2 = b.reg_head(b.encode(xb[:8].to(device), cond=cb[:8].to(device),
                                 extras=eb[:8].to(device)).mean(dim=1))
    assert float((o1["prediction"] - o2).abs().max()) < 1e-4, "zero-init broken"

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)
    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, cond, extras, prior in loader:
            x, y = x.to(device), y.to(device)
            cond, extras, prior = (cond.to(device), extras.to(device),
                                   prior.to(device))
            opt.zero_grad(set_to_none=True)
            pred = model(x, cond, extras, prior)["prediction"]
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
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
        print(f"[H3 {args.subset} f{args.fold} s{args.seed}] ep{epoch:03d} "
              f"val={val:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    p, t = evaluate(model, vloader, device)

    out_dir = GATE1_DIR / args.subset / f"H3_f{args.fold}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "unit": mva[:, 0], "end_cycle": mva[:, 1], "y_true": t, "y_pred": p,
        "h1_pred": pva.reshape(-1) * 125.0,
    }).to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "subset": args.subset, "arm": "H3", "fold": args.fold,
        "seed": args.seed, "best_val_rmse": best, "best_epoch": best_epoch,
        "params": n_params, "n_train_windows": len(xtr),
        "n_val_windows": len(xva), "outer_seed": OUTER_SEED,
        "elapsed_seconds": time.time() - t0,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }, indent=2))
    print(f"BEST {args.subset} H3 f{args.fold} s{args.seed} "
          f"val={best:.4f} params={n_params}", flush=True)


if __name__ == "__main__":
    main()
