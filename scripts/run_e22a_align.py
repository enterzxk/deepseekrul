"""E22a-scale alignment arm: big backbone (d_model=88, 2 layers) + prior injection.

Purpose: the historical best PE (E22a, official-test RMSE FD002 11.51) used a
larger config than the frozen IDA small config (d_model=64, 1 layer). This
script runs the SAME IDA protocol (outer folds, prior tables, PriorWrapper,
late-weight loss, cosine schedule) with ONLY the backbone capacity and the
E22a schedule changed, so we can compare:
  - M0_big vs M1p_big  (does the prior still help at E22a scale?)
  - M1p_big vs historical E22a PE  (zero-init guarantee: M1p_big >= PE at
    equal capacity, so the final test number cannot be worse).

Arms: M0 / M1 / M1p (same semantics as run_ida.py).
Output: experiments/ida/e22a_align/{subset}/{arm}_f{fold}_s{seed}/
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import torch
import torch.nn as nn

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import compute_rmse  # noqa: E402
from candidates.model import CandidateModel  # noqa: E402
from lifetime_prior.nested_prior import StaircasePrior  # noqa: E402
from run_ida import (  # noqa: E402
    OUTER_SEED, PriorWrapper, WindowSet, evaluate, outer_folds,
    prep_fold, prior_vectors, set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--arm", required=True, choices=("M0", "M1", "M1p"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=40)      # E22a schedule
    ap.add_argument("--patience", type=int, default=15)    # E22a schedule
    ap.add_argument("--d-model", type=int, default=88)     # E22a backbone
    ap.add_argument("--num-layers", type=int, default=2)   # E22a backbone
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--prior-dir", default="experiments/ida/prior")
    ap.add_argument("--out", default="experiments/ida/e22a_align")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, df = outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]

    prior_dir = ROOT / args.prior_dir
    table = pd.read_csv(prior_dir / f"{args.subset}_prior_cutoffs.csv")
    meta_p = json.loads((prior_dir / f"{args.subset}_prior_meta.json").read_text())
    median_T = float(meta_p["folds"][str(args.fold)]["median_T_train"])
    sp = StaircasePrior(table, {int(k): v for k, v in
                                ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())},
                        args.subset)

    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = prep_fold(
        df, args.subset, args.fold, tr_u, va_u, window=args.window)
    ptr, (mu_tr, sg_tr) = prior_vectors(mtr, sp, args.fold, args.arm, median_T)
    pva, (mu_va, sg_va) = prior_vectors(mva, sp, args.fold, args.arm, median_T)
    train_set = WindowSet(xtr, ytr, ctr, etr, ptr)
    val_set = WindowSet(xva, yva, cva, eva, pva)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2)

    body = CandidateModel(
        num_sensors=14, window_size=args.window, d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=0.15, use_baseline=True, use_ema_features=True, use_cycle=True,
        use_pe=True, use_graph=False, use_dual_head=False)
    model = PriorWrapper(body, args.arm, d_model=args.d_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # zero-init equivalence assertion (spec: init exactly equals PE)
    if args.arm != "M0":
        model.eval()
        with torch.no_grad():
            xb, yb, cb, eb, pb = next(iter(val_loader))
            o1 = model(xb[:8].to(device), cb[:8].to(device), eb[:8].to(device), pb[:8].to(device))
            o2 = model.body.reg_head(model.body.encode(
                xb[:8].to(device), cond=cb[:8].to(device), extras=eb[:8].to(device)).mean(dim=1))
            err = float((o1["prediction"] - o2).abs().max())
        assert err < 1e-4, f"zero-init broken: max delta {err}"

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs,
                                                           eta_min=1e-5)
    LATE = 1.8
    best, best_epoch, best_state = float("inf"), -1, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for x, y, cond, extras, prior in train_loader:
            x, y = x.to(device), y.to(device)
            cond, extras, prior = cond.to(device), extras.to(device), prior.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(x, cond, extras, prior)["prediction"]
            y = y.reshape(-1, 1)
            delta = pred - y
            loss = nn.functional.mse_loss(pred, y)
            late = torch.where(delta > 0, LATE * delta.square(), delta.square())
            loss = loss + 0.5 * late.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tot += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()
        p, t = evaluate(model, val_loader, device)
        val = float(compute_rmse(t, p))
        if val < best - 1e-4:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
        print(f"[{args.subset} {args.arm}big f{args.fold} s{args.seed}] ep{epoch:03d} "
              f"loss={tot/max(seen,1):.3f} val={val:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    p, t = evaluate(model, val_loader, device)

    out_dir = ROOT / args.out / f"{args.subset}" / f"{args.arm}big_f{args.fold}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "unit": mva[:, 0], "end_cycle": mva[:, 1], "y_true": t, "y_pred": p,
        "mu_T": mu_va, "sigma_T": sg_va,
    }).to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "subset": args.subset, "arm": f"{args.arm}big", "fold": args.fold,
        "seed": args.seed,
        "best_val_rmse": best, "best_epoch": best_epoch, "params": n_params,
        "d_model": args.d_model, "num_layers": args.num_layers,
        "epochs": args.epochs, "patience": args.patience,
        "n_train_windows": len(xtr), "n_val_windows": len(xva),
        "n_val_units": len(set(mva[:, 0].tolist())),
        "elapsed_seconds": time.time() - t0,
        "epochs_ran": min(args.epochs, best_epoch + args.patience),
        "outer_seed": OUTER_SEED,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }, indent=2))
    print(f"BEST {args.subset} {args.arm}big f{args.fold} s{args.seed} "
          f"val={best:.4f} params={n_params}", flush=True)


if __name__ == "__main__":
    main()
