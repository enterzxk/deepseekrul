"""IDA-PE Module B / C arms (frozen spec: gates.json -> modules_BC).

Arms (all build on M1p: zero-init context injection of phi(tau)++u):
  Module B: MB_tau / MB_abs / MB_shuf   (degradation-conditioned attention)
  Module C: MC_tau / MC_abs / MC_shuf / MC_static  (sensor mixer)

The module-level condition vector differs per arm while the context
injection stays the individualised M1p vector everywhere, so comparisons
isolate the causal content of the module input:
  tau   : phi(t/mu_T) ++ u            (individualised)
  abs   : phi_abs(t) ++ u             (absolute cycle, periods 25/50/125/250)
  shuf  : tau of a permuted engine    (within-fold unit permutation)
  static: constant vector             (capacity matched, condition-free)

Zero-init assertion: at initialisation every arm output equals the M1p
PriorWrapper exactly (tests/test_modulebc_equivalence.py T3).
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from dataset import compute_rmse  # noqa: E402
from candidates.model import CandidateModel  # noqa: E402
from candidates.deg_modules import DegCandidateModel  # noqa: E402
from lifetime_prior.nested_prior import StaircasePrior  # noqa: E402
from run_ida import (  # noqa: E402
    TAU_PERIODS, OUTER_SEED, PriorWrapper, outer_folds, prep_fold,
    prior_vectors, set_seed,
)

ARM_MODULES = {
    "MB_tau": ("B", "tau"), "MB_abs": ("B", "abs"), "MB_shuf": ("B", "shuf"),
    "MC_tau": ("C", "tau"), "MC_abs": ("C", "abs"), "MC_shuf": ("C", "shuf"),
    "MC_static": ("C", "static"),
}
ABS_PERIODS = (25.0, 50.0, 125.0, 250.0)


class DegWindowSet(Dataset):
    def __init__(self, x, y, cond, extras, prior, deg):
        self.x, self.y = torch.from_numpy(x), torch.from_numpy(y)
        self.cond = torch.from_numpy(cond) if cond is not None else None
        self.extras = torch.from_numpy(extras)
        self.prior = torch.from_numpy(prior)
        self.deg = torch.from_numpy(deg)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        cond = self.cond[i] if self.cond is not None else torch.zeros(50, 3)
        return (self.x[i], self.y[i], cond, self.extras[i],
                self.prior[i], self.deg[i])


def deg_vectors(meta, mu, sg, cond_mode, fold, sp=None, shuffle_seed=777):
    """Module condition vector (9-dim) per window, per arm.

    shuf semantics follow M1_shuf exactly: permute engines within the fold,
    then look up the permuted engine's prior at the same cutoff t (so the
    marginal mu distribution is preserved while alignment to the true
    engine's degradation stage is destroyed).
    """
    t = meta[:, 1].astype(np.float64)
    u = sg / np.maximum(mu, 1.0)
    if cond_mode == "static":
        return np.ones((len(meta), 9), dtype=np.float32)
    if cond_mode == "abs":
        ang = 2.0 * np.pi * t[:, None] / np.asarray(ABS_PERIODS)[None, :]
        phi = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
        return np.concatenate([phi, u[:, None]], axis=1).astype(np.float32)
    # tau / shuf
    mu_q = mu
    if cond_mode == "shuf":
        rng = np.random.default_rng(shuffle_seed + fold)
        units = np.unique(meta[:, 0])
        perm = {int(a): int(b) for a, b in zip(units, rng.permutation(units))}
        mu_q = np.array([float(sp.lookup(fold, perm[int(u_)], int(t_))[0])
                         for u_, t_ in zip(meta[:, 0], t)])
    tau = t / np.maximum(mu_q, 1.0)
    ang = 2.0 * np.pi * tau[:, None] / np.asarray(TAU_PERIODS)[None, :]
    phi = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
    return np.concatenate([phi, u[:, None]], axis=1).astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y, cond, extras, prior, deg in loader:
        out = model(x.to(device), cond.to(device), extras.to(device),
                    deg.to(device), prior.to(device))
        preds.append(out["prediction"].reshape(-1).cpu().numpy())
        targets.append(y.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARM_MODULES))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--prior-dir", default="experiments/ida/prior")
    ap.add_argument("--out", default="experiments/ida/moduleBC")
    args = ap.parse_args()
    module, cond_mode = ARM_MODULES[args.arm]

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, df = outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]

    prior_dir = ROOT / args.prior_dir
    table = pd.read_csv(prior_dir / f"{args.subset}_prior_cutoffs.csv")
    meta_p = json.loads((prior_dir / f"{args.subset}_prior_meta.json").read_text())
    median_T = float(meta_p["folds"][str(args.fold)]["median_T_train"])
    sp = StaircasePrior(
        table, {int(k): v for k, v in
                ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())},
        args.subset)

    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = prep_fold(
        df, args.subset, args.fold, tr_u, va_u, window=args.window)
    ptr, (mu_tr, sg_tr) = prior_vectors(mtr, sp, args.fold, "M1p", median_T)
    pva, (mu_va, sg_va) = prior_vectors(mva, sp, args.fold, "M1p", median_T)
    dtr = deg_vectors(mtr, mu_tr, sg_tr, cond_mode, args.fold, sp)
    dva = deg_vectors(mva, mu_va, sg_va, cond_mode, args.fold, sp)
    train_set = DegWindowSet(xtr, ytr, ctr, etr, ptr, dtr)
    val_set = DegWindowSet(xva, yva, cva, eva, pva, dva)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2)

    body_kw = dict(num_sensors=14, window_size=args.window, d_model=64,
                   num_layers=1, dropout=0.15, use_baseline=True,
                   use_ema_features=True, use_cycle=True, use_pe=True,
                   use_graph=False, use_dual_head=False)
    model = DegCandidateModel(module_b=(module == "B"), module_c=(module == "C"),
                              cond_dim=9, **body_kw).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # zero-init equivalence vs M1p wrapper (identical weights, zero modules)
    ref_body = CandidateModel(**body_kw)
    missing, unexpected = model.load_state_dict(ref_body.state_dict(), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    extra_keys = ("deg_bias", "mixer", "prior_proj")
    assert not [k for k in missing if not any(e in k for e in extra_keys)], \
        f"body keys missing: {[k for k in missing if not any(e in k for e in extra_keys)][:5]}"
    nn.init.zeros_(model.prior_proj.weight)   # re-zero (weights were
    nn.init.zeros_(model.prior_proj.bias)     # overwritten by load)
    if model.module_b:
        nn.init.zeros_(model.deg_bias.alpha.weight)
        nn.init.zeros_(model.deg_bias.alpha.bias)
    if model.module_c:
        nn.init.zeros_(model.mixer.gate.weight)
        nn.init.zeros_(model.mixer.gate.bias)
    wrapper = PriorWrapper(ref_body, "M1p").to(device).eval()
    model.eval()
    with torch.no_grad():
        xb, yb, cb, eb, pb, db = next(iter(val_loader))
        o1 = model(xb[:8].to(device), cb[:8].to(device), eb[:8].to(device),
                   db[:8].to(device), pb[:8].to(device))["prediction"]
        o2 = wrapper(xb[:8].to(device), cb[:8].to(device), eb[:8].to(device),
                     pb[:8].to(device))["prediction"]
        err = float((o1 - o2).abs().max())
    assert err < 1e-4, f"zero-init broken vs M1p: {err}"

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs,
                                                           eta_min=1e-5)
    LATE = 1.8
    best, best_epoch, best_state = float("inf"), -1, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for x, y, cond, extras, prior, deg in train_loader:
            x, y = x.to(device), y.to(device)
            cond, extras = cond.to(device), extras.to(device)
            prior, deg = prior.to(device), deg.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(x, cond, extras, deg, prior)["prediction"]
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
        print(f"[{args.subset} {args.arm} f{args.fold} s{args.seed}] ep{epoch:03d} "
              f"loss={tot/max(seen,1):.3f} val={val:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    p, t = evaluate(model, val_loader, device)

    out_dir = ROOT / args.out / f"{args.subset}" / f"{args.arm}_f{args.fold}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "unit": mva[:, 0], "end_cycle": mva[:, 1], "y_true": t, "y_pred": p,
        "mu_T": mu_va, "sigma_T": sg_va,
    }).to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "subset": args.subset, "arm": args.arm, "fold": args.fold, "seed": args.seed,
        "best_val_rmse": best, "best_epoch": best_epoch, "params": n_params,
        "n_train_windows": len(xtr), "n_val_windows": len(xva),
        "n_val_units": len(set(mva[:, 0].tolist())),
        "elapsed_seconds": time.time() - t0,
        "epochs_ran": min(args.epochs, best_epoch + args.patience),
        "outer_seed": OUTER_SEED,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }, indent=2))
    print(f"BEST {args.subset} {args.arm} f{args.fold} s{args.seed} "
          f"val={best:.4f} params={n_params}", flush=True)


if __name__ == "__main__":
    main()
