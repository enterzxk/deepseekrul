"""End-to-end CQR on the IDA-PE backbone.

Round-2 experiment: the round-1 post-hoc quantile heads failed because the
frozen representation carries no width information. Here the backbone is
trained jointly with the quantile loss, so conditional width can actually be
learned. Calibration is still split conformal (CQR score) on held-out engines,
so the coverage statement remains distribution-free.

Zero intrusion: run_ida.py is imported, never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ida as R  # noqa: E402
from src.punet.conformal import engine_calib_split  # noqa: E402
from src.punet.evaluate import evaluate_cqr_intervals  # noqa: E402


class E2ECQR(nn.Module):
    """Backbone + zero-init prior projection + centre/half-width heads."""

    def __init__(self, body: nn.Module, d_model: int = 64, prior_dim: int = 2) -> None:
        super().__init__()
        self.body = body
        self.proj = nn.Linear(prior_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.centre = nn.Linear(d_model, 1)
        self.half_head = nn.Linear(d_model, 1)

    def forward(self, x, cond, extras, prior):
        pooled = self.body.encode(x, cond=cond, extras=extras)
        ctx = pooled.mean(dim=1) + self.proj(prior)
        centre = self.centre(ctx).squeeze(-1)
        half = F.softplus(self.half_head(ctx).squeeze(-1))
        return centre - half, centre + half, centre


def pinball(pred, target, q):
    d = target - pred
    return torch.mean(torch.maximum(q * d, (q - 1.0) * d))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--use-ema", action="store_true",
                    help="evaluate with EMA weights (default: best validation state)")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--prior-dir", default="experiments/ida/prior_boot")
    ap.add_argument("--out", default="experiments/punet_e2e")
    ap.add_argument("--ablate", default="none", choices=["none", "A4"])
    ap.add_argument("--heads", default="cqr", choices=["cqr", "point"])
    args = ap.parse_args(argv)

    R.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, df = R.outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]

    prior_dir = ROOT / args.prior_dir
    table = pd.read_csv(prior_dir / f"{args.subset}_prior_cutoffs.csv")
    meta_p = json.loads((prior_dir / f"{args.subset}_prior_meta.json").read_text())
    median_T = float(meta_p["folds"][str(args.fold)]["median_T_train"])
    sp = R.StaircasePrior(
        table,
        {int(k): v for k, v in ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())},
        args.subset,
    )

    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(
        df, args.subset, args.fold, tr_u, va_u, window=args.window
    )
    ptr, (mu_tr, sg_tr) = R.prior_vectors(mtr, sp, args.fold, "M1", median_T)
    pva, (mu_va, sg_va) = R.prior_vectors(mva, sp, args.fold, "M1", median_T)
    if args.ablate == "A4":
        _g = np.random.default_rng(args.seed)
        ptr = ptr.copy(); pva = pva.copy()
        ptr[:, 1] = ptr[_g.permutation(len(ptr)), 1]
        _g2 = np.random.default_rng(args.seed + 1)
        pva[:, 1] = pva[_g2.permutation(len(pva)), 1]

    train_loader = DataLoader(R.WindowSet(xtr, ytr, ctr, etr, ptr), batch_size=256, shuffle=True)
    val_loader = DataLoader(R.WindowSet(xva, yva, cva, eva, pva), batch_size=512, shuffle=False)

    body = R.CandidateModel(
        num_sensors=14, window_size=args.window, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
        use_graph=False, use_dual_head=False,
    )
    model = E2ECQR(body).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    ema_decay = float(args.ema_decay)

    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, cond, extras, prior in train_loader:
            x, y = x.to(device), y.to(device)
            cond, extras, prior = cond.to(device), extras.to(device), prior.to(device)
            lo, hi, centre = model(x, cond, extras, prior)
            y2 = y.reshape(-1)
            if args.heads == "point":
                loss = F.mse_loss(centre, y2)
            else:
                loss = pinball(lo, y2, 0.05) + pinball(hi, y2, 0.95) + 0.1 * F.mse_loss(centre, y2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
        sched.step()
        model.eval()
        errs = []
        with torch.no_grad():
            for x, y, cond, extras, prior in val_loader:
                _, _, centre = model(x.to(device), cond.to(device), extras.to(device), prior.to(device))
                errs.append(((centre.cpu() - y.reshape(-1)) ** 2).numpy())
        val = float(np.sqrt(np.concatenate(errs).mean()))
        if val < best - 1e-4:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
        if epoch % 10 == 0:
            print(f"[{args.subset} e2e] ep{epoch:03d} val_centre_rmse={val:.4f}", flush=True)
    if args.use_ema:
        model.load_state_dict({k: v.to(dtype=model.state_dict()[k].dtype) for k, v in ema.items()})
    elif best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    los, his, ys, units = [], [], [], []
    with torch.no_grad():
        for x, y, cond, extras, prior in val_loader:
            lo, hi, _ = model(x.to(device), cond.to(device), extras.to(device), prior.to(device))
            los.append(lo.cpu().numpy()); his.append(hi.cpu().numpy()); ys.append(y.numpy())
    lo_all = np.concatenate(los); hi_all = np.concatenate(his); y_all = np.concatenate(ys)
    if args.heads == "point":
        # constant-width baseline: collapse the quantile heads onto the centre,
        # so CQR calibration reduces to absolute-residual split conformal
        lo_all = hi_all.copy() if False else (lo_all + hi_all) / 2.0
        hi_all = lo_all.copy()
    units = mva[:, 0]

    calib_idx, test_idx = engine_calib_split(units, calib_frac=0.3, seed=args.seed)
    calib_mask = np.zeros(len(units), dtype=bool); calib_mask[calib_idx] = True
    test_mask = np.zeros(len(units), dtype=bool); test_mask[test_idx] = True

    result = evaluate_cqr_intervals(
        units, mva[:, 1].astype(float), np.asarray(mu_va), y_all, lo_all, hi_all,
        calib_mask, test_mask, delta=args.delta,
    )
    result.update(method=f"e2e_{args.heads}", ablate=args.ablate, subset=args.subset, fold=args.fold,
                  best_centre_rmse=best, best_epoch=best_epoch)
    out_dir = ROOT / args.out / args.subset / f"f{args.fold}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"e2e_{args.heads}_{args.ablate}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
