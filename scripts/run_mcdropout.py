"""MC-Dropout baseline: model-based uncertainty as the conformal scale.

Trains the same point-prediction backbone, then at inference keeps dropout
active for T stochastic passes. The predictive standard deviation becomes the
per-sample scale, and split conformal calibrates the quantile of
|y - mu| / sd on held-out engines.

This isolates "uncertainty learned by the quantile head" from "uncertainty
sampled from dropout".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ida as R  # noqa: E402
from run_e2e_cqr import E2ECQR  # noqa: E402
from src.punet.conformal import engine_calib_split  # noqa: E402
from src.punet.evaluate import evaluate_intervals  # noqa: E402


def enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, loader, device, passes: int):
    mus, sds, ys = [], [], []
    for x, y, cond, extras, prior in loader:
        x, cond, extras, prior = x.to(device), cond.to(device), extras.to(device), prior.to(device)
        preds = []
        for _ in range(passes):
            _, _, c = model(x, cond, extras, prior)
            preds.append(c.cpu().numpy())
        stack = np.stack(preds, axis=0)
        mus.append(stack.mean(axis=0))
        sds.append(stack.std(axis=0) + 1e-6)
        ys.append(y.numpy())
    return np.concatenate(mus), np.concatenate(sds), np.concatenate(ys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--prior-dir", default="experiments/ida/prior_boot")
    ap.add_argument("--root", default="experiments/punet_e2e")
    args = ap.parse_args(argv)

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
    ptr, _ = R.prior_vectors(mtr, sp, args.fold, "M1", median_T)
    pva, (mu_va, sg_va) = R.prior_vectors(mva, sp, args.fold, "M1", median_T)
    val_loader = DataLoader(R.WindowSet(xva, yva, cva, eva, pva), batch_size=512, shuffle=False)

    body = R.CandidateModel(
        num_sensors=14, window_size=args.window, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
        use_graph=False, use_dual_head=False,
    )
    model = E2ECQR(body).to(device)
    ckpt = ROOT / args.root / args.subset / f"f{args.fold}_s{args.seed}" / "model.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt} (train with --heads point --save-ckpt first)")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()
    enable_dropout(model)

    mu, sd, y = mc_predict(model, val_loader, device, args.passes)
    units = mva[:, 0]
    calib_idx, test_idx = engine_calib_split(units, calib_frac=0.3, seed=args.seed)
    calib_mask = np.zeros(len(units), dtype=bool); calib_mask[calib_idx] = True
    test_mask = np.zeros(len(units), dtype=bool); test_mask[test_idx] = True

    result = evaluate_intervals(
        units, mva[:, 1].astype(float), np.asarray(mu_va), y, mu, sd,
        calib_mask, test_mask, delta=args.delta,
    )
    result.update(method="mcdropout", subset=args.subset, fold=args.fold, passes=args.passes)
    out = ROOT / args.root / args.subset / f"f{args.fold}_s{args.seed}"
    pd.DataFrame({"unit": units, "y_true": y, "mu": mu, "sd": sd}).to_csv(
        out / "preds_mcd.csv", index=False)
    (out / "mcdropout.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
