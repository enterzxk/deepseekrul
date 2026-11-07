"""PU-Net stage runner: train the M1 backbone, freeze it, then fit the heads.

Reuses the exact training recipe of scripts/run_ida.py by importing its helpers
(outer_folds / prep_fold / prior_vectors / WindowSet / PriorWrapper / evaluate),
so run_ida.py itself is never modified.

Stages:
    backbone  train M1 and save the checkpoint
    heads     load the checkpoint, extract h, fit the two heads, calibrate, evaluate
    all       both (default)

Usage:
    python scripts/run_punet_stage.py --subset FD001 --fold 0 --seed 42 --stage all
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

from src.punet.conformal import block_split, engine_calib_split  # noqa: E402
from src.punet.evaluate import evaluate_intervals  # noqa: E402
from src.punet.train import fit_heads  # noqa: E402


def build(subset, fold, seed, window, prior_dir):
    folds, df = R.outer_folds(subset)
    tr_u, va_u = folds[fold]

    prior_dir = ROOT / prior_dir
    table = pd.read_csv(prior_dir / f"{subset}_prior_cutoffs.csv")
    meta_p = json.loads((prior_dir / f"{subset}_prior_meta.json").read_text())
    median_T = float(meta_p["folds"][str(fold)]["median_T_train"])
    sp = R.StaircasePrior(
        table,
        {int(k): v for k, v in ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())},
        subset,
    )

    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(
        df, subset, fold, tr_u, va_u, window=window
    )
    ptr, (mu_tr, sg_tr) = R.prior_vectors(mtr, sp, fold, "M1", median_T)
    pva, (mu_va, sg_va) = R.prior_vectors(mva, sp, fold, "M1", median_T)

    train_set = R.WindowSet(xtr, ytr, ctr, etr, ptr)
    val_set = R.WindowSet(xva, yva, cva, eva, pva)
    regime_va = None
    if subset in ("FD002", "FD004"):
        from sklearn.cluster import KMeans

        op = ["op_1", "op_2", "op_3"]
        tr_df = df[df.unit.isin(tr_u)]
        va_df = df[df.unit.isin(va_u)].sort_values(["unit", "cycle"])
        km = KMeans(n_clusters=6, random_state=7, n_init=10).fit(tr_df[op].to_numpy(np.float64))
        lab = km.predict(va_df[op].to_numpy(np.float64))
        key = {(int(u), int(c)): int(l) for u, c, l in zip(va_df["unit"], va_df["cycle"], lab)}
        regime_va = np.array([key[(int(u), int(c))] for u, c in mva])

    return {
        "regime_va": regime_va,
        "train_loader": DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0),
        "val_loader": DataLoader(val_set, batch_size=512, shuffle=False, num_workers=0),
        "meta_tr": mtr,
        "meta_va": mva,
        "mu_tr": mu_tr,
        "sg_tr": sg_tr,
        "mu_va": mu_va,
        "sg_va": sg_va,
        "ytr": ytr,
        "yva": yva,
    }


def make_model(device):
    body = R.CandidateModel(
        num_sensors=14, window_size=50, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
        use_graph=False, use_dual_head=False,
    )
    return R.PriorWrapper(body, "M1").to(device)


def train_backbone(model, data, args, device):
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-5)
    late_w = 1.8
    best, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, cond, extras, prior in data["train_loader"]:
            x, y = x.to(device), y.to(device)
            cond, extras, prior = cond.to(device), extras.to(device), prior.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(x, cond, extras, prior)["prediction"]
            y2 = y.reshape(-1, 1)
            delta = pred - y2
            loss = nn.functional.mse_loss(pred, y2)
            late = torch.where(delta > 0, late_w * delta.square(), delta.square())
            loss = loss + 0.5 * late.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()
        p, t = R.evaluate(model, data["val_loader"], device)
        val = float(R.compute_rmse(t, p))
        if val < best - 1e-4:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
        if epoch % 10 == 0:
            print(f"[{args.subset} f{args.fold}] ep{epoch:03d} val={val:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_rmse": best, "best_epoch": best_epoch}


@torch.no_grad()
def extract(model, loader, device):
    """Return (h, y_base, y_true) for a loader; h is the regression-head input."""
    model.eval()
    hs, ps, ts = [], [], []
    for x, y, cond, extras, prior in loader:
        x, cond, extras, prior = x.to(device), cond.to(device), extras.to(device), prior.to(device)
        pooled = model.body.encode(x, cond=cond, extras=extras)
        ctx = pooled.mean(dim=1)
        if model.proj is not None:
            ctx = ctx + model.proj(prior)
        hs.append(ctx.cpu())
        ps.append(model.body.reg_head(ctx).reshape(-1).cpu())
        ts.append(y.reshape(-1).cpu())
    return torch.cat(hs), torch.cat(ps), torch.cat(ts)


def run_heads(model, data, args, device):
    h_tr, y_base_tr, y_tr = extract(model, data["train_loader"], device)
    h_va, y_base_va, y_va = extract(model, data["val_loader"], device)

    mu_tr = torch.tensor(np.asarray(data["mu_tr"]), dtype=torch.float32)
    sg_tr = torch.tensor(np.asarray(data["sg_tr"]), dtype=torch.float32)
    cut_tr = np.asarray(data["meta_tr"])[:, 1].astype(float)
    tau_tr = torch.tensor(cut_tr / np.maximum(np.asarray(data["mu_tr"]), 1e-6), dtype=torch.float32)

    mu_va = torch.tensor(np.asarray(data["mu_va"]), dtype=torch.float32)
    sg_va = torch.tensor(np.asarray(data["sg_va"]), dtype=torch.float32)
    cut_va = np.asarray(data["meta_va"])[:, 1].astype(float)
    tau_va = torch.tensor(cut_va / np.maximum(np.asarray(data["mu_va"]), 1e-6), dtype=torch.float32)

    units = np.asarray(data["meta_va"])[:, 0]
    units_tr = np.asarray(data["meta_tr"])[:, 0]

    # ---- calibration / test split (computed BEFORE any label-dependent choice) ----
    if args.split == "engine":
        calib_idx, test_idx = engine_calib_split(units, calib_frac=args.calib_frac, seed=args.seed)
        split_desc = {"calib": int(len(calib_idx)), "test": int(len(test_idx)), "mode": "engine"}
    else:
        split = block_split(units, cut_va, seed=args.seed)
        calib_idx, test_idx = split.calib, split.test
        split_desc = dict(split.describe(), mode="block")
    calib_mask = np.zeros(len(units), dtype=bool)
    test_mask = np.zeros(len(units), dtype=bool)
    calib_mask[calib_idx] = True
    test_mask[test_idx] = True

    # ---- PU-CQR: conditional quantile heads + conformal calibration ----
    if args.method == "cqr":
        from src.punet.cqr import QuantileHeads, train_quantile_heads
        from src.punet.evaluate import evaluate_cqr_intervals

        _sg_tr, _sg_va = sg_tr, sg_va
        if args.ablate == "A2":
            _sg_tr = torch.ones_like(sg_tr)
            _sg_va = torch.ones_like(sg_va)
        elif args.ablate == "A4":
            _g = torch.Generator().manual_seed(args.seed)
            _sg_tr = _sg_tr[torch.randperm(_sg_tr.numel(), generator=_g)]
            _g2 = torch.Generator().manual_seed(args.seed + 1)
            _sg_va = _sg_va[torch.randperm(_sg_va.numel(), generator=_g2)]

        heads = QuantileHeads(h_tr.shape[1])
        heads, _lo_tr, _hi_tr, _loss = train_quantile_heads(
            heads, h_tr, _sg_tr, tau_tr, y_tr,
            epochs=args.head_epochs, lr=args.head_lr, device=device, seed=args.seed,
        )
        with torch.no_grad():
            _lo_va, _hi_va = heads(h_va.to(device), _sg_va.to(device), tau_va.to(device))
        result = evaluate_cqr_intervals(
            units, cut_va, np.asarray(data["mu_va"]), y_va.numpy(),
            _lo_va.cpu().numpy(), _hi_va.cpu().numpy(), calib_mask, test_mask, delta=args.delta,
        )
        result.update(split=split_desc, method="cqr", ablate=args.ablate,
                      n_val=int(len(units)), head_loss=float(_loss),
                      sigma_mode=args.sigma_mode)
        return result, None

    # ---- adaptive scaling: rho estimated on CALIBRATION data only (no test labels) ----
    gamma = 0.0
    rho_hat = float("nan")
    if args.sigma_mode == "auto_gamma":
        # pick gamma on a held-out tuning block, never on the test block
        rng = np.random.default_rng(args.seed + 7)
        _cal_units = np.unique(units[calib_mask])
        rng.shuffle(_cal_units)
        n_tune = max(1, int(round(0.4 * _cal_units.size)))
        tune_units = _cal_units[:n_tune]
        tune_mask = np.isin(units, tune_units)
        inner_calib_mask = calib_mask & ~tune_mask
        best = (float("inf"), 0.0)
        for _g in (0.0, 0.25, 0.5, 0.75, 1.0):
            _eng = pd.Series(np.asarray(sg_va, dtype=float)).groupby(units).mean()
            _s_e = torch.tensor([float(_eng[int(u)]) for u in units], dtype=torch.float32)
            _sg_va = sg_va / _s_e.clamp_min(1e-6).pow(_g)
            _f = fit_heads(h_tr, sg_tr, tau_tr, y_tr, y_base_tr, d_model=h_tr.shape[1],
                           epochs=max(50, args.head_epochs // 2), lr=args.head_lr, lam=args.lam,
                           delta=args.delta, device=device, seed=args.seed,
                           use_scale_head=args.ablate != "A1", use_bias_head=args.ablate != "A6",
                           scale_simple=(args.scale_mode == "power"))
            with torch.no_grad():
                _w = _f["scale_head"](h_va.to(device), _sg_va.to(device), tau_va.to(device)).cpu()
                _yh = y_base_va + _f["bias_head"](h_va.to(device), _sg_va.to(device)).cpu()
            from src.punet.metrics import winkler as _wink
            _scores = np.abs(y_va.numpy()[inner_calib_mask] - _yh.numpy()[inner_calib_mask]) / _w.numpy()[inner_calib_mask]
            _q = float(np.quantile(_scores, 0.9, method="higher"))
            _lo = _yh.numpy()[tune_mask] - _q * _w.numpy()[tune_mask]
            _hi = _yh.numpy()[tune_mask] + _q * _w.numpy()[tune_mask]
            _wk = _wink(y_va.numpy()[tune_mask], _lo, _hi, alpha=args.delta)
            if _wk < best[0]:
                best = (_wk, _g)
        gamma = best[1]
        calib_mask = inner_calib_mask
        print(f"[auto_gamma] picked gamma={gamma:.2f} (tune Winkler {best[0]:.2f})", flush=True)

    if args.sigma_mode == "adaptive":
        from scipy.stats import spearmanr
        _err = np.abs(y_base_va.numpy() - y_va.numpy())
        rho_hat = float(spearmanr(np.asarray(sg_va)[calib_mask], _err[calib_mask]).statistic)
        gamma = 1.0 if rho_hat < 0.0 else 0.0
        print(f"[adaptive] rho_hat(calib)={rho_hat:+.3f} gamma={gamma:.1f}", flush=True)

    def _apply(sg, units_arr):
        if args.sigma_mode in ("raw", "adaptive") and gamma == 0.0:
            return sg
        eng = pd.Series(np.asarray(sg, dtype=float)).groupby(units_arr).mean()
        s_e = torch.tensor([float(eng[int(u)]) for u in units_arr], dtype=torch.float32)
        if args.sigma_mode == "engine_rel" or (args.sigma_mode == "adaptive" and gamma > 0):
            return sg / s_e.clamp_min(1e-6)
        if args.sigma_mode == "auto_gamma":
            return sg / s_e.clamp_min(1e-6).pow(gamma)
        if args.sigma_mode == "blend":
            lam = float(args.sigma_blend)
            return (1.0 - lam) * s_e + lam * sg
        return sg

    sg_tr = _apply(sg_tr, units_tr)
    sg_va = _apply(sg_va, units)

    if args.sigma_mode == "rel_life":
        sg_va = sg_va / mu_va.clamp_min(1e-3)
        sg_tr = sg_tr / mu_tr.clamp_min(1e-3)

    if args.ablate == "A2":
        sg_tr = torch.ones_like(sg_tr)
        sg_va = torch.ones_like(sg_va)
    elif args.ablate == "A4":
        g = torch.Generator().manual_seed(args.seed)
        sg_tr = sg_tr[torch.randperm(sg_tr.numel(), generator=g)]
        g2 = torch.Generator().manual_seed(args.seed + 1)
        sg_va = sg_va[torch.randperm(sg_va.numel(), generator=g2)]

    fitted = fit_heads(
        h_tr, sg_tr, tau_tr, y_tr, y_base_tr,
        d_model=h_tr.shape[1], epochs=args.head_epochs, lr=args.head_lr,
        lam=args.lam, delta=args.delta, device=device, seed=args.seed,
        use_scale_head=args.ablate != "A1",
        use_bias_head=args.ablate != "A6",
        scale_simple=(args.scale_mode == "power"),
    )

    with torch.no_grad():
        w_va = torch.ones_like(y_va) if fitted["scale_head"] is None else fitted["scale_head"](
            h_va.to(device), sg_va.to(device), tau_va.to(device)).cpu()
        if fitted["bias_head"] is None:
            y_hat_va = y_base_va
        else:
            y_hat_va = y_base_va + fitted["bias_head"](h_va.to(device), sg_va.to(device)).cpu()

    strata = None
    if args.calib_strata != "none":
        eng_sigma = pd.Series(np.asarray(sg_va, dtype=float)).groupby(units).mean()
        labels = pd.qcut(eng_sigma, 3, labels=False, duplicates="drop")
        mapping = {int(u): int(v) for u, v in labels.items()}
        sigma_bin = np.array([mapping[int(u)] for u in units], dtype=int)
        regime = data.get("regime_va")
        if args.calib_strata in ("regime", "regime_sigma") and regime is not None:
            strata = regime.astype(int) * 3 + sigma_bin if args.calib_strata == "regime_sigma" else regime.astype(int)
        else:
            strata = sigma_bin

    eng_sigma_d = pd.Series(np.asarray(sg_va, dtype=float)).groupby(units).mean()
    difficulty = np.array([float(eng_sigma_d[int(u)]) for u in units])

    result = evaluate_intervals(
        units, cut_va, np.asarray(data["mu_va"]), y_va.numpy(),
        y_hat_va.numpy(), w_va.numpy(), calib_mask, test_mask, delta=args.delta,
        strata=strata, weighted=(args.calib_mode == "weighted"), difficulty=difficulty,
        two_component=(args.calib_mode == "two_component"),
    )
    result["split"] = split_desc
    result["head_loss"] = fitted["final_loss"]
    result["ablate"] = args.ablate
    result["sigma_mode"] = args.sigma_mode
    result["rho_hat_calib"] = rho_hat
    result["gamma"] = gamma
    result["n_val"] = int(len(units))
    result["base_rmse"] = float(np.sqrt(np.mean((y_base_va.numpy() - y_va.numpy()) ** 2)))
    return result, fitted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--head-epochs", type=int, default=100)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--split", default="engine", choices=["engine", "block"])
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--method", default="scale", choices=["scale", "cqr"])
    ap.add_argument("--sigma-mode", default="raw",
                    choices=["raw", "engine_rel", "rel_life", "blend", "adaptive", "auto_gamma"])
    ap.add_argument("--sigma-blend", type=float, default=1.0)
    ap.add_argument("--scale-mode", default="power", choices=["power", "mlp"])
    ap.add_argument("--calib-mode", default="stratified", choices=["stratified", "weighted", "two_component"])
    ap.add_argument("--calib-strata", default="regime_sigma",
                    choices=["none", "sigma", "regime", "regime_sigma"])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--prior-dir", default="experiments/ida/prior")
    ap.add_argument("--out", default="experiments/punet")
    ap.add_argument("--stage", default="all", choices=["backbone", "heads", "all"])
    ap.add_argument("--reuse-ckpt", action="store_true")
    ap.add_argument("--ablate", default="none", choices=["none", "A1", "A2", "A4", "A6"])
    args = ap.parse_args(argv)

    R.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / args.out / args.subset / f"f{args.fold}_s{args.seed}"
    out_dir = run_dir / args.ablate
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "backbone.pt"

    data = build(args.subset, args.fold, args.seed, args.window, args.prior_dir)
    model = make_model(device)

    if args.stage in ("backbone", "all") and not args.reuse_ckpt:
        stats = train_backbone(model, data, args, device)
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "stats": stats}, ckpt)
        print(f"saved {ckpt} best_val_rmse={stats['best_val_rmse']:.4f}", flush=True)
    elif ckpt.exists():
        blob = torch.load(ckpt, map_location=device)
        model.load_state_dict(blob["state_dict"])
        print(f"loaded {ckpt}", flush=True)

    if args.stage in ("heads", "all"):
        result, _ = run_heads(model, data, args, device)
        (out_dir / "punet.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
