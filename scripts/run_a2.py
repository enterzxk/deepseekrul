"""A2: event-conditioned level continuity + condition-modulated degradation rate.

Mechanism (after the L2 review, docs/overnight_tim_research/NO_GO_REVIEW.md):
  * degradation level  d_t = sum_{tau<=t} dd_tau,   dd_tau >= 0  (non-decreasing)
  * rate modulation    dd_t = softplus(w^T h_t) * softplus(b + gamma^T oc_t)
    -> the operating condition MAY change the damage accumulation rate
  * transient          r_t = rho * r_{t-1} + eta^T (oc_t - oc_{t-1}),  rho = sigmoid
  * event continuity   at observable OC-switch steps the increment dd_s is penalised
    only when it exceeds 1.25x the mean non-switch increment (scale-free)
  * readout            y = head([ctx, d_T, r_T])

Arms
  R       direct regression; d_T and r_T are replaced by zeros (same head width)
  A2      full mechanism
  A2-NE   no event-continuity penalty (lambda_cont = 0)
  A2-NR   no rate modulation (gamma = 0, rate = 1)

Evaluation follows the corrected protocol: ES/TEST engine split for early
stopping, engine-level metrics on TEST engines.
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

CAP = 125.0
ES_SEED = 20261110


class A2Model(nn.Module):
    def __init__(self, body: nn.Module, d_model: int = 64, prior_dim: int = 2) -> None:
        super().__init__()
        self.body = body
        self.proj = nn.Linear(prior_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.w_delta = nn.Linear(d_model, 1)
        self.gamma = nn.Linear(3, 1)
        self.b_rate = nn.Parameter(torch.zeros(1))
        self.eta = nn.Linear(3, 1)
        self.rho_raw = nn.Parameter(torch.zeros(1))
        self.head = nn.Sequential(nn.Linear(d_model + 2, 64), nn.ReLU(), nn.Linear(64, 1))
        # quality gate (A2-QG): suppress increments when sensor changes are incoherent/transient
        self.qg = nn.Linear(d_model + 2, 1)
        self.qg_g = nn.Linear(d_model, 1)
        self.register_buffer("xscale", torch.ones(14))

    def forward(self, x, cond, extras, prior, arm: str):
        if cond is None:  # FD001/FD003 have no operating-condition planes
            cond = torch.zeros(x.shape[0], x.shape[1], 3, device=x.device, dtype=x.dtype)
        pooled = self.body.encode(x, cond=cond, extras=extras)      # (B, W, d)
        ctx = pooled.mean(dim=1) + self.proj(prior)                 # (B, d)
        w = F.softplus(self.w_delta(pooled).squeeze(-1))            # (B, W)
        if arm in ("A2-QG", "A2-QGg"):
            dx = torch.zeros_like(x)
            dx[:, 1:] = x[:, 1:] - x[:, :-1]
            sc = self.xscale.to(x.device).clamp(min=1e-6)
            c = (dx.abs() > sc).float().mean(dim=-1, keepdim=True)
            iq = torch.arange(x.shape[1], device=x.device)
            dq = (iq[:, None] - iq[None, :]).clamp(min=0).to(x.dtype)
            wq = torch.where(iq[:, None] >= iq[None, :], torch.full_like(dq, 0.9).pow(dq), torch.zeros_like(dq))
            pema = torch.einsum("tw,bw->bt", wq * 0.1, dx.norm(dim=-1)).unsqueeze(-1)
            if arm == "A2-QG":
                g = torch.sigmoid(self.qg(torch.cat([pooled, c, pema], dim=-1))).squeeze(-1)
            else:
                g = torch.sigmoid(self.qg_g(pooled)).squeeze(-1)
            w = w * g
        if arm == "A2-NR":
            rate = torch.ones_like(w)
        else:
            rate = F.softplus(self.b_rate + self.gamma(cond).squeeze(-1))
        dd = w * rate                                               # (B, W) >= 0
        d = torch.cumsum(dd, dim=1)
        d_T = d[:, -1]

        doc = torch.zeros_like(cond)
        doc[:, 1:] = cond[:, 1:] - cond[:, :-1]
        eta = self.eta(doc).squeeze(-1)                              # (B, W)
        rho = torch.sigmoid(self.rho_raw)
        # closed form of r_t = rho*r_{t-1} + eta_t  ->  r = W @ eta, W[t,k]=rho^{t-k} (k<=t)
        W = cond.shape[1]
        idx = torch.arange(W, device=cond.device)
        diff = (idx[:, None] - idx[None, :]).clamp(min=0).to(eta.dtype)
        wmat = torch.where(idx[:, None] >= idx[None, :], rho.pow(diff), torch.zeros_like(diff))
        r = torch.einsum("tw,bw->bt", wmat, eta)
        r_T = r[:, -1]

        if arm == "R":
            d_T = torch.zeros_like(d_T)
            r_T = torch.zeros_like(r_T)
        feat = torch.cat([ctx, d_T.unsqueeze(-1), r_T.unsqueeze(-1)], dim=-1)
        return self.head(feat).squeeze(-1), dd, doc


def split_es_test(units, T, seed=ES_SEED):
    u = np.array(sorted(units))
    life = np.array([float(T[int(x)]) for x in u])
    rng = np.random.default_rng(seed)
    order = np.argsort(life + rng.normal(0.0, 1e-6, size=len(life)))
    es, te = [], []
    for i, idx in enumerate(order):
        (es if i % 2 == 0 else te).append(u[idx])
    return np.array(es), np.array(te)


def engine_level(pred: pd.DataFrame) -> dict:
    last = pred.sort_values("cutoff").groupby("unit").tail(1)
    e = last["y_hat"].to_numpy() - last["y_true"].to_numpy()
    return dict(rmse_em=float(np.sqrt(np.mean(e ** 2))), mae_em=float(np.mean(np.abs(e))),
                n_engines=int(len(last)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", required=True, choices=["R", "A2", "A2-NE", "A2-NR", "A2-PS", "A2-GL", "A2-QG", "A2-QGg"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam-cont", type=float, default=0.5)
    ap.add_argument("--shuffle-events", action="store_true")
    ap.add_argument("--prior-dir", default="experiments/ida/prior_boot")
    ap.add_argument("--out", default="experiments/tim_sprint")
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
    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva_, mva, cva) = R.prep_fold(
        df, args.subset, args.fold, tr_u, va_u, window=args.window
    )
    ptr, _ = R.prior_vectors(mtr, sp, args.fold, "M1", median_T)
    pva, _ = R.prior_vectors(mva, sp, args.fold, "M1", median_T)

    # OC-switch threshold from the TRAINING condition planes only
    if ctr is not None:
        doc_tr = np.sqrt((np.diff(ctr, axis=1) ** 2).sum(axis=-1)).reshape(-1)
        tau = float(np.quantile(doc_tr, 0.90))
    else:
        tau = float("inf")

    T = df.groupby("unit")["cycle"].max().to_dict()
    es_u, te_u = split_es_test(va_u, T)
    es_idx = np.isin(mva[:, 0], es_u)
    te_idx = np.isin(mva[:, 0], te_u)

    train_loader = DataLoader(R.WindowSet(xtr, ytr, ctr, etr, ptr), batch_size=256, shuffle=True)
    es_loader = DataLoader(R.WindowSet(xva[es_idx], yva[es_idx], cva[es_idx] if cva is not None else None,
                                       eva_[es_idx], pva[es_idx]), batch_size=512, shuffle=False)
    te_loader = DataLoader(R.WindowSet(xva[te_idx], yva[te_idx], cva[te_idx] if cva is not None else None,
                                       eva_[te_idx], pva[te_idx]), batch_size=512, shuffle=False)

    body = R.CandidateModel(
        num_sensors=14, window_size=args.window, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
        use_graph=False, use_dual_head=False,
    )
    model = A2Model(body).to(device)
    _xs = xtr.reshape(-1, xtr.shape[-1]).std(axis=0) + 1e-6
    model.xscale.copy_(torch.from_numpy(_xs.astype(np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    lam_cont = 0.0 if args.arm in ("R", "A2-NE", "A2-QG", "A2-QGg") else args.lam_cont
    if args.arm == "A2-PS":
        args.shuffle_events = True

    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, cond, extras, prior in train_loader:
            x, y = x.to(device), y.to(device)
            cond, extras, prior = cond.to(device), extras.to(device), prior.to(device)
            y_hat, dd, doc = model(x, cond, extras, prior, args.arm)
            loss = F.mse_loss(y_hat, y.reshape(-1))
            if lam_cont > 0 and cond is not None:
                if args.arm == "A2-GL":
                    # global excess penalty on EVERY step: scale-matched control
                    eps = 1.25 * dd.mean().detach() + 1e-6
                    excess = torch.clamp(dd - eps, min=0.0)
                    loss = loss + lam_cont * (excess ** 2).mean() / (eps ** 2)
                else:
                    norm = doc.norm(dim=-1)
                    switch = norm > tau
                    if args.shuffle_events:
                        perm = torch.randperm(switch.shape[1], device=switch.device)
                        switch = switch[:, perm]
                    switch = switch.cpu()
                    if switch.any():
                        eps = 1.25 * dd[~switch].mean().detach() + 1e-6
                        excess = torch.clamp(dd[switch] - eps, min=0.0)
                        loss = loss + lam_cont * (excess ** 2).mean() / (eps ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        sched.step()

        model.eval()
        errs = []
        with torch.no_grad():
            for x, y, cond, extras, prior in es_loader:
                y_hat, _, _ = model(x.to(device), cond.to(device) if cond is not None else None,
                                    extras.to(device), prior.to(device), args.arm)
                errs.append(((y_hat.cpu() - y.reshape(-1)) ** 2).numpy())
        val = float(np.sqrt(np.concatenate(errs).mean()))
        if val < best - 1e-4:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
        if epoch % 10 == 0:
            print(f"[{args.subset} {args.arm}] ep{epoch:03d} es_rmse={val:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    ys, yhs = [], []
    dd_sw, dd_non, dts, yts = [], [], [], []
    with torch.no_grad():
        for x, y, cond, extras, prior in te_loader:
            y_hat, dd, doc = model(x.to(device), cond.to(device) if cond is not None else None,
                                   extras.to(device), prior.to(device), args.arm)
            ys.append(y.numpy()); yhs.append(y_hat.cpu().numpy())
            if cond is not None:
                norm = doc.norm(dim=-1)
                switch = (norm > tau).cpu()
                dd_c = dd.detach().cpu()
                dd_sw.append(dd_c[switch].numpy())
                dd_non.append(dd_c[~switch].numpy())
                dts.append(dd_c.sum(dim=1).numpy())
                yts.append(y.numpy())
    y_all = np.concatenate(ys); yh_all = np.concatenate(yhs)

    mech = {}
    if dd_sw and dd_non:
        sw = np.concatenate(dd_sw); non = np.concatenate(dd_non)
        mech["dd_switch_mean"] = float(sw.mean()) if sw.size else None
        mech["dd_nonswitch_mean"] = float(non.mean()) if non.size else None
        mech["switch_ratio"] = float(sw.mean() / (non.mean() + 1e-8)) if sw.size and non.size else None
        d_t = np.concatenate(dts); y_t = np.concatenate(yts)
        if d_t.size > 3 and np.std(d_t) > 1e-9:
            mech["spearman_dT_rul"] = float(pd.Series(d_t).corr(pd.Series(y_t), method="spearman"))

    pred = pd.DataFrame({"unit": mva[te_idx][:, 0], "cutoff": mva[te_idx][:, 1],
                         "y_true": y_all, "y_hat": yh_all})
    met = engine_level(pred)
    met.update(arm=args.arm, subset=args.subset, fold=args.fold, seed=args.seed,
               best_es_rmse=best, best_epoch=best_epoch,
               n_windows=int(len(pred)), pooled_rmse=float(np.sqrt(np.mean((yh_all - y_all) ** 2))),
               n_params=int(sum(p.numel() for p in model.parameters())),
               tau=tau if np.isfinite(tau) else None, **mech)

    out_dir = ROOT / args.out / args.subset / f"f{args.fold}_s{args.seed}_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pred.to_csv(out_dir / "preds.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(met, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(met, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
