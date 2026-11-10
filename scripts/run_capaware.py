"""Protocol-aware joint learning for cap-125 RUL labels (method C).

Fixes applied after the protocol review (docs/overnight_tim_research/04_EXPERIMENT_PROTOCOL.md):
  * validation engines are split into ES (early stopping) and TEST (reporting)
    halves, stratified by lifespan -- early stopping never sees a test engine;
  * headline metrics are engine-level (last cutoff per engine), window-pooled
    numbers are reported only as a secondary reference.

Arms
  M0  direct regression to the capped label + auxiliary event classifier
  M1  coupled output: y_hat = 125*p + (1-p)*m, p = sigmoid(event head)
  M2  M1 with a stop-gradient on p (the regression loss does not train the gate)

Gates: G-C1 M1 sub-threshold RMSE < M0 by >=2% (2 seeds same direction);
       G-C2 M2 loses that gain; G-C3 event Brier <= base-rate Brier.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ida as R  # noqa: E402

CAP = 125.0
ES_SEED = 20261110


class CapAware(nn.Module):
    def __init__(self, body: nn.Module, d_model: int = 64, prior_dim: int = 2) -> None:
        super().__init__()
        self.body = body
        self.proj = nn.Linear(prior_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.event = nn.Linear(d_model, 1)
        self.sub = nn.Linear(d_model, 1)

    def forward(self, x, cond, extras, prior):
        pooled = self.body.encode(x, cond=cond, extras=extras)
        ctx = pooled.mean(dim=1) + self.proj(prior)
        return self.event(ctx).squeeze(-1), self.sub(ctx).squeeze(-1)


class CapSet(Dataset):
    def __init__(self, base, ev):
        self.base = base
        self.ev = torch.from_numpy(np.asarray(ev, dtype=np.float32))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, cond, extras, prior = self.base[i]
        return x, y, cond, extras, prior, self.ev[i]


def split_es_test(units, T, seed=ES_SEED):
    """Alternate engines (sorted by lifespan) into ES / TEST halves."""
    u = np.array(sorted(units))
    life = np.array([float(T[int(x)]) for x in u])
    rng = np.random.default_rng(seed)
    order = np.argsort(life + rng.normal(0.0, 1e-6, size=len(life)))
    es, te = [], []
    for i, idx in enumerate(order):
        (es if i % 2 == 0 else te).append(u[idx])
    return np.array(es), np.array(te)


def combine(p_logit, m, arm: str):
    p = torch.sigmoid(p_logit)
    m_clamped = torch.clamp(m, 0.0, CAP)
    if arm == "M0":
        return m_clamped, p
    if arm == "M1":
        return CAP * p + (1.0 - p) * m_clamped, p
    if arm == "M2":
        p_sg = p.detach()
        return CAP * p_sg + (1.0 - p_sg) * m_clamped, p
    raise ValueError(arm)


def nasa_score(y, yhat):
    d = yhat - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def engine_level(pred: pd.DataFrame) -> dict:
    """One prediction per engine: the window with the largest cutoff."""
    last = pred.sort_values("cutoff").groupby("unit").tail(1)
    e = last["y_hat"].to_numpy() - last["y_true"].to_numpy()
    raw = last["y_raw"].to_numpy()
    out = dict(rmse_em=float(np.sqrt(np.mean(e ** 2))), mae_em=float(np.mean(np.abs(e))),
               score_em=float(np.mean(np.where(e < 0, np.exp(-e / 13.0) - 1.0, np.exp(e / 10.0) - 1.0))),
               n_engines=int(len(last)))
    sub = raw < CAP
    if sub.sum() >= 3:
        out["rmse_em_sub"] = float(np.sqrt(np.mean(e[sub] ** 2)))
        out["n_engines_sub"] = int(sub.sum())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", required=True, choices=["M0", "M1", "M2"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam-event", type=float, default=0.5)
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

    T = df.groupby("unit")["cycle"].max().to_dict()
    raw_tr = np.array([T[int(u)] - c for u, c in mtr], dtype=float)
    raw_va = np.array([T[int(u)] - c for u, c in mva], dtype=float)
    ev_tr = (raw_tr >= CAP).astype(np.float32)
    ev_va = (raw_va >= CAP).astype(np.float32)

    es_u, te_u = split_es_test(va_u, T)
    es_idx = np.isin(mva[:, 0], es_u)
    te_idx = np.isin(mva[:, 0], te_u)

    train_loader = DataLoader(CapSet(R.WindowSet(xtr, ytr, ctr, etr, ptr), ev_tr),
                              batch_size=256, shuffle=True)
    es_loader = DataLoader(CapSet(R.WindowSet(xva[es_idx], yva[es_idx], cva[es_idx], eva_[es_idx], pva[es_idx]),
                                  ev_va[es_idx]), batch_size=512, shuffle=False)
    te_loader = DataLoader(CapSet(R.WindowSet(xva[te_idx], yva[te_idx], cva[te_idx], eva_[te_idx], pva[te_idx]),
                                  ev_va[te_idx]), batch_size=512, shuffle=False)

    body = R.CandidateModel(
        num_sensors=14, window_size=args.window, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
        use_graph=False, use_dual_head=False,
    )
    model = CapAware(body).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, cond, extras, prior, ev in train_loader:
            x, y = x.to(device), y.to(device)
            cond, extras, prior, ev = cond.to(device), extras.to(device), prior.to(device), ev.to(device)
            p_logit, m = model(x, cond, extras, prior)
            y_hat, _ = combine(p_logit, m, args.arm)
            loss = F.mse_loss(y_hat, y.reshape(-1)) + args.lam_event * F.binary_cross_entropy_with_logits(p_logit, ev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        sched.step()

        model.eval()
        errs = []
        with torch.no_grad():
            for x, y, cond, extras, prior, _ev in es_loader:
                p_logit, m = model(x.to(device), cond.to(device), extras.to(device), prior.to(device))
                y_hat, _ = combine(p_logit, m, args.arm)
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

    ys, yhs, ps = [], [], []
    with torch.no_grad():
        for x, y, cond, extras, prior, _ev in te_loader:
            p_logit, m = model(x.to(device), cond.to(device), extras.to(device), prior.to(device))
            y_hat, p = combine(p_logit, m, args.arm)
            ys.append(y.numpy()); yhs.append(y_hat.cpu().numpy()); ps.append(p.cpu().numpy())
    y_all = np.concatenate(ys); yh_all = np.concatenate(yhs); p_all = np.concatenate(ps)

    pred = pd.DataFrame({"unit": mva[te_idx][:, 0], "cutoff": mva[te_idx][:, 1],
                         "y_true": y_all, "y_raw": raw_va[te_idx],
                         "y_hat": yh_all, "p_event": p_all})
    met = engine_level(pred)
    met.update(arm=args.arm, subset=args.subset, fold=args.fold, seed=args.seed,
               best_es_rmse=best, best_epoch=best_epoch,
               brier=float(np.mean((p_all - ev_va[te_idx]) ** 2)),
               base_brier=float(np.mean((ev_va[te_idx].mean() - ev_va[te_idx]) ** 2)),
               event_rate=float(ev_va[te_idx].mean()),
               n_windows=int(len(pred)),
               pooled_rmse=float(np.sqrt(np.mean((yh_all - y_all) ** 2))),
               n_es_engines=int(len(es_u)), n_test_engines=int(len(te_u)))

    out_dir = ROOT / args.out / args.subset / f"f{args.fold}_s{args.seed}_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pred.to_csv(out_dir / "preds.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(met, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(met, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
