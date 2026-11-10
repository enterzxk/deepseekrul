"""B2: measurement-quality-aware failure-time updating (strictly causal).

Mechanism (Codex D035):
  * causal encoder over the raw window (GRU), so h_t only sees x_{<=t}
  * one-step predictor  x_hat_t = p(h_{t-1})   ->  innovation e_t = x_t - x_hat_t
  * quality signals     c_t = fraction of sensors with |e| above the sensor scale
                        p_t = EMA of ||e_t|| (persistence)
  * gate                alpha_t = sigmoid(a^T [h_t, c_t, p_t])
  * update              F_t = F_{t-1} + alpha_t * delta_t,
                        delta_t = D * tanh(v(h_t)^T e_t)      (bounded, zero-preserving)
  * output              y_t = clamp(F_t - t, 0, 125)
  * supervision         every prefix t is supervised with min(125, raw + (cutoff - t))

Arms
  B2       full quality gate
  B2-G     generic learned gain: gate input is h_t only (no c_t, p_t)

Both arms share the encoder, predictor, delta head and supervision; only the
gate input differs, so the comparison isolates the quality signal.
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
D_MAX = 30.0
EMA = 0.9


class B2Model(nn.Module):
    def __init__(self, d_in: int = 14, d_h: int = 64, quality: bool = True) -> None:
        super().__init__()
        self.quality = quality
        self.gru = nn.GRU(d_in, d_h, batch_first=True)
        self.pred = nn.Linear(d_h, d_in)
        self.init_F = nn.Linear(d_h, 1)
        self.v = nn.Linear(d_h, d_in)
        self.gate = nn.Linear(d_h + 2, 1) if quality else nn.Linear(d_h, 1)
        self.register_buffer("scale", torch.ones(d_in))

    def forward(self, x):
        B, W, _ = x.shape
        h, _ = self.gru(x)
        xhat = self.pred(h)                       # xhat_t from h_t (predicts x_{t+1})
        e = torch.zeros_like(x)
        e[:, 1:] = x[:, 1:] - xhat[:, :-1]        # strictly causal innovation

        scale = self.scale.to(e.device).clamp(min=1e-6)          # fixed from TRAINING data (causal)
        big = (e.abs() > scale).float()
        c = big.mean(dim=-1, keepdim=True)                        # (B, W, 1)

        idx = torch.arange(W, device=x.device)
        diff = (idx[:, None] - idx[None, :]).clamp(min=0).to(x.dtype)
        wmat = torch.where(idx[:, None] >= idx[None, :], torch.full_like(diff, EMA).pow(diff),
                           torch.zeros_like(diff))
        enorm = e.norm(dim=-1)                                    # (B, W)
        p = torch.einsum("tw,bw->bt", wmat * (1 - EMA), enorm).unsqueeze(-1)

        if self.quality:
            g = torch.sigmoid(self.gate(torch.cat([h, c, p], dim=-1))).squeeze(-1)
        else:
            g = torch.sigmoid(self.gate(h)).squeeze(-1)
        delta = D_MAX * torch.tanh((self.v(h) * e).sum(dim=-1))   # (B, W)
        F0 = self.init_F(h[:, 0]).squeeze(-1)                     # (B,)
        Ft = F0.unsqueeze(1) + torch.cumsum(g * delta, dim=1)     # (B, W)
        t = torch.arange(W, device=x.device).float().unsqueeze(0)
        y = torch.clamp(Ft - t, 0.0, CAP)
        return y, g, delta


class B2Set(Dataset):
    """Window + per-step prefix RUL labels."""

    def __init__(self, base, labels):
        self.base = base
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, cond, extras, prior = self.base[i]
        return x, self.labels[i]


def split_es_test(units, T, seed=ES_SEED):
    u = np.array(sorted(units))
    life = np.array([float(T[int(x)]) for x in u])
    rng = np.random.default_rng(seed)
    order = np.argsort(life + rng.normal(0.0, 1e-6, size=len(life)))
    es, te = [], []
    for i, idx in enumerate(order):
        (es if i % 2 == 0 else te).append(u[idx])
    return np.array(es), np.array(te)


def prefix_labels(raw, cutoff, W):
    """RUL at inner step t of a window ending at `cutoff`.

    The window covers cycles [cutoff-W+1, cutoff]; at inner step t the absolute
    cycle is cutoff-W+1+t and the failure cycle is cutoff+raw, so
    RUL_t = raw + (W-1) - t.  `cutoff` is unused (kept for signature stability).
    """
    t = np.arange(W)[None, :]
    return np.clip(raw[:, None] + (W - 1) - t, 0.0, CAP).astype(np.float32)


def engine_level(pred: pd.DataFrame) -> dict:
    e = pred["y_hat"].to_numpy() - pred["y_true"].to_numpy()
    g = pd.DataFrame({"u": pred["unit"], "e2": e ** 2}).groupby("u")["e2"].mean()
    return dict(rmse_macro=float(np.sqrt(g.mean())), n_engines=int(len(g)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", required=True, choices=["B2", "B2-G"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="experiments/tim_sprint")
    args = ap.parse_args(argv)

    R.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, df = R.outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]
    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva_, mva, cva) = R.prep_fold(
        df, args.subset, args.fold, tr_u, va_u, window=args.window
    )
    T = df.groupby("unit")["cycle"].max().to_dict()
    raw_tr = np.array([T[int(u)] - c for u, c in mtr], dtype=float)
    raw_va = np.array([T[int(u)] - c for u, c in mva], dtype=float)
    lab_tr = prefix_labels(raw_tr, mtr[:, 1].astype(float), args.window)
    lab_va = prefix_labels(raw_va, mva[:, 1].astype(float), args.window)

    es_u, te_u = split_es_test(va_u, T)
    es_idx = np.isin(mva[:, 0], es_u)
    te_idx = np.isin(mva[:, 0], te_u)

    empty = np.zeros((len(xtr), 2), np.float32)
    empty_va = np.zeros((len(xva), 2), np.float32)
    tr_loader = DataLoader(B2Set(R.WindowSet(xtr, ytr, None, etr, empty), lab_tr), batch_size=256, shuffle=True)
    es_loader = DataLoader(B2Set(R.WindowSet(xva[es_idx], yva[es_idx], None, eva_[es_idx], empty_va[es_idx]),
                                 lab_va[es_idx]), batch_size=512, shuffle=False)
    te_loader = DataLoader(B2Set(R.WindowSet(xva[te_idx], yva[te_idx], None, eva_[te_idx], empty_va[te_idx]),
                                 lab_va[te_idx]), batch_size=512, shuffle=False)

    model = B2Model(quality=(args.arm == "B2")).to(device)
    _sc = xtr.reshape(-1, xtr.shape[-1]).std(axis=0) + 1e-6
    model.scale.copy_(torch.from_numpy(_sc.astype(np.float32)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, lab in tr_loader:
            x, lab = x.to(device), lab.to(device)
            y_hat, _, _ = model(x)
            loss = F.mse_loss(y_hat, lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        errs = []
        with torch.no_grad():
            for x, lab in es_loader:
                y_hat, _, _ = model(x.to(device))
                errs.append(((y_hat[:, -1].cpu() - lab[:, -1]) ** 2).numpy())
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
    ys, lhs = [], []
    with torch.no_grad():
        for x, lab in te_loader:
            y_hat, g, d = model(x.to(device))
            ys.append(y_hat[:, -1].cpu().numpy())
            lhs.append(lab[:, -1].numpy())
    pred = pd.DataFrame({"unit": mva[te_idx][:, 0], "cutoff": mva[te_idx][:, 1],
                         "y_true": np.concatenate(lhs), "y_hat": np.concatenate(ys)})
    met = engine_level(pred)
    met.update(arm=args.arm, subset=args.subset, fold=args.fold, seed=args.seed,
               best_es_rmse=best, best_epoch=best_epoch,
               n_params=int(sum(p.numel() for p in model.parameters())))

    out_dir = ROOT / args.out / args.subset / f"f{args.fold}_s{args.seed}_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pred.to_csv(out_dir / "preds.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(met, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(met, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
