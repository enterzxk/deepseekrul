"""B2v2: quality-aware failure-time updating with a richer causal input.

Same mechanism as run_b2.py (causal innovation -> quality gate -> bounded
failure-time update), but the causal encoder now also receives the current
operating condition and the window prior features, which are available at
time t (no future information), and the GRU is wider.

Arms
  B2       full quality gate (cross-sensor consistency c_t + persistence p_t)
  B2-G     generic learned gain (gate sees h_t only)

Only the gate input differs between arms; the encoder, predictor, delta head,
labels and supervision are identical.
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


class B2V2(nn.Module):
    def __init__(self, d_in: int = 19, d_h: int = 128, quality: bool = True) -> None:
        super().__init__()
        self.quality = quality
        self.gru = nn.GRU(d_in, d_h, num_layers=2, batch_first=True, dropout=0.1)
        self.pred = nn.Linear(d_h, 14)
        self.init_F = nn.Linear(d_h, 1)
        self.v = nn.Linear(d_h, 14)
        self.gate = nn.Linear(d_h + 2, 1) if quality else nn.Linear(d_h, 1)

    def forward(self, inp, x_raw):
        B, W, _ = inp.shape
        h, _ = self.gru(inp)
        xhat = self.pred(h)
        e = torch.zeros_like(x_raw)
        e[:, 1:] = x_raw[:, 1:] - xhat[:, :-1]

        scale = e[:, 1:].abs().mean(dim=(0, 1)).clamp(min=1e-6)
        c = (e.abs() > scale).float().mean(dim=-1, keepdim=True)

        idx = torch.arange(W, device=inp.device)
        diff = (idx[:, None] - idx[None, :]).clamp(min=0).to(inp.dtype)
        wmat = torch.where(idx[:, None] >= idx[None, :], torch.full_like(diff, EMA).pow(diff), torch.zeros_like(diff))
        p = torch.einsum("tw,bw->bt", wmat * (1 - EMA), e.norm(dim=-1)).unsqueeze(-1)

        gate_in = torch.cat([h, c, p], dim=-1) if self.quality else h
        g = torch.sigmoid(self.gate(gate_in)).squeeze(-1)
        delta = D_MAX * torch.tanh((self.v(h) * e).sum(dim=-1))
        F0 = self.init_F(h[:, 0]).squeeze(-1)
        Ft = F0.unsqueeze(1) + torch.cumsum(g * delta, dim=1)
        t = torch.arange(W, device=inp.device).float().unsqueeze(0)
        return torch.clamp(Ft - t, 0.0, CAP), g, delta


class B2Set(Dataset):
    def __init__(self, base, labels):
        self.base = base
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, cond, extras, prior = self.base[i]
        return x, cond if cond is not None else torch.zeros(x.shape[0], 3), prior, self.labels[i]


def split_es_test(units, T, seed=ES_SEED):
    u = np.array(sorted(units))
    life = np.array([float(T[int(x)]) for x in u])
    rng = np.random.default_rng(seed)
    order = np.argsort(life + rng.normal(0.0, 1e-6, size=len(life)))
    es, te = [], []
    for i, idx in enumerate(order):
        (es if i % 2 == 0 else te).append(u[idx])
    return np.array(es), np.array(te)


def prefix_labels(raw, W):
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
    sp = R.StaircasePrior(table, {int(k): v for k, v in ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())}, args.subset)

    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva_, mva, cva) = R.prep_fold(df, args.subset, args.fold, tr_u, va_u, window=args.window)
    ptr, _ = R.prior_vectors(mtr, sp, args.fold, "M1", median_T)
    pva, _ = R.prior_vectors(mva, sp, args.fold, "M1", median_T)
    T = df.groupby("unit")["cycle"].max().to_dict()
    raw_tr = np.array([T[int(u)] - c for u, c in mtr], dtype=float)
    raw_va = np.array([T[int(u)] - c for u, c in mva], dtype=float)
    lab_tr = prefix_labels(raw_tr, args.window)
    lab_va = prefix_labels(raw_va, args.window)

    es_u, te_u = split_es_test(va_u, T)
    es_idx = np.isin(mva[:, 0], es_u)
    te_idx = np.isin(mva[:, 0], te_u)

    tr_loader = DataLoader(B2Set(R.WindowSet(xtr, ytr, ctr, etr, ptr), lab_tr), batch_size=256, shuffle=True)
    es_loader = DataLoader(B2Set(R.WindowSet(xva[es_idx], yva[es_idx], cva[es_idx] if cva is not None else None,
                                            eva_[es_idx], pva[es_idx]), lab_va[es_idx]), batch_size=512, shuffle=False)
    te_loader = DataLoader(B2Set(R.WindowSet(xva[te_idx], yva[te_idx], cva[te_idx] if cva is not None else None,
                                            eva_[te_idx], pva[te_idx]), lab_va[te_idx]), batch_size=512, shuffle=False)

    model = B2V2(quality=(args.arm == "B2")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    def make_input(x, cond, prior):
        if cond is None:
            cond = torch.zeros(x.shape[0], x.shape[1], 3, device=x.device)
        pr = prior.unsqueeze(1).expand(-1, x.shape[1], -1)
        return torch.cat([x, cond, pr], dim=-1)

    best, best_epoch, best_state = float("inf"), -1, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, cond, prior, lab in tr_loader:
            x, lab = x.to(device), lab.to(device)
            inp = make_input(x, cond.to(device), prior.to(device))
            y_hat, _, _ = model(inp, x)
            loss = F.mse_loss(y_hat, lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        errs = []
        with torch.no_grad():
            for x, cond, prior, lab in es_loader:
                x = x.to(device)
                inp = make_input(x, cond.to(device), prior.to(device))
                y_hat, _, _ = model(inp, x)
                errs.append(((y_hat[:, -1].cpu() - lab[:, -1]) ** 2).numpy())
        val = float(np.sqrt(np.concatenate(errs).mean()))
        if val < best - 1e-4:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
        if epoch % 10 == 0:
            print(f"[{args.subset} {args.arm}v2] ep{epoch:03d} es_rmse={val:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ys, lhs = [], []
    with torch.no_grad():
        for x, cond, prior, lab in te_loader:
            x = x.to(device)
            inp = make_input(x, cond.to(device), prior.to(device))
            y_hat, _, _ = model(inp, x)
            ys.append(y_hat[:, -1].cpu().numpy()); lhs.append(lab[:, -1].numpy())
    pred = pd.DataFrame({"unit": mva[te_idx][:, 0], "cutoff": mva[te_idx][:, 1],
                         "y_true": np.concatenate(lhs), "y_hat": np.concatenate(ys)})
    met = engine_level(pred)
    met.update(arm=args.arm + "v2", subset=args.subset, fold=args.fold, seed=args.seed,
               best_es_rmse=best, best_epoch=best_epoch,
               n_params=int(sum(p.numel() for p in model.parameters())))

    out_dir = ROOT / args.out / args.subset / f"f{args.fold}_s{args.seed}_{args.arm}v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pred.to_csv(out_dir / "preds.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(met, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(met, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
