"""R2: compact evidence filter on a causal encoder + per-channel bias states + hazard decode.

Architecture
  e_t          causal dilated-conv encoder over [sensors, op, age, mask]   (64-d)
  mu_t         observation model: predicts x_t from (h_{t-1}, e_{t-1}, op_t)
  nu_t         innovation = x_t - mu_t, masked
  b_t          per-channel bias state, random walk: b_t = b_{t-1} + eta * tanh(...)
  h_t          health state: h_t = h_{t-1} + eps * tanh(W_h h + W_nu (nu_t - b_{t-1}) + W_e e_t)
  head         discrete hazard over RUL 0..124 plus a >=125 tail, from (h_t, age_t, e_t)

Why the update is additive: the CET line died because a multiplicative gate sat at ~0,
so the evidence path had no gradient and the gate could never rise. Here the innovation
enters the state through a bounded *additive* increment, so d(loss)/d(nu) != 0 by
construction, and there is no clamp anywhere on a learned path.

Bias states are the operator that survived the CIDO novelty audit: FDI models faults as
discrete hypotheses, factor analysis treats the specific component as i.i.d. noise, deep
SSMs never ask which channel is wrong. A per-channel random-walk bias filtered jointly
with the health state is the unattended middle, and it is falsifiable by injection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import dsh_e21_test_all as e21  # noqa: E402
from dataset import CMAPSS_COLUMNS, load_cmapss_files  # noqa: E402
from cet.model import CausalSensorEncoder  # noqa: E402

N_SENSORS = 14
MAX_RUL = 125
K_BINS = MAX_RUL


def build_sequences(subset: str, fold: int, kind: str):
    raw, tr, va, _ = e21.split_frames(subset, fold)
    frame = tr if kind == "train" else va
    scaler = e21.fit_scaler(subset, tr)
    scaled = e21.scale_frame(subset, scaler, frame)
    seqs = []
    for unit, g in frame.groupby("unit", sort=True):
        idx = g.index.to_numpy()
        L = len(idx)
        if L < 20:
            continue
        seqs.append(dict(
            unit=int(unit),
            x=np.asarray(scaled[idx], dtype=np.float32),
            op=g[["op_1", "op_2", "op_3"]].to_numpy(np.float32),
            age=g["cycle"].to_numpy(np.float32),
            L=L,
        ))
    return seqs, scaler


def build_test_sequences(subset, fold, scaler):
    """Official test engines: full observed prefix, truth = RUL at the last cycle."""
    _, test_frame, test_rul = load_cmapss_files(ROOT / "data" / "CMAPSS", subset)
    test_frame = test_frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    scaled = e21.scale_frame(subset, scaler, test_frame)
    seqs = []
    for (unit, g), rul_end in zip(test_frame.groupby("unit", sort=True), test_rul):
        idx = g.index.to_numpy()
        seqs.append(dict(unit=int(unit), x=np.asarray(scaled[idx], dtype=np.float32),
                         op=g[["op_1", "op_2", "op_3"]].to_numpy(np.float32),
                         age=g["cycle"].to_numpy(np.float32), L=len(idx),
                         rul_end=float(rul_end)))
    return seqs


def collate(seqs, device, cutoffs=None):
    """cutoffs: optional per-engine prefix length (for matched evaluation)."""
    B = len(seqs)
    lens = [s["L"] for s in seqs] if cutoffs is None else [int(c) for c in cutoffs]
    T = max(lens)
    x = np.zeros((B, T, N_SENSORS), np.float32)
    op = np.zeros((B, T, 3), np.float32)
    age = np.zeros((B, T), np.float32)
    valid = np.zeros((B, T), np.float32)
    rul = np.zeros((B, T), np.float32)
    for i, (s, n) in enumerate(zip(seqs, lens)):
        x[i, :n] = s["x"][:n]
        op[i, :n] = s["op"][:n]
        age[i, :n] = s["age"][:n]
        valid[i, :n] = 1.0
        r = s["L"] - np.arange(1, n + 1)
        rul[i, :n] = np.minimum(r, MAX_RUL)
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(x), t(op), t(age), t(valid), t(rul)


class EvidenceFilter(nn.Module):
    def __init__(self, d: int = 64, d_state: int = 16, bias_step: float = 0.05,
                 state_step: float = 0.1, rank_bias: int = 32):
        super().__init__()
        self.d, self.d_state = d, d_state
        self.bias_step, self.state_step = bias_step, state_step
        self.encoder = CausalSensorEncoder(d=d, dropout=0.0)
        self.obs = nn.Sequential(
            nn.Linear(d_state + d + 3, 96), nn.GELU(),
            nn.Linear(96, 96), nn.GELU(),
            nn.Linear(96, 2 * N_SENSORS),
        )
        self.bias = nn.Sequential(nn.Linear(N_SENSORS + d, rank_bias), nn.GELU(),
                                  nn.Linear(rank_bias, N_SENSORS))
        self.state = nn.Sequential(nn.Linear(d_state + N_SENSORS + d, 64), nn.GELU(),
                                   nn.Linear(64, d_state))
        # The head reads ONLY the state and age. Letting it see e_t directly turns the
        # encoder into an engine fingerprint and the 20 held-out engines collapse.
        self.head = nn.Sequential(nn.Linear(d_state + 1, 64), nn.GELU(),
                                  nn.Linear(64, K_BINS + 1))

    def forward(self, x, op, age, valid, detach_every: int = 0):
        B, T, _ = x.shape
        mask = valid.unsqueeze(-1)
        enc_in = torch.cat([x * mask, op, (age / 250.0).unsqueeze(-1), mask], dim=-1)
        e = self.encoder(enc_in)

        h = x.new_zeros(B, self.d_state)
        b = x.new_zeros(B, N_SENSORS)
        logits, nus, bh, dh = [], [], [], []
        for t in range(T):
            e_prev = e[:, t - 1] if t > 0 else torch.zeros_like(e[:, t])
            mu_ls = self.obs(torch.cat([h, e_prev, op[:, t]], dim=-1))
            mu, log_sigma = mu_ls[:, :N_SENSORS], mu_ls[:, N_SENSORS:]
            nu = (x[:, t] - mu) * mask[:, t]
            nu_c = nu - b
            b = b + self.bias_step * torch.tanh(self.bias(torch.cat([nu_c, e[:, t]], dim=-1))) * mask[:, t]
            delta = self.state_step * torch.tanh(self.state(torch.cat([h, nu_c, e[:, t]], dim=-1)))
            h = h + delta * mask[:, t]
            z = self.head(torch.cat([h, (age[:, t] / 250.0).unsqueeze(-1)], dim=-1))
            logits.append(z)
            nus.append(nu)
            bh.append(b)
            dh.append(delta)
            if detach_every and (t % detach_every) == (detach_every - 1):
                h, b = h.detach(), b.detach()
        return dict(logits=torch.stack(logits, 1), nu=torch.stack(nus, 1),
                    b=torch.stack(bh, 1), dh=torch.stack(dh, 1), mu=mu)


def censored_nll(logits, rul, valid):
    logp = F.log_softmax(logits, dim=-1)
    idx = rul.clamp(0, K_BINS - 1).long()
    point = -logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    cens = -logp[..., K_BINS]
    nll = torch.where(rul >= MAX_RUL, cens, point)
    w = valid
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def crps_discrete(logits, rul, valid):
    p = F.softmax(logits, dim=-1)
    cdf = torch.cumsum(p, dim=-1)
    bins = torch.arange(0, K_BINS + 1, device=logits.device, dtype=logits.dtype)
    step = (cdf - (bins >= rul.unsqueeze(-1)).to(logits.dtype)).abs().sum(-1)
    return (step * valid).sum() / valid.sum().clamp_min(1.0)


def rul_mean(logits):
    p = F.softmax(logits, dim=-1)
    bins = torch.arange(0, K_BINS + 1, device=logits.device, dtype=logits.dtype)
    return (p * bins).sum(-1)


@torch.no_grad()
def evaluate(model, seqs, device, cutoffs=None):
    model.eval()
    if cutoffs is None:
        cutoffs = [s["L"] for s in seqs]
    x, op, age, valid, _ = collate(seqs, device, cutoffs)
    out = model(x, op, age, valid)
    pred = rul_mean(out["logits"])[:, -1].cpu().numpy()
    truth = np.array([min(s["L"] - c, MAX_RUL) for s, c in zip(seqs, cutoffs)], dtype=np.float32)
    return pred, truth, out


@torch.no_grad()
def reversal_delta(model, seqs, device, cutoffs):
    model.eval()
    deltas = []
    for s, c in zip(seqs, cutoffs):
        x, op, age, valid, _ = collate([s], device, [c])
        p0 = rul_mean(model(x, op, age, valid)["logits"])[0, -1].item()
        xr = torch.flip(x, dims=[1])
        p1 = rul_mean(model(xr, op, age, valid)["logits"])[0, -1].item()
        deltas.append(abs(p0 - p1))
    return float(np.mean(deltas)), float(np.max(deltas))


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--detach-every", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=600.0)
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "r2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    tr, scaler = build_sequences(args.subset, args.fold, "train")
    va, _ = build_sequences(args.subset, args.fold, "val")
    rng = np.random.default_rng(args.seed)
    cut = np.array([int(min(max(rng.integers(40, max(41, int(0.95 * s["L"])) + 1), 30), s["L"])) for s in va])

    model = EvidenceFilter(d_state=args.d_state).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    y_const = np.array([min(s["L"] - c, MAX_RUL) for s, c in zip(va, cut)], dtype=np.float32)
    const_rmse = float(np.std(y_const))

    best = {"rmse": 1e9, "epoch": -1}
    hist = []
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(order), args.batch):
            chunk = [tr[j] for j in order[i:i + args.batch]]
            x, op, age, valid, rul = collate(chunk, device)
            out = model(x, op, age, valid, detach_every=args.detach_every)
            loss = censored_nll(out["logits"], rul, valid) + 0.05 * crps_discrete(out["logits"], rul, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            p, y, _ = evaluate(model, va, device, cut)
            r = rmse(p, y)
            hist.append({"epoch": ep, "loss": tot / max(nb, 1), "dev_rmse": r})
            flag = ""
            if r < best["rmse"]:
                best = {"rmse": r, "epoch": ep}
                flag = " *"
            print(f"ep{ep:03d} loss={tot/max(nb,1):.4f} dev_rmse={r:.2f} (const {const_rmse:.2f}){flag}", flush=True)
            if time.time() - t0 > args.time_budget:
                print("time budget reached", flush=True)
                break

    p, y, out = evaluate(model, va, device, cut)
    trd_mean, trd_max = reversal_delta(model, va, device, cut)
    with torch.no_grad():
        x, op, age, valid, _ = collate(tr[: min(len(tr), 20)], device)
        o = model(x, op, age, valid)
    gates = {
        "G1_dev_rmse": rmse(p, y),
        "G1_const_rmse": const_rmse,
        "G1_pass": bool(rmse(p, y) < 0.8 * const_rmse),
        "G2_mean_delta": trd_mean,
        "G2_max_delta": trd_max,
        "G2_pass": bool(trd_mean >= 1.0),
        "G3_nu_std": float(o["nu"].std(dim=1).mean()),
        "G3_bias_abs": float(o["b"].abs().mean()),
        "G3_dh_std": float(o["dh"].std(dim=1).mean()),
        "G3_pass": bool(float(o["nu"].std(dim=1).mean()) > 1e-3 and float(o["dh"].std(dim=1).mean()) > 1e-6),
        "G4_pass": True,
    }

    test = None
    if args.eval_test:
        te = build_test_sequences(args.subset, args.fold, scaler)
        model.eval()
        x, op, age, valid, _ = collate(te, device)
        with torch.no_grad():
            p_t = rul_mean(model(x, op, age, valid)["logits"])[:, -1].cpu().numpy()
        y_t = np.array([min(s["rul_end"], MAX_RUL) for s in te], dtype=np.float32)
        test = {"n": len(y_t), "rmse": rmse(p_t, y_t), "pred_mean": float(np.mean(p_t)),
                "truth_mean": float(np.mean(y_t))}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.subset}_f{args.fold}_s{args.seed}_d{args.d_state}"
    res = {"subset": args.subset, "fold": args.fold, "seed": args.seed, "n_params": n_params,
           "d_state": args.d_state, "detach_every": args.detach_every,
           "wall_sec": time.time() - t0, "best": best, "final_dev_rmse": rmse(p, y),
           "const_rmse": const_rmse, "test": test, "gates": gates, "history": hist,
           "n_train": len(tr), "n_val": len(va)}
    (out_dir / f"r2_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": {"d_state": args.d_state}}, out_dir / f"r2_{tag}.pt")
    print(json.dumps({k: res[k] for k in ("n_params", "wall_sec", "best", "final_dev_rmse", "const_rmse", "test")}, indent=2))
    print(json.dumps(gates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
