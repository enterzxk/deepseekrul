"""R3: compact evidence filter + per-channel bias states (specific-innovation only).

Core (proven in R1/R3 controls):
    mu_t = g(h_{t-1})                          nonlinear observation model
    nu_t = x_t - mu_t
    nu_c = nu_t - b_{t-1}
    h_t  = h_{t-1} + eta_h * tanh(W_h h + W_nu nu_c)

Bias operator (the CIDO survivor):
    nu_spec = nu_t - mean_channels(nu_t)       only the channel-specific component
    b_t     = decay * b_{t-1} + eta_b * tanh(W_b nu_spec)

The specific-only update is not cosmetic. With the full innovation the bias absorbs the
common degradation drift, starves the health state, and FD001 test RMSE degrades from
14.0 to 19.8 (measured). A sensor offset is channel-specific by definition; the shared
component belongs to health.
"""

from __future__ import annotations

import argparse
import json
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
from dataset import SELECTED_SENSOR_COLUMNS, SensorStandardScaler, load_cmapss_files  # noqa: E402

N_SENSORS = 14
MAX_RUL = 125
K_BINS = MAX_RUL
COLS = list(SELECTED_SENSOR_COLUMNS)


def make_transform(subset, tr, kind):
    if kind == "zscore":
        sc = SensorStandardScaler(COLS)
        sc.fit(tr[COLS].to_numpy(np.float64))
        return sc, (lambda f: sc.transform(f[COLS].to_numpy(np.float64)))
    sc = e21.fit_scaler(subset, tr)
    return sc, (lambda f: e21.scale_frame(subset, sc, f))


def _seqs_from(frame, scaled):
    out = []
    for unit, g in frame.groupby("unit", sort=True):
        idx = g.index.to_numpy()
        L = len(idx)
        if L < 20:
            continue
        out.append(dict(unit=int(unit), x=np.asarray(scaled[idx], dtype=np.float32),
                        age=g["cycle"].to_numpy(np.float32), L=L))
    return out


def build_sequences(subset, fold, kind, scaler_kind="zscore"):
    raw, tr, va, _ = e21.split_frames(subset, fold)
    frame = tr if kind == "train" else va
    _, tf = make_transform(subset, tr, scaler_kind)
    return _seqs_from(frame, tf(frame))


def build_test_sequences(subset, fold, scaler_kind="zscore"):
    _, tr, _, _ = e21.split_frames(subset, fold)
    _, test_frame, test_rul = load_cmapss_files(ROOT / "data" / "CMAPSS", subset)
    test_frame = test_frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    _, tf = make_transform(subset, tr, scaler_kind)
    seqs = _seqs_from(test_frame, tf(test_frame))
    for s, r in zip(seqs, test_rul):
        s["rul_end"] = float(r)
    return seqs


def collate(seqs, device, cutoffs=None, rng=None, inject_p=0.0, inject_max=4.0):
    """inject_p > 0 augments single-channel step offsets (fault-augmented training)."""
    B = len(seqs)
    lens = [s["L"] for s in seqs] if cutoffs is None else [int(c) for c in cutoffs]
    T = max(lens)
    x = np.zeros((B, T, N_SENSORS), np.float32)
    age = np.zeros((B, T), np.float32)
    valid = np.zeros((B, T), np.float32)
    rul = np.zeros((B, T), np.float32)
    for i, (s, n) in enumerate(zip(seqs, lens)):
        x[i, :n] = s["x"][:n]
        age[i, :n] = s["age"][:n]
        valid[i, :n] = 1.0
        rul[i, :n] = np.minimum(s["L"] - np.arange(1, n + 1), MAX_RUL)
        if rng is not None and inject_p > 0 and rng.random() < inject_p:
            ch = int(rng.integers(0, N_SENSORS))
            start = int(rng.integers(max(5, n // 3), max(6, n - 2)))
            x[i, start:n, ch] += float(rng.uniform(1.0, inject_max))
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(x), t(age), t(valid), t(rul)


class BiasFilter(nn.Module):
    def __init__(self, d_state=16, use_bias=True, bias_specific=True, bias_step=0.05,
                 bias_decay=0.995, state_step=0.1):
        super().__init__()
        self.d, self.use_bias = d_state, use_bias
        self.bias_specific, self.bias_step, self.bias_decay = bias_specific, bias_step, bias_decay
        self.state_step = state_step
        self.obs = nn.Sequential(nn.Linear(d_state, 48), nn.GELU(),
                                 nn.Linear(48, 48), nn.GELU(), nn.Linear(48, N_SENSORS))
        self.state = nn.Sequential(nn.Linear(d_state + N_SENSORS, 64), nn.GELU(),
                                   nn.Linear(64, d_state))
        self.bias = nn.Sequential(nn.Linear(N_SENSORS, 32), nn.GELU(), nn.Linear(32, N_SENSORS))
        self.head = nn.Sequential(nn.Linear(d_state + 1, 64), nn.GELU(), nn.Linear(64, K_BINS + 1))

    def forward(self, x, age, valid):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.d)
        b = x.new_zeros(B, N_SENSORS)
        logits, nus, dh, bs = [], [], [], []
        for t in range(T):
            mu = self.obs(h)
            nu = (x[:, t] - mu) * valid[:, t].unsqueeze(-1)
            nu_c = nu - b if self.use_bias else nu
            if self.use_bias:
                src = nu - nu.mean(dim=-1, keepdim=True) if self.bias_specific else nu
                b = self.bias_decay * b + self.bias_step * torch.tanh(self.bias(src)) * valid[:, t].unsqueeze(-1)
            delta = self.state_step * torch.tanh(self.state(torch.cat([h, nu_c], dim=-1)))
            h = h + delta * valid[:, t].unsqueeze(-1)
            logits.append(self.head(torch.cat([h, (age[:, t] / 250.0).unsqueeze(-1)], dim=-1)))
            nus.append(nu); dh.append(delta); bs.append(b)
        return dict(logits=torch.stack(logits, 1), nu=torch.stack(nus, 1),
                    dh=torch.stack(dh, 1), b=torch.stack(bs, 1))


def censored_nll(logits, rul, valid):
    logp = F.log_softmax(logits, dim=-1)
    idx = rul.clamp(0, K_BINS - 1).long()
    point = -logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    cens = -logp[..., K_BINS]
    nll = torch.where(rul >= MAX_RUL, cens, point)
    return (nll * valid).sum() / valid.sum().clamp_min(1.0)


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
def predict_seqs(model, seqs, device, cutoffs=None):
    model.eval()
    x, age, valid, _ = collate(seqs, device, cutoffs)
    return rul_mean(model(x, age, valid)["logits"])[:, -1].cpu().numpy()


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--bias", type=int, default=1)
    ap.add_argument("--bias-step", type=float, default=0.05)
    ap.add_argument("--bias-decay", type=float, default=0.995)
    ap.add_argument("--bias-specific", type=int, default=1)
    ap.add_argument("--scaler", default="zscore", choices=["zscore", "minmax"])
    ap.add_argument("--inject-p", type=float, default=0.0, help="fraction of engines augmented per epoch")
    ap.add_argument("--time-budget", type=float, default=600.0)
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "r3")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    tr = build_sequences(args.subset, args.fold, "train", args.scaler)
    va = build_sequences(args.subset, args.fold, "val", args.scaler)
    rng = np.random.default_rng(args.seed)
    cut = np.array([int(min(max(rng.integers(40, max(41, int(0.95 * s["L"])) + 1), 30), s["L"])) for s in va])
    y_dev = np.array([min(s["L"] - c, MAX_RUL) for s, c in zip(va, cut)], dtype=np.float32)
    const_rmse = float(np.std(y_dev))

    model = BiasFilter(d_state=args.d_state, use_bias=bool(args.bias),
                       bias_specific=bool(args.bias_specific), bias_step=args.bias_step,
                       bias_decay=args.bias_decay).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    xb, ab, mb, rb = collate(tr, device)
    best, hist = {"rmse": 1e9, "epoch": -1}, []
    for ep in range(args.epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        xb, ab, mb, rb = collate(tr, device, rng=(rng if args.inject_p > 0 else None),
                                 inject_p=args.inject_p) if args.inject_p > 0 else (xb, ab, mb, rb)
        out = model(xb, ab, mb)
        loss = censored_nll(out["logits"], rb, mb) + 0.05 * crps_discrete(out["logits"], rb, mb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            p = predict_seqs(model, va, device, cut)
            r = rmse(p, y_dev)
            hist.append({"epoch": ep, "loss": float(loss.item()), "dev_rmse": r})
            flag = ""
            if r < best["rmse"]:
                best = {"rmse": r, "epoch": ep}; flag = " *"
            print(f"ep{ep:03d} loss={loss.item():.4f} dev={r:.2f} (const {const_rmse:.2f}){flag}", flush=True)
            if time.time() - t0 > args.time_budget:
                print("budget", flush=True); break

    p = predict_seqs(model, va, device, cut)
    dev_rmse = rmse(p, y_dev)
    with torch.no_grad():
        o = model(xb, ab, mb)
    gates = {
        "G1_dev": dev_rmse, "G1_const": const_rmse, "G1_pass": bool(dev_rmse < 0.8 * const_rmse),
        "G3_nu_std": float(o["nu"].std(dim=1).mean()),
        "G3_dh_std": float(o["dh"].std(dim=1).mean()),
        "G3_bias_abs": float(o["b"].abs().mean()),
        "G3_pass": bool(float(o["nu"].std(dim=1).mean()) > 1e-3),
    }
    test = None
    if args.eval_test:
        te = build_test_sequences(args.subset, args.fold, args.scaler)
        pt = predict_seqs(model, te, device)
        yt = np.array([min(s["rul_end"], MAX_RUL) for s in te], dtype=np.float32)
        test = {"n": len(yt), "rmse": rmse(pt, yt), "pred_mean": float(pt.mean()), "truth_mean": float(yt.mean())}

    tag = (f"{args.subset}_f{args.fold}_s{args.seed}_d{args.d_state}_b{args.bias}"
           f"_sp{args.bias_specific}_{args.scaler}_inj{args.inject_p}")
    args.out.mkdir(parents=True, exist_ok=True)
    res = {"subset": args.subset, "fold": args.fold, "seed": args.seed, "n_params": n_params,
           "config": {"d_state": args.d_state, "bias": bool(args.bias), "bias_step": args.bias_step,
                      "bias_decay": args.bias_decay, "bias_specific": bool(args.bias_specific),
                      "scaler": args.scaler, "epochs": args.epochs, "lr": args.lr},
           "wall_sec": time.time() - t0, "best": best, "dev_rmse": dev_rmse,
           "const_rmse": const_rmse, "test": test, "gates": gates, "history": hist}
    (args.out / f"r3_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": res["config"]}, args.out / f"r3_{tag}.pt")
    print(json.dumps({k: res[k] for k in ("n_params", "wall_sec", "best", "dev_rmse", "const_rmse", "test")}, indent=2))
    print(json.dumps(gates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
