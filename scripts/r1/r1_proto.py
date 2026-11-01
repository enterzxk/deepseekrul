"""R1 prototype: compact recursive state + additive innovation update + hazard decode.

Purpose (smoke gate, not a paper model):
  G1 loss descends and beats a constant predictor by >=20% on held-out engines
  G2 the output depends on the input: time-reversal changes predictions >=1 cycle
  G3 the innovation path is alive: nu variance > 0, d(h) variance > 0, grad(W_nu) != 0
  G4 no absorbing clip anywhere on a gradient-carrying path (by construction: no clamp)

Design constraints inherited from the CET post-mortem:
  * no multiplicative gate on the evidence path (deadlock, defects 2-3)
  * observation model forecasts y_t from the state built on t-1 (defect 7)
  * distribution is decoded by a discrete hazard head, no softmax tilt (defect 4)
  * state lives in absolute cycle semantics, hazard reads age explicitly (defects 5-6)

Everything is deliberately small: 8-dim state, 14 sensors, 126 hazard outputs.
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

RUL_ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(RUL_ROOT / "src"))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS,
    SensorStandardScaler,
    load_train_only,
    make_engine_folds,
)

MAX_RUL = 125
K_BINS = MAX_RUL  # 0..124 plus a tail bin for >=125


class CompactStateRUL(nn.Module):
    """h_t = h_{t-1} + s * tanh(W_h h_{t-1} + W_nu nu_t + b), nu_t = x_t - g(h_{t-1})."""

    def __init__(self, n_sensors: int = 14, d_state: int = 8, hidden: int = 32, step: float = 0.1):
        super().__init__()
        self.d_state = d_state
        self.step = step
        self.obs = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU(), nn.Linear(hidden, n_sensors))
        self.w_h = nn.Linear(d_state, d_state, bias=False)
        self.w_nu = nn.Linear(n_sensors, d_state, bias=True)
        self.head = nn.Sequential(nn.Linear(d_state + 1, 64), nn.GELU(), nn.Linear(64, K_BINS + 1))

    def forward(self, x: torch.Tensor, age: torch.Tensor):
        """x: (B,T,14) z-scored sensors; age: (B,T) normalized cycle index."""
        b, t, _ = x.shape
        h = x.new_zeros(b, self.d_state)
        logits, nus, dh = [], [], []
        for i in range(t):
            yhat = self.obs(h)                       # forecast x_i from h_{i-1}
            nu = x[:, i] - yhat                      # innovation
            delta = self.step * torch.tanh(self.w_h(h) + self.w_nu(nu))
            h = h + delta
            z = self.head(torch.cat([h, age[:, i : i + 1]], dim=-1))
            logits.append(z)
            nus.append(nu)
            dh.append(delta)
        return torch.stack(logits, 1), torch.stack(nus, 1), torch.stack(dh, 1)


def hazard_nll(logits: torch.Tensor, rul: torch.Tensor, censored: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    idx = rul.clamp(0, K_BINS - 1).long()
    nll_point = -logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    nll_cens = -logp[..., K_BINS]
    nll = torch.where(censored, nll_cens, nll_point)
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def hazard_mean(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    bins = torch.arange(0, K_BINS + 1, device=logits.device, dtype=logits.dtype)
    return (p * bins).sum(-1)


def pad_batch(seqs: list[np.ndarray], ages: list[np.ndarray], device):
    t_max = max(s.shape[0] for s in seqs)
    b = len(seqs)
    x = np.zeros((b, t_max, seqs[0].shape[1]), dtype=np.float32)
    a = np.zeros((b, t_max), dtype=np.float32)
    m = np.zeros((b, t_max), dtype=np.float32)
    for i, (s, g) in enumerate(zip(seqs, ages)):
        n = s.shape[0]
        x[i, :n] = s
        a[i, :n] = g
        m[i, :n] = 1.0
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(a).to(device),
        torch.from_numpy(m).to(device),
    )


def engine_arrays(frame, sensors, units, window_scale=300.0):
    seqs, ages, lens = [], [], []
    for u in units:
        g = frame.loc[frame["unit"] == u]
        pos = g.index.to_numpy(dtype=np.int64)
        seq = sensors[pos]
        n = seq.shape[0]
        seqs.append(seq)
        ages.append(np.minimum(np.arange(1, n + 1) / window_scale, 1.5).astype(np.float32))
        lens.append(n)
    return seqs, ages, np.asarray(lens)


def truncate_batch(seqs, ages, lens, rng, device, min_cut=40):
    """One random prefix per engine: matches the deployment (truncated) distribution."""
    xs, ags, rs, cs = [], [], [], []
    for s, a, n in zip(seqs, ages, lens):
        hi = max(min_cut + 1, int(0.95 * n))
        c = int(rng.integers(min_cut, hi + 1))
        c = min(max(c, 30), n)
        xs.append(s[:c])
        ags.append(a[:c])
        r = n - np.arange(1, c + 1)
        rs.append(np.minimum(r, MAX_RUL).astype(np.float32))
        cs.append((r >= MAX_RUL).astype(np.float32))
    x, a, m = pad_batch(xs, ags, device)
    rb = np.zeros(m.shape, dtype=np.float32); cb = np.zeros(m.shape, dtype=bool)
    for i, (r, c) in enumerate(zip(rs, cs)):
        rb[i, : r.shape[0]] = r; cb[i, : r.shape[0]] = c
    return x, a, m, torch.from_numpy(rb).to(device), torch.from_numpy(cb).to(device)


def dev_cutoffs(lens, seed=2026):
    rng = np.random.default_rng(seed)
    out = []
    for n in lens:
        lo = 40
        hi = max(41, int(0.95 * n))
        out.append(int(min(max(rng.integers(lo, hi + 1), 30), n)))
    return np.asarray(out)


@torch.no_grad()
def evaluate(model, seqs, ages, lens, cutoffs, device):
    preds, truths = [], []
    for s, a, c, n in zip(seqs, ages, cutoffs, lens):
        x = torch.from_numpy(s[:c]).unsqueeze(0).to(device)
        g = torch.from_numpy(a[:c]).unsqueeze(0).to(device)
        logits, _, _ = model(x, g)
        preds.append(float(hazard_mean(logits[0, -1]).item()))
        truths.append(float(min(n - c, MAX_RUL)))
    return np.asarray(preds), np.asarray(truths)


@torch.no_grad()
def time_reversal_delta(model, seqs, ages, cutoffs, device):
    deltas = []
    for s, a, c in zip(seqs, ages, cutoffs):
        x = torch.from_numpy(s[:c]).unsqueeze(0).to(device)
        g = torch.from_numpy(a[:c]).unsqueeze(0).to(device)
        p_fwd = hazard_mean(model(x, g)[0][0, -1]).item()
        xr = torch.from_numpy(s[:c][::-1].copy()).unsqueeze(0).to(device)
        p_rev = hazard_mean(model(xr, g)[0][0, -1]).item()
        deltas.append(abs(p_fwd - p_rev))
    return float(np.mean(deltas)), float(np.max(deltas))


@torch.no_grad()
def sensor_blank_rmse(model, seqs, ages, lens, cutoffs, device):
    preds, truths = [], []
    for s, a, c, n in zip(seqs, ages, cutoffs, lens):
        xs = np.zeros_like(s[:c])
        x = torch.from_numpy(xs).unsqueeze(0).to(device)
        g = torch.from_numpy(a[:c]).unsqueeze(0).to(device)
        preds.append(float(hazard_mean(model(x, g)[0][0, -1]).item()))
        truths.append(float(min(n - c, MAX_RUL)))
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(truths)) ** 2)))


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--d-state", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--time-budget", type=float, default=600.0)
    ap.add_argument("--data-dir", type=Path, default=RUL_ROOT / "data" / "CMAPSS")
    ap.add_argument("--out", type=Path, default=RUL_ROOT / "experiments" / "r1")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--train-truncate", action="store_true",
                    help="sample one deployment-style cutoff per engine each epoch")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    frame = load_train_only(args.data_dir, args.subset)
    units = np.sort(frame["unit"].unique())
    folds = make_engine_folds(units, n_splits=5, seed=args.seed)
    train_units, val_units = folds[args.fold]

    tr_frame = frame.loc[frame["unit"].isin(train_units)].reset_index(drop=True)
    va_frame = frame.loc[frame["unit"].isin(val_units)].reset_index(drop=True)
    scaler = SensorStandardScaler(list(SELECTED_SENSOR_COLUMNS))
    tr_sensors = scaler.fit_transform(tr_frame.loc[:, list(SELECTED_SENSOR_COLUMNS)].to_numpy(np.float64))
    va_sensors = scaler.transform(va_frame.loc[:, list(SELECTED_SENSOR_COLUMNS)].to_numpy(np.float64))

    tr_seqs, tr_ages, tr_lens = engine_arrays(tr_frame, tr_sensors, train_units)
    va_seqs, va_ages, va_lens = engine_arrays(va_frame, va_sensors, val_units)
    cut = dev_cutoffs(va_lens, seed=args.seed)

    # per-cycle targets (uncapped lifetime -> RUL, censored at MAX_RUL)
    tr_rul, tr_cens = [], []
    for n in tr_lens:
        r = n - np.arange(1, n + 1)
        tr_rul.append(np.minimum(r, MAX_RUL).astype(np.float32))
        tr_cens.append((r >= MAX_RUL).astype(np.float32))

    xb, ab, mb = pad_batch(tr_seqs, tr_ages, device)
    rb = np.zeros(mb.shape, dtype=np.float32)
    cb = np.zeros(mb.shape, dtype=np.float32)
    for i, (r, c) in enumerate(zip(tr_rul, tr_cens)):
        rb[i, : r.shape[0]] = r
        cb[i, : r.shape[0]] = c
    rb = torch.from_numpy(rb).to(device)
    cb = torch.from_numpy(cb > 0).to(device)

    model = CompactStateRUL(d_state=args.d_state).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    const_pred = float(np.mean([min(n - c, MAX_RUL) for n, c in zip(va_lens, cut)]))
    const_rmse = rmse(np.full(len(cut), const_pred), [min(n - c, MAX_RUL) for n, c in zip(va_lens, cut)])

    hist, best = [], {"rmse": 1e9, "epoch": -1}
    grad_nu_first = grad_nu_last = None
    rng = np.random.default_rng(args.seed + 1)
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        if args.train_truncate:
            xb_e, ab_e, mb_e, rb_e, cb_e = truncate_batch(tr_seqs, tr_ages, tr_lens, rng, device)
            logits, nus, dh = model(xb_e, ab_e)
            loss = hazard_nll(logits, rb_e, cb_e, mb_e)
        else:
            logits, nus, dh = model(xb, ab)
            loss = hazard_nll(logits, rb, cb, mb)
        loss.backward()
        gnorm = float(model.w_nu.weight.grad.norm().item())
        if ep == 0:
            grad_nu_first = gnorm
        grad_nu_last = gnorm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # global norm only; no per-unit clamp
        opt.step()
        sched.step()

        if ep % 10 == 0 or ep == args.epochs - 1:
            model.eval()
            p, y = evaluate(model, va_seqs, va_ages, va_lens, cut, device)
            r = rmse(p, y)
            hist.append({"epoch": ep, "loss": float(loss.item()), "dev_rmse": r})
            if r < best["rmse"]:
                best = {"rmse": r, "epoch": ep}
            print(f"ep {ep:3d} loss {loss.item():.4f} dev_rmse {r:.2f} (const {const_rmse:.2f})", flush=True)
            if time.time() - t0 > args.time_budget:
                print("time budget reached", flush=True)
                break

    model.eval()
    p, y = evaluate(model, va_seqs, va_ages, va_lens, cut, device)
    tr_mean_delta, tr_max_delta = time_reversal_delta(model, va_seqs, va_ages, cut, device)
    blank_rmse = sensor_blank_rmse(model, va_seqs, va_ages, va_lens, cut, device)
    with torch.no_grad():
        logits, nus, dh = model(xb, ab)
    innov_std = float(nus.std(dim=1).mean().item())
    dh_std = float(dh.std(dim=1).mean().item())

    # G3: does the innovation pathway still receive gradient at the end?
    grad_ok = (grad_nu_last is not None) and (grad_nu_last > 0) and (grad_nu_first is not None) and (grad_nu_first > 0)

    gates = {
        "G1_rmse": best["rmse"],
        "G1_const_rmse": const_rmse,
        "G1_pass": bool(best["rmse"] < 0.8 * const_rmse),
        "G2_mean_delta": tr_mean_delta,
        "G2_max_delta": tr_max_delta,
        "G2_pass": bool(tr_mean_delta >= 1.0),
        "G3_innov_std": innov_std,
        "G3_dh_std": dh_std,
        "G3_grad_nu_first": grad_nu_first,
        "G3_grad_nu_last": grad_nu_last,
        "G3_pass": bool(innov_std > 1e-3 and dh_std > 1e-6 and grad_ok),
        "G4_pass": True,  # no clamp on parameters anywhere in the graph
    }
    test_info = None
    if args.eval_test:
        from dataset import load_cmapss_files

        _, te_frame, te_rul = load_cmapss_files(args.data_dir, args.subset)
        te_frame = te_frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
        te_sensors = scaler.transform(te_frame.loc[:, list(SELECTED_SENSOR_COLUMNS)].to_numpy(np.float64))
        te_units = sorted(te_frame["unit"].unique())
        tp, ty = [], []
        for u, rul_val in zip(te_units, te_rul):
            g = te_frame.loc[te_frame["unit"] == u]
            pos = g.index.to_numpy(dtype=np.int64)
            seq = te_sensors[pos]
            n = seq.shape[0]
            age = np.minimum(np.arange(1, n + 1) / 300.0, 1.5).astype(np.float32)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            a = torch.from_numpy(age).unsqueeze(0).to(device)
            with torch.no_grad():
                tp.append(float(hazard_mean(model(x, a)[0][0, -1]).item()))
            ty.append(float(min(rul_val, MAX_RUL)))
        test_info = {
            "n": len(ty),
            "rmse": rmse(tp, ty),
            "pred_mean": float(np.mean(tp)),
            "truth_mean": float(np.mean(ty)),
        }

    result = {
        "subset": args.subset,
        "fold": args.fold,
        "seed": args.seed,
        "n_params": n_params,
        "device": str(device),
        "wall_sec": time.time() - t0,
        "best": best,
        "final_dev_rmse": rmse(p, y),
        "const_rmse": const_rmse,
        "sensor_blank_rmse": blank_rmse,
        "gates": gates,
        "history": hist,
        "n_train_engines": len(train_units),
        "n_val_engines": len(val_units),
        "test": test_info,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.subset}_f{args.fold}_s{args.seed}_d{args.d_state}"
    (args.out / f"r1_{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savetxt(args.out / f"r1_{tag}_pred.csv", np.column_stack([y, p]), delimiter=",", header="truth,pred", comments="")
    print(json.dumps({k: result[k] for k in ("n_params", "wall_sec", "best", "final_dev_rmse", "const_rmse", "sensor_blank_rmse")}, indent=2))
    print(json.dumps(gates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
