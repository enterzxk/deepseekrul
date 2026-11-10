"""Bearing RUL, corrected protocol (v2).

Fixes vs v1
  * early stopping uses a HELD-OUT TRAINING bearing (ES bearing); the evaluation
    bearing is never touched during training or model selection;
  * attribution arms separate "dual encoder" from "learned gate":
        A        shared CNN over both channels (single encoder)
        C        dual encoder + learned multiplicative gate on h_t
        C-Fixed  dual encoder + fixed 0.5 fusion (no gate)
        C-MLP    dual encoder + equal-capacity MLP fusion (no multiplicative gate)
  * evaluation reports the last 15% AND fixed life stages (50/70/85/95% of life);
    life fractions are used only to LOCATE evaluation points, never as input.

Input: raw two-channel snapshots, box-downsampled by --ds (default 8).
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

ROOT = Path(__file__).resolve().parents[1]
FEMTO = Path("/home/yndx/femto")
N_RAW = 2560
STAGES = (0.50, 0.70, 0.85, 0.95)


def bearing_dirs() -> list[tuple[str, Path]]:
    out = []
    for base in ["Learning_set", "Full_Test_Set"]:
        d = FEMTO / base
        if not d.exists():
            continue
        for b in sorted(p.name for p in d.iterdir() if p.is_dir()):
            out.append((f"{base[:4]}_{b}", d / b))
    return out


def load_bearing(path: Path, ds: int) -> np.ndarray:
    """Strict load: any invalid snapshot aborts the run (no zero fill)."""
    from bearing_data import load_bearing_strict, SnapshotError
    try:
        raw, rep = load_bearing_strict(path)
    except SnapshotError as e:
        raise SystemExit(f"INVALID_BEARING {path.name}: {e}")
    T = raw.shape[1] // ds
    arr = np.zeros((raw.shape[0], 2, T), dtype=np.float32)
    for c, col in enumerate((4, 5)):
        x = raw[:, :, col].astype(np.float32)
        n = x.shape[1] // ds * ds
        arr[:, c] = x[:, :n].reshape(x.shape[0], -1, ds).mean(axis=2)
    return arr


class Encoder(nn.Module):
    def __init__(self, d: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(32, d, 9, stride=2, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Net(nn.Module):
    def __init__(self, arm: str, d: int = 64) -> None:
        super().__init__()
        self.arm = arm
        self.enc = Encoder(d) if arm == "A" else None
        if arm != "A":
            self.enc_h, self.enc_v = Encoder(d), Encoder(d)
        self.fuse = nn.Linear(2 * d, d) if arm in ("A", "C", "C-Fixed", "C-NoGate") else nn.Sequential(
            nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, 1)
        self.gate = nn.Linear(d, 1)

    def forward(self, x):                       # (B, W, 2, T)
        B, W, C, T = x.shape
        if self.arm == "A":
            hh = self.enc(x[:, :, 0].reshape(B * W, 1, T)).reshape(B, W, -1)
            vv = self.enc(x[:, :, 1].reshape(B * W, 1, T)).reshape(B, W, -1)
        else:
            hh = self.enc_h(x[:, :, 0].reshape(B * W, 1, T)).reshape(B, W, -1)
            vv = self.enc_v(x[:, :, 1].reshape(B * W, 1, T)).reshape(B, W, -1)
        z = self.fuse(torch.cat([hh, vv], dim=-1))
        out, _ = self.gru(z)
        if self.arm == "C":
            g = torch.sigmoid(self.gate(out))
            out = out * g
        elif self.arm == "C-Fixed":
            out = out * 0.5
        return self.head(out).squeeze(-1)[:, -1]


def windows(arr, rul, w, stride):
    xs, ys = [], []
    for e in range(w, len(arr) + 1, stride):
        xs.append(arr[e - w:e]); ys.append(rul[e - 1])
    return np.stack(xs), np.asarray(ys, np.float32)


def stage_points(n: int, w: int) -> list[int]:
    """Window-end indices at fixed life fractions (used only to locate points)."""
    return [min(n - 1, max(w - 1, int(round(f * n)) - 1)) for f in STAGES]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "C", "C-Fixed", "C-MLP"])
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--es", default="")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--out", default="experiments/bearing_tim2")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = dict(bearing_dirs())
    if args.holdout not in dirs:
        print("unknown holdout", args.holdout); return 1
    data = {k: load_bearing(v, args.ds) for k, v in dirs.items()}

    pool = [n for n in data if n != args.holdout]
    es_name = args.es or sorted(pool, key=lambda n: len(data[n]))[len(pool) // 2]
    tr_names = [n for n in pool if n != es_name]

    mu = np.concatenate([data[n] for n in tr_names]).mean(axis=(0, 2), keepdims=True)
    sd = np.concatenate([data[n] for n in tr_names]).std(axis=(0, 2), keepdims=True) + 1e-6

    def build(names, mode):
        X, Y = [], []
        for n in names:
            a = (data[n] - mu) / sd
            rul = np.arange(len(a) - 1, -1, -1, dtype=np.float32)
            x, y = windows(a, rul, args.window, args.stride)
            if mode == "last":
                keep = np.arange(len(a) - args.window, -1, -args.stride)[::-1]
                keep = np.asarray([e for e in range(args.window, len(a) + 1, args.stride)]) - 1
                sel = keep >= int(len(a) * (1 - args.eval_frac))
                x, y = x[sel], y[sel]
            X.append(x); Y.append(y)
        return np.concatenate(X), np.concatenate(Y)

    xtr, ytr = build(tr_names, "all")
    xes, yes = build([es_name], "last")
    scale = float(ytr.max()) + 1e-6

    model = Net(args.arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr = torch.from_numpy(xtr), torch.from_numpy(ytr / scale)
    Xes = torch.from_numpy(xes).to(device)
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            loss = F.mse_loss(model(Xtr[idx].to(device)), Ytr[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(Xes).cpu().numpy() * scale
        rmse = float(np.sqrt(np.mean((p - yes) ** 2)))      # ES bearing only
        if rmse < best - 1e-3:
            best, best_ep = rmse, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    a = (data[args.holdout] - mu) / sd
    n = len(a)
    rul = np.arange(n - 1, -1, -1, dtype=np.float32)
    rows = []
    with torch.no_grad():
        for e in stage_points(n, args.window):
            lo = max(0, e - args.window + 1)
            w = a[lo:e + 1]
            if len(w) < args.window:
                w = np.concatenate([np.repeat(w[:1], args.window - len(w), axis=0), w])
            xb = torch.from_numpy(w[None]).to(device)
            pred = float(model(xb).cpu().numpy()[0] * scale)
            rows.append(dict(holdout=args.holdout, arm=args.arm, end=e, true_rul=float(rul[e]),
                             pred_rul=pred, err=pred - float(rul[e])))
    res = pd.DataFrame(rows)
    out = ROOT / args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{args.holdout}.csv", index=False)
    (out / f"{args.holdout}.json").write_text(json.dumps(dict(
        arm=args.arm, holdout=args.holdout, es_bearing=es_name, best_es_rmse=best, best_epoch=best_ep,
        n_train=int(len(xtr)), ds=args.ds), indent=1), encoding="utf-8")
    print(json.dumps(dict(arm=args.arm, holdout=args.holdout, es=es_name, best_es=round(best, 1),
                          ep=best_ep, stage_rmse=float(np.sqrt(np.mean(res.err ** 2)))), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
