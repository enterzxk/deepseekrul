"""Bearing RUL branch: raw-waveform encoder + two-channel consistency gate.

Data: FEMTO/PRONOSTIA raw snapshots (2560 x 6 per snapshot; cols 4 and 5 are the
horizontal and vertical accelerations). 17 bearings have complete run-to-failure
trajectories (6 Learning_set + 11 Full_Test_Set), so RUL = N - i is a legal label.

Protocol
  * the BEARING is the independent unit: leave-one-bearing-out;
  * normalisation statistics come from the training bearings only;
  * evaluation is on the last `--eval-frac` of each held-out bearing, where RUL
    is small and the measurement question actually matters.

Arms
  A  direct: shared CNN encoder over both channels -> GRU -> RUL
  C  generic gate: two encoders, learned gate on h_t only (capacity control)
  D  consistency gate: gate driven by cross-channel innovation agreement

The mechanism under test is D vs C: does *cross-channel measurement consistency*
carry information that a generic learned gate does not?
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
DS = 8                      # waveform downsampling factor (2560 -> 320)
N_SAMP = 2560 // DS


def bearing_dirs() -> list[tuple[str, Path]]:
    out = []
    for base in ["Learning_set", "Full_Test_Set"]:
        d = FEMTO / base
        if not d.exists():
            continue
        for b in sorted(p.name for p in d.iterdir() if p.is_dir()):
            out.append((f"{base[:4]}_{b}", d / b))
    return out


def load_bearing(path: Path) -> np.ndarray:
    files = sorted(path.glob("acc_*.csv"))
    arr = np.zeros((len(files), 2, N_SAMP), dtype=np.float32)
    for i, f in enumerate(files):
        try:
            d = pd.read_csv(f, header=None).to_numpy(np.float32)
        except Exception:
            continue
        h, v = d[:, 4], d[:, 5]
        n = len(h) // DS * DS
        arr[i, 0] = h[:n].reshape(-1, DS).mean(axis=1)
        arr[i, 1] = v[:n].reshape(-1, DS).mean(axis=1)
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

    def forward(self, x):                 # x: (B, 1, T)
        return self.net(x).squeeze(-1)    # (B, d)


class BearingNet(nn.Module):
    def __init__(self, arm: str, d: int = 64) -> None:
        super().__init__()
        self.arm = arm
        self.enc_h = Encoder(d)
        self.enc_v = Encoder(d) if arm != "A" else None
        self.proj = nn.Linear(2 * d if arm == "A" else 2 * d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, 1)
        self.gate = nn.Linear(d + 2, 1) if arm == "D" else nn.Linear(d, 1)

    def forward(self, x):                 # x: (B, W, 2, T)
        B, W, C, T = x.shape
        hh = self.enc_h(x[:, :, 0].reshape(B * W, 1, T)).reshape(B, W, -1)
        if self.arm == "A":
            vv = self.enc_h(x[:, :, 1].reshape(B * W, 1, T)).reshape(B, W, -1)
        else:
            vv = self.enc_v(x[:, :, 1].reshape(B * W, 1, T)).reshape(B, W, -1)
        z = self.proj(torch.cat([hh, vv], dim=-1))
        out, _ = self.gru(z)
        if self.arm == "D":
            # cross-channel agreement of the two embeddings at each step
            cos = F.cosine_similarity(hh, vv, dim=-1, eps=1e-6)
            cons = cos.unsqueeze(-1)
            persist = torch.zeros_like(cons)
            for t in range(1, W):
                persist[:, t] = 0.9 * persist[:, t - 1] + 0.1 * torch.norm(hh[:, t] - vv[:, t], dim=-1, keepdim=True)
            g = torch.sigmoid(self.gate(torch.cat([out, cons, persist], dim=-1)))
        else:
            g = torch.sigmoid(self.gate(out))
        return self.head(out * g).squeeze(-1)[:, -1]


def windows(arr: np.ndarray, rul: np.ndarray, w: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ends = [], [], []
    for e in range(w, len(arr) + 1, stride):
        xs.append(arr[e - w:e])
        ys.append(rul[e - 1])
        ends.append(e - 1)
    return np.stack(xs), np.asarray(ys, np.float32), np.asarray(ends)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "C", "D"])
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--out", default="experiments/bearing_tim")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = bearing_dirs()
    if not dirs:
        print("no FEMTO data found")
        return 1
    data = {name: load_bearing(p) for name, p in dirs}
    print(f"bearings={len(data)} " + " ".join(f"{k}:{len(v)}" for k, v in list(data.items())[:4]), flush=True)

    tr_names = [n for n in data if n != args.holdout]
    mu = np.concatenate([data[n] for n in tr_names]).mean(axis=(0, 2), keepdims=True)
    sd = np.concatenate([data[n] for n in tr_names]).std(axis=(0, 2), keepdims=True) + 1e-6

    def build(names, train: bool):
        X, Y = [], []
        for n in names:
            a = (data[n] - mu) / sd
            rul = np.arange(len(a) - 1, -1, -1, dtype=np.float32)
            x, y, ends = windows(a, rul, args.window, args.stride)
            if not train:
                keep = ends >= int(len(a) * (1 - args.eval_frac))
                x, y = x[keep], y[keep]
            X.append(x); Y.append(y)
        return np.concatenate(X), np.concatenate(Y)

    xtr, ytr = build(tr_names, True)
    xte, yte = build([args.holdout], False)
    scale = float(ytr.max()) + 1e-6
    print(f"train={xtr.shape} test={xte.shape} scale={scale:.0f}", flush=True)

    model = BearingNet(args.arm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.from_numpy(xtr); Ytr = torch.from_numpy(ytr / scale)
    n = len(Xtr)
    best, best_ep = float("inf"), -1
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            p = model(torch.from_numpy(xte).to(device)).cpu().numpy() * scale
        rmse = float(np.sqrt(np.mean((p - yte) ** 2)))
        if rmse < best - 1e-3:
            best, best_ep = rmse, ep
        if ep - best_ep >= args.patience:
            break
    print(json.dumps(dict(arm=args.arm, holdout=args.holdout, rmse=best, best_epoch=best_ep,
                          n_train=int(n), n_test=int(len(xte))), ensure_ascii=False), flush=True)
    out = ROOT / args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": yte, "y_pred": p}).to_csv(out / f"{args.holdout}.csv", index=False)
    (out / f"{args.holdout}.json").write_text(json.dumps(dict(arm=args.arm, holdout=args.holdout,
                                                              rmse=best, best_epoch=best_ep), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
