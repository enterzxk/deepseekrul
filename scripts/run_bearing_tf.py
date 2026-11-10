"""Transformer baselines for bearing RUL (T1 / T2 / T3).

T1  snapshot tokens: the same per-snapshot CNN encoder as the CNN-GRU baseline,
    one token per snapshot, causal Transformer over time.
T2  patch tokens: patches of P snapshots flattened, linear projection, causal
    Transformer over patches.
T3  time-frequency tokens: per snapshot and channel, 5 band energies -> one token
    per (time, channel, band) with channel and band identity embeddings; causal
    Transformer over time (tokens of the same time step may see each other).

Protocol is identical to run_bearing2.py: strict loader, triple-disjoint LOBO,
RUL = N - i, MSE, ES-bearing early stopping, evaluation at 50/70/85/95% of life.
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
NB = 5


def bearing_dirs() -> dict[str, Path]:
    out = {}
    for base in ["Learning_set", "Full_Test_Set"]:
        d = FEMTO / base
        if not d.exists():
            continue
        for b in sorted(p.name for p in d.iterdir() if p.is_dir()):
            out[f"{base[:4]}_{b}"] = d / b
    return out


def load_bearing(path: Path, ds: int) -> np.ndarray:
    from bearing_data import load_bearing_strict, SnapshotError
    try:
        raw, _ = load_bearing_strict(path)
    except SnapshotError as e:
        raise SystemExit(f"INVALID_BEARING {path.name}: {e}")
    T = raw.shape[1] // ds
    arr = np.zeros((raw.shape[0], 2, T), dtype=np.float32)
    for c, col in enumerate((4, 5)):
        x = raw[:, :, col].astype(np.float32)
        n = x.shape[1] // ds * ds
        arr[:, c] = x[:, :n].reshape(x.shape[0], -1, ds).mean(axis=2)
    return arr


class SnapshotEncoder(nn.Module):
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


class TNet(nn.Module):
    def __init__(self, arch: str, d: int = 64, n_heads: int = 4, layers: int = 2,
                 window: int = 20, patch: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.arch, self.d, self.W, self.P = arch, d, window, patch
        if arch == "T1":
            self.enc_h, self.enc_v = SnapshotEncoder(d), SnapshotEncoder(d)
            self.n_tok_per_step = 1
        elif arch == "T2":
            self.patch_proj = nn.Linear(patch * 2 * 320, d)
            self.n_tok_per_step = 0          # handled separately
        elif arch == "T2C":
            self.enc_h, self.enc_v = SnapshotEncoder(d), SnapshotEncoder(d)
            self.patch_proj = nn.Linear(patch * 2 * d, d)
        else:
            self.band_proj = nn.Linear(1, d)
            self.ch_emb = nn.Embedding(2, d)
            self.band_emb = nn.Embedding(NB, d)
            self.n_tok_per_step = 2 * NB
        self.pos = nn.Parameter(torch.zeros(1, 256, d))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def band_energy(self, x):                # x (B, W, 2, T) -> (B, W, 2, NB)
        P = torch.log1p(torch.fft.rfft(x, dim=-1).abs() ** 2)
        f = torch.linspace(0, 1, P.shape[-1], device=x.device)
        edges = torch.linspace(0, 1, NB + 1, device=x.device)
        out = []
        for i in range(NB):
            m = (f >= edges[i]) & (f < edges[i + 1])
            out.append(P[..., m].sum(-1, keepdim=True))
        return torch.cat(out, dim=-1)        # (B, W, 2, NB)

    def forward(self, x):                    # (B, W, 2, T)
        B, W, C, T = x.shape
        if self.arch == "T1":
            hh = self.enc_h(x[:, :, 0].reshape(B * W, 1, T)).reshape(B, W, -1)
            vv = self.enc_v(x[:, :, 1].reshape(B * W, 1, T)).reshape(B, W, -1)
            z = hh + vv                       # (B, W, d)
            z = z + self.pos[:, :W]
            n_tok = W
        elif self.arch == "T2":
            P = self.P
            n = W // P
            xp = x[:, :n * P].reshape(B, n, P * 2 * T)
            z = self.patch_proj(xp) + self.pos[:, :n]
            n_tok = n
        elif self.arch == "T2C":
            P = self.P
            n = W // P
            hh = self.enc_h(x[:, :, 0].reshape(B * W, 1, T)).reshape(B, W, -1)
            vv = self.enc_v(x[:, :, 1].reshape(B * W, 1, T)).reshape(B, W, -1)
            zp = torch.cat([hh, vv], dim=-1).reshape(B, n, P * 2 * self.d)
            z = self.patch_proj(zp) + self.pos[:, :n]
            n_tok = n
        else:
            be = self.band_energy(x)          # (B, W, 2, NB)
            z = self.band_proj(be.unsqueeze(-1)).squeeze(-1)   # (B,W,2,NB,d)
            ch = self.ch_emb.weight.view(1, 1, 2, 1, self.d)
            bd = self.band_emb.weight.view(1, 1, 1, NB, self.d)
            z = z + ch + bd
            z = z.reshape(B, W * 2 * NB, self.d) + self.pos[:, :W].repeat_interleave(2 * NB, dim=1)
            n_tok = W * 2 * NB
        step = max(1, n_tok // W)
        idx = torch.arange(n_tok, device=x.device)
        tix = idx // step
        mask = tix.unsqueeze(0) > tix.unsqueeze(1)
        out = self.tf(z, mask=mask)
        if self.arch == "T3":
            out = out.reshape(B, W, 2 * NB, self.d)[:, -1].mean(dim=1)
        else:
            out = out[:, -1]
        return self.head(out).squeeze(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["T1", "T2", "T3", "T2C"])
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--es", default="")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--out", default="experiments/bearing_tf")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = bearing_dirs()
    if args.holdout not in dirs:
        print("unknown holdout"); return 1
    data = {k: load_bearing(v, args.ds) for k, v in dirs.items()}
    pool = [n for n in data if n != args.holdout]
    es_name = args.es or sorted(pool, key=lambda n: len(data[n]))[len(pool) // 2]
    tr_names = [n for n in pool if n != es_name]

    mu = np.concatenate([data[n] for n in tr_names]).mean(axis=(0, 2), keepdims=True)
    sd = np.concatenate([data[n] for n in tr_names]).std(axis=(0, 2), keepdims=True) + 1e-6

    def build(names, last_only):
        X, Y = [], []
        for n in names:
            a = (data[n] - mu) / sd
            rul = np.arange(len(a) - 1, -1, -1, dtype=np.float32)
            for e in range(args.window, len(a) + 1, args.stride):
                if last_only and e - 1 < int(len(a) * (1 - args.eval_frac)):
                    continue
                X.append(a[e - args.window:e]); Y.append(rul[e - 1])
        return np.stack(X), np.asarray(Y, np.float32)

    xtr, ytr = build(tr_names, False)
    xes, yes = build([es_name], True)
    scale = float(ytr.max()) + 1e-6

    model = TNet(args.arch, window=args.window, patch=args.patch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
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
        rmse = float(np.sqrt(np.mean((p - yes) ** 2)))
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
        for e in [min(n - 1, max(args.window - 1, int(round(fr * n)) - 1)) for fr in STAGES]:
            lo = max(0, e - args.window + 1)
            w = a[lo:e + 1]
            if len(w) < args.window:
                w = np.concatenate([np.repeat(w[:1], args.window - len(w), axis=0), w])
            pred = float(model(torch.from_numpy(w[None]).to(device)).cpu().numpy()[0] * scale)
            rows.append(dict(holdout=args.holdout, end=e, true_rul=float(rul[e]),
                             pred_rul=pred, err=pred - float(rul[e])))
    res = pd.DataFrame(rows)
    out = ROOT / args.out / args.arch
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{args.holdout}.csv", index=False)
    (out / f"{args.holdout}.json").write_text(json.dumps(dict(
        arch=args.arch, holdout=args.holdout, es_bearing=es_name, best_es_rmse=best,
        best_epoch=best_ep, n_params=int(sum(p.numel() for p in model.parameters())),
        window=args.window, patch=args.patch), indent=1), encoding="utf-8")
    print(json.dumps(dict(arch=args.arch, holdout=args.holdout, es=es_name,
                          best_es=round(best, 1), ep=best_ep,
                          stage_rmse=float(np.sqrt(np.mean(res.err ** 2)))), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
