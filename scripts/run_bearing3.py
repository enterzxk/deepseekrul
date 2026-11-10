"""E0-E3: does high-frequency information add RUL value beyond RMS?

Same backbone / protocol as run_bearing2.py (strict loader, triple-disjoint LOBO:
holdout bearing for evaluation, ES bearing for early stopping, rest for training),
plus an optional per-snapshot feature vector concatenated to the GRU input.

  --feat none   E0  ds=8 waveform only
  --feat rms    E1  + h_rms, v_rms
  --feat hf     E2  + absolute 8-12.8 kHz band energy (h, v)
  --feat bands  E3  + relative band fractions (5 bands x 2 channels) + h_rms, v_rms

All features are computed from the CURRENT raw snapshot only (causal) and are
standardised with statistics fitted on the TRAINING bearings only.
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
SPEC = ROOT / "data/femto_spec"
N_RAW = 2560
STAGES = (0.50, 0.70, 0.85, 0.95)

FEAT_COLS = {
    "none": [],
    "rms": ["h_rms", "v_rms"],
    "hf": ["h_band4_abs", "v_band4_abs"],
    "bands": ["h_band0", "h_band1", "h_band2", "h_band3", "h_band4",
              "v_band0", "v_band1", "v_band2", "v_band3", "v_band4", "h_rms", "v_rms"],
}


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


def load_features(name: str, n: int) -> np.ndarray:
    f = SPEC / f"{name}.csv"
    d = pd.read_csv(f)
    d["h_band4_abs"] = d["h_band4"] * d["h_rms"] ** 2
    d["v_band4_abs"] = d["v_band4"] * d["v_rms"] ** 2
    cols = FEAT_COLS[CFG]
    if not cols:
        return np.zeros((n, 0), np.float32)
    M = d[cols].to_numpy(np.float32)
    if len(M) < n:
        M = np.concatenate([M, np.repeat(M[-1:], n - len(M), axis=0)])
    return M[:n]


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
    def __init__(self, d: int = 64, n_feat: int = 0) -> None:
        super().__init__()
        self.enc_h, self.enc_v = Encoder(d), Encoder(d)
        self.fuse = nn.Linear(2 * d, d)
        self.gru = nn.GRU(d + n_feat, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, x, f):                    # x (B,W,2,T), f (B,W,F)
        B, W, C, T = x.shape
        hh = self.enc_h(x[:, :, 0].reshape(B * W, 1, T)).reshape(B, W, -1)
        vv = self.enc_v(x[:, :, 1].reshape(B * W, 1, T)).reshape(B, W, -1)
        z = self.fuse(torch.cat([hh, vv], dim=-1))
        if f.shape[-1] > 0:
            z = torch.cat([z, f], dim=-1)
        out, _ = self.gru(z)
        return self.head(out).squeeze(-1)[:, -1]


def main(argv=None) -> int:
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, choices=list(FEAT_COLS))
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--es", default="")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--ds", type=int, default=8)
    ap.add_argument("--out", default="experiments/bearing_tim4")
    args = ap.parse_args(argv)
    CFG = args.feat

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = bearing_dirs()
    if args.holdout not in dirs:
        print("unknown holdout"); return 1
    data = {k: load_bearing(v, args.ds) for k, v in dirs.items()}
    feats = {k: load_features(k, len(v)) for k, v in data.items()}

    pool = [n for n in data if n != args.holdout]
    es_name = args.es or sorted(pool, key=lambda n: len(data[n]))[len(pool) // 2]
    tr_names = [n for n in pool if n != es_name]

    mu = np.concatenate([data[n] for n in tr_names]).mean(axis=(0, 2), keepdims=True)
    sd = np.concatenate([data[n] for n in tr_names]).std(axis=(0, 2), keepdims=True) + 1e-6
    n_feat = feats[tr_names[0]].shape[1]
    if n_feat:
        fm = np.concatenate([feats[n] for n in tr_names]).mean(axis=0)
        fs = np.concatenate([feats[n] for n in tr_names]).std(axis=0) + 1e-6
    else:
        fm = fs = None

    def build(names, last_only):
        X, Y, FF = [], [], []
        for n in names:
            a = (data[n] - mu) / sd
            f = ((feats[n] - fm) / fs).astype(np.float32) if n_feat else np.zeros((len(a), 0), np.float32)
            rul = np.arange(len(a) - 1, -1, -1, dtype=np.float32)
            for e in range(args.window, len(a) + 1, args.stride):
                if last_only and e - 1 < int(len(a) * (1 - args.eval_frac)):
                    continue
                X.append(a[e - args.window:e]); Y.append(rul[e - 1]); FF.append(f[e - args.window:e])
        return (np.stack(X), np.asarray(Y, np.float32),
                np.stack(FF) if FF and FF[0].shape[-1] else np.zeros((len(X), args.window, 0), np.float32))

    xtr, ytr, ftr = build(tr_names, False)
    xes, yes, fes = build([es_name], True)
    scale = float(ytr.max()) + 1e-6

    model = Net(n_feat=n_feat).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Ftr = torch.from_numpy(xtr), torch.from_numpy(ytr / scale), torch.from_numpy(ftr)
    Xes, Fes = torch.from_numpy(xes).to(device), torch.from_numpy(fes).to(device)
    best, best_ep, best_state = float("inf"), -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            loss = F.mse_loss(model(Xtr[idx].to(device), Ftr[idx].to(device)), Ytr[idx].to(device))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(Xes, Fes).cpu().numpy() * scale
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
    f = ((feats[args.holdout] - fm) / fs).astype(np.float32) if n_feat else np.zeros((len(a), 0), np.float32)
    n = len(a)
    rul = np.arange(n - 1, -1, -1, dtype=np.float32)
    rows = []
    with torch.no_grad():
        for e in [min(n - 1, max(args.window - 1, int(round(fr * n)) - 1)) for fr in STAGES]:
            lo = max(0, e - args.window + 1)
            w = a[lo:e + 1]; wf = f[lo:e + 1]
            if len(w) < args.window:
                w = np.concatenate([np.repeat(w[:1], args.window - len(w), axis=0), w])
                wf = np.concatenate([np.repeat(wf[:1], args.window - len(wf), axis=0), wf])
            pred = float(model(torch.from_numpy(w[None]).to(device),
                               torch.from_numpy(wf[None]).to(device)).cpu().numpy()[0] * scale)
            rows.append(dict(holdout=args.holdout, end=e, true_rul=float(rul[e]),
                             pred_rul=pred, err=pred - float(rul[e])))
    res = pd.DataFrame(rows)
    out = ROOT / args.out / args.feat
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"{args.holdout}.csv", index=False)
    (out / f"{args.holdout}.json").write_text(json.dumps(dict(
        feat=args.feat, holdout=args.holdout, es_bearing=es_name, best_es_rmse=best,
        best_epoch=best_ep, n_train=int(len(xtr)), n_feat=int(n_feat), ds=args.ds), indent=1), encoding="utf-8")
    print(json.dumps(dict(feat=args.feat, holdout=args.holdout, es=es_name,
                          best_es=round(best, 1), ep=best_ep,
                          stage_rmse=float(np.sqrt(np.mean(res.err ** 2)))), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
