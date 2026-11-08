"""Cross-domain RUL experiment on FEMTO/PRONOSTIA bearings (dump architecture).

Leave-one-bearing-out. For each held-out bearing we train exactly two models
(quantile heads and point head), dump their predictions, and derive every
calibration variant offline. This mirrors the C-MAPSS matrix so the two studies
share one protocol.

Variants derived per bearing: CQR, localized CQR, uncalibrated quantiles,
constant width, MC-dropout.
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_e2e_cqr import E2ECQR, pinball  # noqa: E402
from candidates.model import CandidateModel  # noqa: E402


def load_bearings(feat_dir: Path, window: int, cap: float | None, max_windows: int = 0,
                  label: str = "rel"):
    data = {}
    for p in sorted(feat_dir.glob("Bearing*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "snapshot" not in df.columns:
            continue
        x = df.drop(columns=["snapshot"]).to_numpy(np.float32)
        n = len(x)
        if n <= window + 2:
            continue
        k = max(5, n // 10)
        mu, sd = x[:k].mean(0), x[:k].std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        x = (x - mu) / sd
        ruls = (n - 1) - np.arange(n)
        if label == "rel":
            # fraction of life remaining: smooth, bounded, no truncation artefact
            ruls = ruls / float(n - 1)
        elif cap is not None:
            ruls = np.minimum(ruls, cap)
        xs, ys = [], []
        for end in range(window, n):
            xs.append(x[end - window:end])
            ys.append(ruls[end])
        X, Y = np.stack(xs), np.asarray(ys, np.float32)
        if max_windows and len(X) > max_windows:
            idx = np.linspace(0, len(X) - 1, max_windows).astype(int)
            X, Y = X[idx], Y[idx]
        data[p.stem] = (X, Y)
    return data


def build_model(n_feat: int, window: int, device):
    body = CandidateModel(
        num_sensors=n_feat, window_size=window, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=True, use_ema_features=False, use_cycle=False, use_pe=False,
        use_graph=False, use_dual_head=False,
    )
    return E2ECQR(body, prior_dim=2).to(device)


def train(model, sets, epochs, device, heads, seed):
    torch.manual_seed(seed)
    X = np.concatenate([s[0] for s in sets])
    Y = np.concatenate([s[1] for s in sets])
    Z = np.zeros((len(X), 2), np.float32)
    dl = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Z)),
                    batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb, pb in dl:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            lo, hi, c = model(xb, None, None, pb)
            y = yb.reshape(-1)
            loss = (F.mse_loss(c, y) if heads == "point"
                    else pinball(lo, y, 0.05) + pinball(hi, y, 0.95) + 0.1 * F.mse_loss(c, y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict(model, xs, device, mc=False, passes=30):
    model.eval()
    if mc:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    t = torch.from_numpy(xs).to(device)
    z = torch.zeros(len(xs), 2, device=device)
    if not mc:
        lo, hi, c = model(t, None, None, z)
        return lo.cpu().numpy(), hi.cpu().numpy(), c.cpu().numpy(), None
    outs = []
    for _ in range(passes):
        _, _, c = model(t, None, None, z)
        outs.append(c.cpu().numpy())
    st = np.stack(outs)
    return None, None, st.mean(0), st.std(0) + 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="data/femto_features")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=float, default=125.0)
    ap.add_argument("--only", default="", help="comma-separated bearing names")
    ap.add_argument("--label", default="rel", choices=["rel", "abs"])
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--out", default="experiments/femto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bearings(Path(args.feat_dir), args.window, args.cap, args.max_windows, args.label)
    names = sorted(data)
    if args.only:
        keep = [n.strip() for n in args.only.split(",") if n.strip()]
        names = [n for n in names if n in keep]
        data = {k: v for k, v in data.items() if k in names}
    print("bearings:", len(names), "device:", device, "label:", args.label, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for held in names:
        d = out / held
        d.mkdir(exist_ok=True)
        if (d / "preds_q.csv").exists() and (d / "preds_p.csv").exists():
            print("skip", held, flush=True)
            continue
        others = [n for n in names if n != held]
        calib, train_sets = others[0], [data[n] for n in others[1:]]
        n_feat = data[held][0].shape[-1]

        if not (d / "preds_q.csv").exists():
            m = build_model(n_feat, args.window, device)
            train(m, train_sets, args.epochs, device, "cqr", args.seed)
            lo_c, hi_c, _, _ = predict(m, data[calib][0], device)
            lo_t, hi_t, _, _ = predict(m, data[held][0], device)
            pd.DataFrame({"split": ["cal"] * len(lo_c) + ["test"] * len(lo_t),
                          "y": np.concatenate([data[calib][1], data[held][1]]),
                          "lo": np.concatenate([lo_c, lo_t]),
                          "hi": np.concatenate([hi_c, hi_t])}).to_csv(d / "preds_q.csv", index=False)
            print("quantile done", held, flush=True)

        if not (d / "preds_p.csv").exists():
            m = build_model(n_feat, args.window, device)
            train(m, train_sets, args.epochs, device, "point", args.seed)
            _, _, c_c, _ = predict(m, data[calib][0], device)
            _, _, c_t, sd_t = predict(m, data[held][0], device, mc=True)
            pd.DataFrame({"split": ["cal"] * len(c_c) + ["test"] * len(c_t),
                          "y": np.concatenate([data[calib][1], data[held][1]]),
                          "mu": np.concatenate([c_c, c_t]),
                          "sd": np.concatenate([np.ones_like(c_c), sd_t])}).to_csv(d / "preds_p.csv", index=False)
            print("point done", held, flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
