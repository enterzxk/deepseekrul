"""Cross-domain RUL experiment on FEMTO/PRONOSTIA bearings.

Protocol mirrors the C-MAPSS study so the two datasets are directly comparable:
leave-one-bearing-out. For each held-out bearing, one of the remaining bearings
is reserved for conformal calibration and the rest are used for training. The
same E2ECQR model (quantile heads + CQR) is used, with the vibration features
replacing the 14 C-MAPSS sensors.

Reported per held-out bearing: PICP, Winkler, and the paired comparison against
the constant-width and uncalibrated-quantile baselines.
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
from src.punet.metrics import picp, pinaw, winkler  # noqa: E402


def load_bearings(feat_dir: Path, window: int, cap: float | None):
    data = {}
    for p in sorted(feat_dir.glob("Bearing*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            print("skip unreadable", p.name, flush=True)
            continue
        if df.empty or "snapshot" not in df.columns:
            print("skip empty", p.name, flush=True)
            continue
        x = df.drop(columns=["snapshot"]).to_numpy(np.float32)
        n = len(x)
        if n <= window + 2:
            continue
        # z-score per bearing using its own first 10% (unsupervised, causal)
        k = max(5, n // 10)
        mu, sd = x[:k].mean(0), x[:k].std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        x = (x - mu) / sd
        ruls = (n - 1) - np.arange(n)
        if cap is not None:
            ruls = np.minimum(ruls, cap)
        xs, ys = [], []
        for end in range(window, n):
            xs.append(x[end - window:end])
            ys.append(ruls[end])
        data[p.stem] = (np.stack(xs), np.asarray(ys, np.float32))
    return data


def train_eval(train_sets, calib_set, test_set, window, device, epochs, seed, heads):
    torch.manual_seed(seed)
    n_feat = train_sets[0][0].shape[-1]
    body = CandidateModel(
        num_sensors=n_feat, window_size=window, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=True, use_ema_features=False, use_cycle=False, use_pe=False,
        use_graph=False, use_dual_head=False,
    )
    model = E2ECQR(body, prior_dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xtr = np.concatenate([s[0] for s in train_sets])
    Ytr = np.concatenate([s[1] for s in train_sets])
    zeros = np.zeros((len(Xtr), 2), np.float32)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr), torch.from_numpy(zeros))
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb, pb in dl:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            lo, hi, centre = model(xb, None, None, pb)
            y2 = yb.reshape(-1)
            if heads == "point":
                loss = F.mse_loss(centre, y2)
            else:
                loss = pinball(lo, y2, 0.05) + pinball(hi, y2, 0.95) + 0.1 * F.mse_loss(centre, y2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    def infer(s):
        xs, ys = s
        with torch.no_grad():
            lo, hi, c = model(torch.from_numpy(xs).to(device), None, None,
                              torch.zeros(len(xs), 2, device=device))
        return lo.cpu().numpy(), hi.cpu().numpy(), c.cpu().numpy(), ys

    lo_c, hi_c, _, y_c = infer(calib_set)
    lo_t, hi_t, c_t, y_t = infer(test_set)
    return lo_c, hi_c, y_c, lo_t, hi_t, c_t, y_t


def calibrate_cqr(y_c, lo_c, hi_c, lo_t, hi_t, delta):
    s = np.maximum(lo_c - y_c, y_c - hi_c)
    q = float(np.quantile(s, min(1.0, (1 - delta) * (1 + 1 / s.size)), method="higher"))
    return lo_t - q, hi_t + q


def calibrate_scaled(y_c, c_c, y_t, c_t, delta, w=None):
    w = 1.0 if w is None else w
    s = np.abs(y_c - c_c) / w
    q = float(np.quantile(s, min(1.0, (1 - delta) * (1 + 1 / s.size)), method="higher"))
    return c_t - q * w, c_t + q * w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="data/femto_features")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=float, default=125.0)
    ap.add_argument("--out", default="experiments/femto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_bearings(Path(args.feat_dir), args.window, args.cap)
    names = sorted(data)
    print("bearings:", names, flush=True)

    rows = []
    for i, held in enumerate(names):
        others = [n for n in names if n != held]
        calib = others[0]
        train = others[1:]
        res = {}
        for heads, tag in (("cqr", "cqr"), ("point", "const"), ("qr_raw", "qr")):
            lo_c, hi_c, y_c, lo_t, hi_t, c_t, y_t = train_eval(
                [data[n] for n in train], data[calib], data[held],
                args.window, device, args.epochs, args.seed, heads)
            if tag == "cqr":
                lo, hi = calibrate_cqr(y_c, lo_c, hi_c, lo_t, hi_t, args.delta)
            elif tag == "const":
                lo, hi = calibrate_scaled(y_c, (lo_c + hi_c) / 2, y_t, c_t, args.delta)
            else:
                lo, hi = lo_t, hi_t
            res[tag] = (float(picp(y_t, lo, hi)), float(winkler(y_t, lo, hi, args.delta)))
        rows.append(dict(bearing=held, **{f"{k}_picp": v[0] for k, v in res.items()},
                         **{f"{k}_winkler": v[1] for k, v in res.items()}))
        print(held, {k: (round(v[0], 3), round(v[1], 1)) for k, v in res.items()}, flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "femto_results.csv", index=False)
    print(df.describe().T[["mean", "std"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
