"""H004: causal-prefix width allocation with engine-stage coverage regularization.

Frozen centre (from the baseline run), trained half-width head, global CQR kept
unchanged. Features use only information observable at prediction time.

Variants:
  cqr            baseline: the existing learned width, global CQR
  eng_pinball    engine-balanced pinball only, no prefix features
  h004           full: prefix features + engine-stage coverage penalty
  noprefix       engine-stage penalty but prefix features removed
  uniform        uniformly widened CQR, matched to H004 mean width

Gate metrics per Codex's pre-registration:
  shortfall  engine-balanced mean stage shortfall  mean_e mean_k [0.9 - C_ek]+
  p10        tenth percentile of per-engine coverage
  width      mean interval width
  winkler    fixed alpha=0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ida as R  # noqa: E402

RUL_CAP = 125.0
STAGE_EDGES = (RUL_CAP / 3.0, 2.0 * RUL_CAP / 3.0)
ALPHA = 0.1
TARGET = 0.90


def stage_of(rul_pred: np.ndarray) -> np.ndarray:
    """Observable stage proxy from the frozen centre prediction."""
    s = np.zeros_like(rul_pred, dtype=int)
    s[rul_pred < STAGE_EDGES[0]] = 0
    s[(rul_pred >= STAGE_EDGES[0]) & (rul_pred < STAGE_EDGES[1])] = 1
    s[rul_pred >= STAGE_EDGES[1]] = 2
    return s


def build_features(x: np.ndarray, cutoff: np.ndarray, units: np.ndarray,
                   mu: np.ndarray, sg: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Causal features: window sensor statistics + prefix summaries of the centre."""
    n, w, c = x.shape
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    half = w // 2
    trend = x[:, half:, :].mean(axis=1) - x[:, :half, :].mean(axis=1)
    feats = [mean, std, trend, cutoff.reshape(-1, 1), mu.reshape(-1, 1),
             sg.reshape(-1, 1), centre.reshape(-1, 1)]
    # causal rolling statistics of the centre within each engine
    roll_mean = np.zeros(n); roll_std = np.zeros(n); roll_slope = np.zeros(n)
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        ser = pd.Series(centre[idx])
        roll_mean[idx] = ser.rolling(10, min_periods=1).mean().values
        roll_std[idx] = ser.rolling(10, min_periods=1).std().fillna(0.0).values
        roll_slope[idx] = (ser - ser.shift(10).fillna(method="bfill")).values
    feats += [roll_mean.reshape(-1, 1), roll_std.reshape(-1, 1), roll_slope.reshape(-1, 1)]
    return np.concatenate(feats, axis=1).astype(np.float32)


class WidthHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        return F.softplus(self.net(x)).squeeze(-1) + 1e-3


def pinball(pred, target, q):
    d = target - pred
    return torch.mean(torch.maximum(q * d, (q - 1) * d))


def group_shortfall_loss(w, err, units, stages, temp=1.0):
    """mean_e mean_k [0.9 - C_ek]+^2 with differentiable coverage."""
    total, count = 0.0, 0
    for u in np.unique(units):
        mu_ = units == u
        for k in np.unique(stages[mu_]):
            m = mu_ & (stages == k)
            cov = torch.sigmoid((w[m] - err[m]) / temp).mean()
            total = total + torch.clamp(TARGET - cov, min=0.0) ** 2
            count += 1
    return total / max(count, 1)


def match_coverage(y, centre, half, mask, target=TARGET, lo=-50.0, hi=200.0):
    """Additive q so that coverage on `mask` equals the target (bisection)."""
    def cov(q):
        h = np.maximum(0.0, half + q)
        return float(np.mean((y[mask] >= centre[mask] - h[mask]) & (y[mask] <= centre[mask] + h[mask])))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cov(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def engine_balanced(values, units):
    return float(np.mean([values[units == u].mean() for u in np.unique(units)]))


def metrics(y, lo, hi, units, centre):
    covered = (y >= lo) & (y <= hi)
    per_engine = np.array([covered[units == u].mean() for u in np.unique(units)])
    stages = stage_of(centre)
    short = []
    for u in np.unique(units):
        for k in np.unique(stages[units == u]):
            m = (units == u) & (stages == k)
            short.append(max(0.0, TARGET - covered[m].mean()))
    w = hi - lo
    wink = w + (2 / ALPHA) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    return dict(coverage=float(covered.mean()), p10=float(np.percentile(per_engine, 10)),
                shortfall=float(np.mean(short)), width=engine_balanced(w, units),
                winkler=engine_balanced(wink, units))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="docs/tables/h004")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        for fold in range(args.folds):
            d = ROOT / "experiments/punet_matrix" / args.subset / f"f{fold}_s{seed}"
            dump = d / "preds_cqr.csv"
            if not dump.exists():
                continue
            df = pd.read_csv(dump)
            units = df["unit"].values.astype(int)
            cutoff = df["cutoff"].values.astype(float)
            y = df["y_true"].values.astype(np.float32)
            centre = ((df["y_lo"] + df["y_hi"]) / 2.0).values.astype(np.float32)
            mu = df["mu_T"].values.astype(np.float32)
            sg = df["sigma_T"].values.astype(np.float32)
            base_half = ((df["y_hi"] - df["y_lo"]) / 2.0).values.astype(np.float32)

            folds, data = R.outer_folds(args.subset)
            tr_u, va_u = folds[fold]
            (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(
                data, args.subset, fold, tr_u, va_u, window=50)
            # align the raw windows with the dump by (unit, cutoff)
            key_raw = {(int(a), float(b)): i for i, (a, b) in enumerate(zip(mva[:, 0], mva[:, 1]))}
            idx = np.array([key_raw.get((int(u), float(c)), -1) for u, c in zip(units, cutoff)])
            if (idx < 0).any():
                print("align failed", args.subset, fold, int((idx < 0).sum())); continue
            Xva = build_features(xva, mva[idx, 1], units, mu, sg, centre)

            # engine-disjoint split of the validation set: fit on 60%, evaluate on 40%
            rng = np.random.default_rng(seed)
            uu = np.unique(units).copy(); rng.shuffle(uu)
            n_fit = max(2, int(round(0.6 * len(uu))))
            fit_u, ev_u = uu[:n_fit], uu[n_fit:]
            m_fit, m_ev = np.isin(units, fit_u), np.isin(units, ev_u)
            if m_fit.sum() < 50 or m_ev.sum() < 50:
                continue
            # inner split of the fit engines: train vs calibration
            rng2 = np.random.default_rng(seed + 1)
            fu = fit_u.copy(); rng2.shuffle(fu)
            h = max(1, len(fu) // 2)
            trn_u, cal_u = fu[:h], fu[h:]
            m_trn, m_cal = np.isin(units, trn_u), np.isin(units, cal_u)

            def fit_head(use_prefix: bool, use_group: bool):
                torch.manual_seed(seed)
                X = Xva.copy()
                if not use_prefix:
                    # keep only the 42 window-sensor columns; drop cutoff/prior/centre/rolling
                    X = X[:, :42]
                head = WidthHead(X.shape[1]).to(device)
                opt = torch.optim.Adam(head.parameters(), lr=1e-3)
                xt = torch.from_numpy(X[m_trn]).to(device)
                yt = torch.from_numpy(y[m_trn]).to(device)
                ct = torch.from_numpy(centre[m_trn]).to(device)
                ut = units[m_trn]; st = stage_of(centre[m_trn])
                for _ in range(args.epochs):
                    w = head(xt)
                    lo_, hi_ = ct - w, ct + w
                    loss = (pinball(lo_, yt, 0.05) + pinball(hi_, yt, 0.95)) / RUL_CAP
                    if use_group:
                        loss = loss + args.lam * group_shortfall_loss(w, torch.abs(yt - ct), ut, st)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                head.eval()
                with torch.no_grad():
                    return head(torch.from_numpy(X).to(device)).cpu().numpy()

            results = {}
            # baseline: existing width + global CQR
            for tag, half in (("cqr", base_half),):
                qq = match_coverage(y, centre, half, m_cal)
                hh = np.maximum(0.0, half + qq)
                results[tag] = metrics(y[m_ev], centre[m_ev] - hh[m_ev], centre[m_ev] + hh[m_ev],
                                       units[m_ev], centre[m_ev])
            for tag, up, ug in (("eng_pinball", False, False), ("prefix_only", True, False),
                                ("grouped_noprefix", False, True), ("h004", True, True)):
                half = fit_head(up, ug)
                qq = match_coverage(y, centre, half, m_cal)
                hh = np.maximum(0.0, half + qq)
                results[tag] = metrics(y[m_ev], centre[m_ev] - hh[m_ev], centre[m_ev] + hh[m_ev],
                                       units[m_ev], centre[m_ev])
            # uniform widening control matched to H004 mean width
            target_w = results["h004"]["width"]
            base_w = results["cqr"]["width"]
            scale = target_w / max(base_w, 1e-9)
            qq = match_coverage(y, centre, base_half, m_cal)
            hh = np.maximum(0.0, (base_half + qq) * scale)
            results["uniform"] = metrics(y[m_ev], centre[m_ev] - hh[m_ev], centre[m_ev] + hh[m_ev],
                                         units[m_ev], centre[m_ev])

            for tag, m in results.items():
                rows.append(dict(subset=args.subset, seed=seed, fold=fold, variant=tag, **m))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"h004_{args.subset}.csv", index=False)
    if len(df):
        print(df.groupby("variant")[["coverage", "shortfall", "p10", "width", "winkler"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
