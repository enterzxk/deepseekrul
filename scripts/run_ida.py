"""IDA-PE Gate 0 arms: M0 (PE), M1 (+concat mu/sigma), M1p (+phi(tau)),
M1_shuf (shuffled prior), M1_const (constant prior).

Runs on the SAME outer 5-fold engine split as the nested lifetime prior
(KFold(5, shuffle, seed=20261104) over sorted units), so M1-style arms train
on outer-train engines' inner-OOF priors and evaluate on outer-val engines'
ensemble priors - no prior leakage by construction (see
tests/test_prior_causality.py).

PE configuration replicates the frozen PE V0 (42,306 params): CandidateModel
num_layers=1, use_pe=True, use_ema_features=True, window 50, epochs 80,
patience 20, late_weight 1.8. Prior injection is a zero-init linear map added
to the pooled context, so at initialisation every arm is exactly M0.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    SELECTED_SENSOR_COLUMNS, OperatingConditionScaler, SensorMinMaxScaler,
    build_piecewise_linear_rul, compute_rmse, load_cmapss_train,
)
from candidates.model import CandidateModel  # noqa: E402
from lifetime_prior.nested_prior import StaircasePrior  # noqa: E402

OUTER_SEED = 20261104          # must match src/lifetime_prior/nested_prior.py
CYCLE_SCALE = 250.0
COLS = list(SELECTED_SENSOR_COLUMNS)
TAU_PERIODS = (0.125, 0.25, 0.5, 1.0)   # tau typical range ~ (0.2, 1.0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WindowSet(Dataset):
    def __init__(self, x, y, cond, extras, prior):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.cond = torch.from_numpy(cond) if cond is not None else None
        self.extras = torch.from_numpy(extras)
        self.prior = torch.from_numpy(prior)   # (n, k) per-arm prior vector

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        cond = self.cond[i] if self.cond is not None else torch.zeros(50, 3)
        return self.x[i], self.y[i], cond, self.extras[i], self.prior[i]


def build_windows_condref_meta(frame, values, window, max_rul, cluster_labels,
                               pop_ref, tau=2.0):
    """Verbatim copy of run_candidate.build_windows_condref (same maths, same
    order) plus a (unit, end_cycle) meta array for prior alignment."""
    xs, ys, extras, meta = [], [], [], []
    for _, group in frame.groupby("unit", sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        clusters = np.asarray(cluster_labels[pos], dtype=np.int64)
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        rul = build_piecewise_linear_rul(group, max_rul=max_rul)
        length = seq.shape[0]
        for end in range(window, length + 1):
            t = end - 1
            m = int(clusters[t])
            early = np.where((cycles <= 30) & (clusters == m) & (np.arange(length) <= t))[0]
            sample_idx = early[-5:] if early.size > 5 else early
            n = int(sample_idx.size)
            if n > 0:
                xbar = seq[sample_idx].mean(axis=0)
                std_ref = float(seq[sample_idx].std(axis=0).mean()) if n > 1 else 0.0
                age = float(cycles[sample_idx].mean())
            else:
                xbar = np.zeros(seq.shape[1], dtype=np.float32)
                std_ref = 0.0
                age = 30.0
            ref = (n / (n + tau)) * xbar + (tau / (n + tau)) * pop_ref[m]
            window_vals = seq[max(0, end - window): end]
            deviation = float(np.abs(window_vals - ref).mean())
            ema_state = seq[0].copy()
            for row in seq[:end]:
                ema_state = 0.5 * ema_state + 0.5 * row
            jitter = float(np.abs(seq[t] - ema_state).mean())
            evidence = [min(1.0, n / 5.0), 1.0 if n >= 1 else 0.0,
                        float(np.exp(-age / 30.0)), float(np.exp(-deviation)),
                        1.0 / (1.0 + std_ref), jitter]
            xs.append(np.ascontiguousarray(window_vals))
            ys.append(float(rul[t]))
            extras.append(np.concatenate([
                ref, [min(cycles[t] / CYCLE_SCALE, 1.0)], evidence]))
            meta.append((int(group["unit"].iloc[0]), int(cycles[t])))
    return (np.stack(xs).astype(np.float32), np.asarray(ys, np.float32),
            np.stack(extras).astype(np.float32),
            np.asarray(meta, dtype=np.int64).reshape(-1, 2))


def build_windows_meta(frame, values, window, max_rul):
    """Verbatim copy of run_candidate.build_windows (stride 1, dev mode) plus
    a (unit, end_cycle) meta array."""
    xs, ys, extras, meta = [], [], [], []
    global_baseline = np.asarray(values, dtype=np.float32)[: 5 * 20].mean(axis=0)
    for unit_id, group in frame.groupby("unit", sort=True):
        pos = group.index.to_numpy(dtype=np.int64)
        seq = np.asarray(values[pos], dtype=np.float32)
        early = seq[: min(5, seq.shape[0])]
        baseline = early.mean(axis=0) if early.shape[0] > 0 else global_baseline
        ref_std = float(early.std(axis=0).mean()) if early.shape[0] > 1 else 0.0
        cycles = group["cycle"].to_numpy(dtype=np.int64)

        def evidence(end_idx):
            w = seq[max(0, end_idx - 49): end_idx + 1]
            deviation = float(np.abs(w - baseline).mean())
            ema_state = seq[0].copy()
            for row in seq[: end_idx + 1]:
                ema_state = 0.5 * ema_state + 0.5 * row
            jitter = float(np.abs(seq[end_idx] - ema_state).mean())
            mean_age = float(np.mean(np.arange(1, min(early.shape[0], 5) + 1)))
            return [min(1.0, early.shape[0] / 5.0),
                    1.0 if early.shape[0] >= 1 else 0.0,
                    float(np.exp(-mean_age / 30.0)), float(np.exp(-deviation)),
                    1.0 / (1.0 + ref_std), jitter]
        rul = build_piecewise_linear_rul(group, max_rul=max_rul)
        for end in range(window - 1, seq.shape[0]):
            xs.append(np.ascontiguousarray(seq[end - window + 1: end + 1]))
            ys.append(float(rul[end]))
            extras.append(np.concatenate([
                baseline, [min(cycles[end] / CYCLE_SCALE, 1.0)], evidence(end)]))
            meta.append((int(unit_id), int(cycles[end])))
    return (np.stack(xs).astype(np.float32), np.asarray(ys, np.float32),
            np.stack(extras).astype(np.float32),
            np.asarray(meta, dtype=np.int64).reshape(-1, 2))


def outer_folds(subset, n_outer=5):
    """Same engine partition as nested_prior.run_subset (must stay in sync)."""
    df = load_cmapss_train(ROOT / "data" / "CMAPSS", subset)
    units = np.array(sorted(df["unit"].unique().tolist()), dtype=np.int64)
    kf = KFold(n_splits=n_outer, shuffle=True, random_state=OUTER_SEED)
    folds = []
    for ti, vi in kf.split(units):
        folds.append((set(int(u) for u in units[ti]),
                      set(int(u) for u in units[vi])))
    return folds, df


def prep_fold(df, subset, fold, tr_u, va_u, window=50, max_rul=125.0):
    """Condition-aware (FD002/FD004) or plain (FD001/FD003) window build."""
    tr_df = df[df.unit.isin(tr_u)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    va_df = df[df.unit.isin(va_u)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    if subset in ("FD002", "FD004"):
        op = ["op_1", "op_2", "op_3"]
        mean = tr_df[op].to_numpy(np.float64).mean(axis=0)
        std = tr_df[op].to_numpy(np.float64).std(axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        cond_t = ((tr_df[op].to_numpy(np.float64) - mean) / std).astype(np.float32)
        cond_v = ((va_df[op].to_numpy(np.float64) - mean) / std).astype(np.float32)
        scaler = OperatingConditionScaler(COLS, n_clusters=6, random_state=7 + fold)
        tr_s, va_s = scaler.fit_transform(tr_df), scaler.transform(va_df)
        kmeans = KMeans(n_clusters=6, random_state=7, n_init=10).fit(
            tr_df[op].to_numpy(np.float64))
        lab_tr = kmeans.labels_.astype(np.int64)
        lab_va = kmeans.predict(va_df[op].to_numpy(np.float64)).astype(np.int64)
        pop_ref = np.zeros((6, tr_s.shape[1]), dtype=np.float32)
        early_mask = tr_df["cycle"].to_numpy(np.int64) <= 15
        for m in range(6):
            sel = early_mask & (lab_tr == m)
            pop_ref[m] = tr_s[sel].mean(axis=0) if sel.any() else tr_s.mean(axis=0)
        xtr, ytr, etr, mtr = build_windows_condref_meta(tr_df, tr_s, window, max_rul, lab_tr, pop_ref)
        xva, yva, eva, mva = build_windows_condref_meta(va_df, va_s, window, max_rul, lab_va, pop_ref)
        ctr = _cond_planes(tr_df, cond_t, mtr, window)
        cva = _cond_planes(va_df, cond_v, mva, window)
    else:
        scaler = SensorMinMaxScaler(COLS)
        tr_s = scaler.fit_transform(tr_df[COLS].to_numpy(np.float64))
        va_s = scaler.transform(va_df[COLS].to_numpy(np.float64))
        xtr, ytr, etr, mtr = build_windows_meta(tr_df, tr_s, window, max_rul)
        xva, yva, eva, mva = build_windows_meta(va_df, va_s, window, max_rul)
        ctr = cva = None
    return (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva)


def _cond_planes(frame, cond_vals, meta, window):
    """(n, window, 3) operating-condition planes per window, matching the
    window list order of build_windows*_meta (unit-major, end-ascending)."""
    out = np.zeros((len(meta), window, 3), dtype=np.float32)
    units = frame["unit"].to_numpy()
    cycles = frame["cycle"].to_numpy(np.int64)
    # frame is unit-major sorted; find each unit's first row
    uniq, first_idx = np.unique(units, return_index=True)
    starts = {int(u): int(fi) for u, fi in zip(uniq, first_idx)}
    for i, (u, end_cycle) in enumerate(meta):
        s = starts[int(u)]
        e = s + int(np.searchsorted(cycles[s:], end_cycle, side="right"))
        lo = max(s, e - window)
        chunk = cond_vals[lo:e]                       # (<=window, 3)
        out[i, window - len(chunk):] = chunk
    return out


def prior_vectors(meta, sp, fold, arm, median_T, shuffle_seed=777):
    """Per-window prior feature vector for each arm.

    M1 / M1_shuf / M1_const -> [mu/250, sigma/250]
    M1p                     -> phi(tau) (4 sin/cos periods) ++ u
    """
    mu = np.empty(len(meta), np.float64)
    sg = np.empty(len(meta), np.float64)
    flag = np.empty(len(meta), np.int64)
    if arm == "M1_shuf":
        rng = np.random.default_rng(shuffle_seed + fold)
        units = np.unique(meta[:, 0])
        perm = {int(a): int(b) for a, b in zip(units, rng.permutation(units))}
    for i, (u, t) in enumerate(meta):
        if arm == "M1_shuf":
            u_q = perm[int(u)]
        else:
            u_q = int(u)
        if arm == "M1_const":
            m, s, fl, c = float(median_T), 0.0, 0, 0
        else:
            m, s, fl, c = sp.lookup(fold, u_q, int(t))
        mu[i], sg[i], flag[i] = m, s, fl
    if arm in ("M1", "M1_shuf", "M1_const"):
        return np.stack([mu / CYCLE_SCALE, sg / CYCLE_SCALE], axis=1).astype(np.float32), (mu, sg)
    # M1p: phi(tau) with tau = t / mu  (guard mu>0), plus uncertainty u
    tau = meta[:, 1].astype(np.float64) / np.maximum(mu, 1.0)
    ang = 2.0 * np.pi * tau[:, None] / np.asarray(TAU_PERIODS)[None, :]
    phi = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
    u = sg / np.maximum(mu, 1.0)
    return np.concatenate([phi, u[:, None]], axis=1).astype(np.float32), (mu, sg)


class PriorWrapper(nn.Module):
    """M0 body + zero-init prior projection added to the pooled context."""

    N_IN = {"M0": 0, "M1": 2, "M1_shuf": 2, "M1_const": 2, "M1p": 9}

    def __init__(self, body, arm, d_model=64):
        super().__init__()
        self.body = body
        self.arm = arm
        n = self.N_IN[arm]
        self.proj = nn.Linear(n, d_model) if n > 0 else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond, extras, prior):
        pooled = self.body.encode(x, cond=cond, extras=extras)
        context = pooled.mean(dim=1)
        if self.proj is not None:
            context = context + self.proj(prior)
        return {"prediction": self.body.reg_head(context)}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y, cond, extras, prior in loader:
        out = model(x.to(device), cond.to(device), extras.to(device), prior.to(device))
        preds.append(out["prediction"].reshape(-1).cpu().numpy())
        targets.append(y.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--arm", required=True,
                    choices=("M0", "M1", "M1p", "M1_shuf", "M1_const"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--prior-dir", default="experiments/ida/prior")
    ap.add_argument("--out", default="experiments/ida/gate0")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, df = outer_folds(args.subset)
    tr_u, va_u = folds[args.fold]

    # lifetime prior tables + per-fold training median
    prior_dir = ROOT / args.prior_dir
    table = pd.read_csv(prior_dir / f"{args.subset}_prior_cutoffs.csv")
    meta_p = json.loads((prior_dir / f"{args.subset}_prior_meta.json").read_text())
    median_T = float(meta_p["folds"][str(args.fold)]["median_T_train"])
    sp = StaircasePrior(table, {int(k): v for k, v in
                                ((fk, fm["median_T_train"]) for fk, fm in meta_p["folds"].items())},
                        args.subset)

    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = prep_fold(
        df, args.subset, args.fold, tr_u, va_u, window=args.window)
    ptr, (mu_tr, sg_tr) = prior_vectors(mtr, sp, args.fold, args.arm, median_T)
    pva, (mu_va, sg_va) = prior_vectors(mva, sp, args.fold, args.arm, median_T)
    train_set = WindowSet(xtr, ytr, ctr, etr, ptr)
    val_set = WindowSet(xva, yva, cva, eva, pva)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2)

    body = CandidateModel(
        num_sensors=14, window_size=args.window, d_model=64, num_layers=1,
        dropout=0.15, use_baseline=True, use_ema_features=True, use_cycle=True,
        use_pe=True, use_graph=False, use_dual_head=False)
    model = PriorWrapper(body, args.arm).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # zero-init equivalence assertion (spec: init exactly equals PE)
    if args.arm != "M0":
        model.eval()
        with torch.no_grad():
            xb, yb, cb, eb, pb = next(iter(val_loader))
            o1 = model(xb[:8].to(device), cb[:8].to(device), eb[:8].to(device), pb[:8].to(device))
            o2 = model.body.reg_head(model.body.encode(
                xb[:8].to(device), cond=cb[:8].to(device), extras=eb[:8].to(device)).mean(dim=1))
            err = float((o1["prediction"] - o2).abs().max())
        assert err < 1e-4, f"zero-init broken: max delta {err}"

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs,
                                                           eta_min=1e-5)
    LATE = 1.8
    best, best_epoch, best_state = float("inf"), -1, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, seen = 0.0, 0
        for x, y, cond, extras, prior in train_loader:
            x, y = x.to(device), y.to(device)
            cond, extras, prior = cond.to(device), extras.to(device), prior.to(device)
            optimiser.zero_grad(set_to_none=True)
            pred = model(x, cond, extras, prior)["prediction"]
            y = y.reshape(-1, 1)
            delta = pred - y
            loss = nn.functional.mse_loss(pred, y)
            late = torch.where(delta > 0, LATE * delta.square(), delta.square())
            loss = loss + 0.5 * late.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tot += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        scheduler.step()
        p, t = evaluate(model, val_loader, device)
        val = float(compute_rmse(t, p))
        if val < best - 1e-4:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            break
        print(f"[{args.subset} {args.arm} f{args.fold} s{args.seed}] ep{epoch:03d} "
              f"loss={tot/max(seen,1):.3f} val={val:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    p, t = evaluate(model, val_loader, device)

    out_dir = ROOT / args.out / f"{args.subset}" / f"{args.arm}_f{args.fold}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "unit": mva[:, 0], "end_cycle": mva[:, 1], "y_true": t, "y_pred": p,
        "mu_T": mu_va, "sigma_T": sg_va,
    }).to_csv(out_dir / "val_predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({
        "subset": args.subset, "arm": args.arm, "fold": args.fold, "seed": args.seed,
        "best_val_rmse": best, "best_epoch": best_epoch, "params": n_params,
        "n_train_windows": len(xtr), "n_val_windows": len(xva),
        "n_val_units": len(set(mva[:, 0].tolist())),
        "elapsed_seconds": time.time() - t0,
        "epochs_ran": min(args.epochs, best_epoch + args.patience),
        "outer_seed": OUTER_SEED,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }, indent=2))
    print(f"BEST {args.subset} {args.arm} f{args.fold} s{args.seed} "
          f"val={best:.4f} params={n_params}", flush=True)


if __name__ == "__main__":
    main()
