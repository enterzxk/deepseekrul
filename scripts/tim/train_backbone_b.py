"""Second backbone for the TIM study: STG-Mamba, same protocol, different architecture.

Why this script exists
----------------------
Every number in the paper is produced by one backbone (a 41 858-parameter candidate
model). A reviewer is entitled to ask whether the three conventions are properties of
the *measurement protocol* or of that particular network. This script trains a
structurally different model -- ``Lite_STG_Mamba`` from ``src/model.py``: a graph
convolution over the sensor set plus a multi-scale state-space temporal branch, with
attention pooling and no cycle/extras conditioning -- on exactly the same engine folds,
the same fold-wise scalers and the same deployment-style cutoffs as the primary
backbone. Only the network changes.

Protocol parity (nothing else is allowed to move)
-------------------------------------------------
* engine partition : ``e21.split_frames`` (SPLIT_SEED 7, 80/20 per fold), identical
* preprocessing    : ``e21.fit_scaler`` / ``e21.scale_frame``, fitted on training
                     engines only, identical
* dev rows         : ``rc.build_windows_matched`` -- one deployment cutoff U[40, .95T]
                     per held-out engine, identical cutoff seed
* test rows        : ``e21.build_test_features`` -- last observed window per test engine
* target           : ``rc.build_windows``' piecewise-linear RUL, capped at 125
* seeds/folds      : 2027/2028/2029 x 0..4, i.e. 15 members per fleet, identical

Outputs land in a parallel tree so that ``core.load_fleet`` can be pointed at them with
RUL_PRED_ROOT / RUL_PRED_TAG and every downstream script runs unchanged:

    experiments/tim/bb/cal/dev_preds/<subset>/B_s<seed>_f<fold>.csv
    experiments/tim/bb/calx/fold_test/<subset>/B_s<seed>_f<fold>.csv
    experiments/tim/bb/cal/audit/<subset>_B.npz     (y, yhat, mu, sigma, p_cens)
    experiments/tim/bb/summary.json                 (per-run val RMSE, params, timing)

The npz is rebuilt rather than copied: the parametric layer is a *method*, so the
backbone-B reference layer is refitted on backbone-B residuals with the same functional
form as ``run_matrix.fit_scale`` and the same censored binning as ``cal/layer.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tim"))

import dsh_e21_test_all as e21  # noqa: E402
import run_candidate as rc  # noqa: E402
from dataset import CMAPSS_COLUMNS  # noqa: E402
from model import Lite_STG_Mamba  # noqa: E402

WINDOW = 50
MAX_RUL = 125.0
K = 126
VERBOSE = bool(int(os.environ.get("BB_VERBOSE", "0")))


# --------------------------------------------------------------------------- pmf
def censored_pmf(mu, sigma):
    """Tobit binning on 0..125: p[k] = F(k+1) - F(k), with p[125] = 1 - F(125).

    Mirrors ``cal/layer.py:pmf_np(mode="censored")`` without importing torch there.
    """
    from scipy.special import ndtr

    mu = np.asarray(mu, float)[:, None]
    sigma = np.asarray(sigma, float)[:, None]
    edges = np.arange(K + 1, dtype=float)[None, :]
    cdf = ndtr((edges - mu) / sigma)
    p = np.diff(cdf, axis=1)
    p[:, -1] = 1.0 - cdf[:, -2]
    return np.clip(p, 1e-12, None)


def fit_scale(yhat, r):
    """sigma(yhat) = softplus(a + b*u)*20 + 1 -- the paper's parametric-layer scale."""
    from scipy import optimize

    u = np.asarray(yhat, float) / MAX_RUL

    def nll(p):
        a, b = p
        s = np.log1p(np.exp(np.clip(a + b * u, -30, 30))) * 20.0 + 1.0
        return float(np.mean(np.log(s) + 0.5 * (r / s) ** 2))

    res = optimize.minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead",
                            options=dict(xatol=1e-4, fatol=1e-8))
    a, b = res.x

    def sigma(v):
        uu = np.asarray(v, float) / MAX_RUL
        return np.log1p(np.exp(np.clip(a + b * uu, -30, 30))) * 20.0 + 1.0

    return sigma


# --------------------------------------------------------------------------- data
def load_test_frame(subset):
    traj = pd.read_csv(ROOT / "data" / "CMAPSS" / f"test_{subset}.txt", sep=r"\s+", header=None)
    assert traj.shape[1] == len(CMAPSS_COLUMNS), f"test_{subset}.txt: {traj.shape[1]} cols"
    traj.columns = list(CMAPSS_COLUMNS)
    rul = pd.read_csv(ROOT / "data" / "CMAPSS" / f"RUL_{subset}.txt", sep=r"\s+", header=None)[0]
    return traj, rul.to_numpy(float)


# --------------------------------------------------------------------------- model
class BiLSTMAttn(torch.nn.Module):
    """Canonical second backbone: input projection -> 2-layer BiLSTM -> additive attention.

    Deliberately conventional. The primary backbone is a bespoke candidate model with a
    cycle channel, a healthy baseline and reliability evidence in its input; the point of
    this arm is to ask whether the conventions survive on a network that shares none of
    that structure and that a reviewer would recognise as standard practice for C-MAPSS.
    """

    def __init__(self, n_sensors=14, hidden=48, layers=2, dropout=0.10):
        super().__init__()
        self.inp = torch.nn.Sequential(torch.nn.Linear(n_sensors, 64), torch.nn.GELU(),
                                       torch.nn.Dropout(dropout))
        self.rnn = torch.nn.LSTM(64, hidden, num_layers=layers, batch_first=True,
                                 bidirectional=True,
                                 dropout=dropout if layers > 1 else 0.0)
        self.att = torch.nn.Sequential(torch.nn.Linear(2 * hidden, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 1))
        self.head = torch.nn.Sequential(torch.nn.Linear(4 * hidden, 64), torch.nn.GELU(),
                                        torch.nn.Dropout(dropout), torch.nn.Linear(64, 1))

    def forward(self, x):  # x: (B, T, S)
        h = self.inp(x)
        out, _ = self.rnn(h)
        a = torch.softmax(self.att(out), dim=1)
        pooled = (a * out).sum(1)
        return self.head(torch.cat([out[:, -1], pooled], dim=-1))

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build(arch_seed, arch="lstm", dropout=0.10):
    torch.manual_seed(arch_seed)
    if arch == "lstm":
        return BiLSTMAttn(dropout=dropout)
    return Lite_STG_Mamba(
        num_sensors=14, window_size=WINDOW, d_model=96, graph_embed_dim=16,
        num_gcn_layers=2, state_dim=16, kernels=(3, 5, 7), dropout=dropout,
        target_scale=MAX_RUL, output_activation="linear", bidirectional_ssm=True,
        use_graph=True, use_delta=True, temporal_mode="ssm",
    )


def predict(model, x, device, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).to(device)
            out.append(model(xb).reshape(-1).float().cpu().numpy())
    return np.clip(np.concatenate(out) * MAX_RUL, 0.0, MAX_RUL)


def train_one(subset, fold, seed, epochs, patience, stride, device, log, arch="lstm"):
    raw, tr, va, val_units = e21.split_frames(subset, fold)
    scaler = e21.fit_scaler(subset, tr)
    tr_v = e21.scale_frame(subset, scaler, tr)
    va_v = e21.scale_frame(subset, scaler, va)

    x, y, _ = rc.build_windows(tr, tr_v, WINDOW, stride, MAX_RUL)
    xv, yv, _, units_v = rc.build_windows_matched(va, va_v, WINDOW, MAX_RUL, cutoff_seed=2026)

    xn = torch.from_numpy(np.ascontiguousarray(x))
    yn = torch.from_numpy((y / MAX_RUL).astype(np.float32)).reshape(-1, 1)
    xvn = torch.from_numpy(np.ascontiguousarray(xv))
    yv_t = torch.from_numpy((yv / MAX_RUL).astype(np.float32)).reshape(-1, 1)
    # Keep the whole training set on the accelerator: at 50x14x4 bytes it is 0.1 GB even
    # for FD002, and per-batch host-to-device copies dominate otherwise.
    if device.type == "cuda" and xn.numel() * 4 < 3.0e9:
        xn, yn, xvn = xn.to(device), yn.to(device), xvn.to(device)

    model = build(seed, arch).to(device)
    n_par = model.num_parameters
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    lossf = torch.nn.SmoothL1Loss(beta=0.01)

    gen = torch.Generator().manual_seed(seed)
    n = len(xn)
    best, best_state, bad = 1e9, None, 0
    bs = 256
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(xn.device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = xn[idx], yn[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if ep % 2 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                pv = torch.cat([model(xvn[i:i + bs]).reshape(-1)
                                for i in range(0, len(xvn), bs)])
            rmse = float(np.sqrt(np.mean(((pv.cpu().numpy() * MAX_RUL - yv) ** 2))))
            if VERBOSE:
                print(f"    ep {ep:3d}  val RMSE {rmse:6.2f}  best {min(best, rmse):6.2f}", flush=True)
            if rmse < best - 1e-4:
                best, bad = rmse, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- predictions: held-out dev engines (matched cutoff) and test engines
    p_dev = predict(model, xv, device)
    traj, rul = load_test_frame(subset)
    ts = e21.scale_frame(subset, scaler, traj)
    xt, _, units_t, _ = e21.build_test_features(ts, traj)
    p_tst = predict(model, xt, device)
    y_tst = np.clip(np.array([rul[u - 1] for u in units_t], float), None, MAX_RUL)

    log.append(dict(subset=subset, fold=fold, seed=seed, params=int(n_par),
                    windows=int(n), epochs_run=int(ep), val_rmse=float(best),
                    n_dev=int(len(p_dev)), n_test=int(len(p_tst))))
    return (units_v, yv, p_dev), (units_t, y_tst, p_tst)


def write_csv(path, u, y, p):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("unit_id,y_true,y_pred\n")
        for uu, yy, pp in zip(u, y, p):
            fh.write(f"{int(uu)},{float(yy):.6f},{float(pp):.6f}\n")


def read_csv_round(path):
    import csv

    us, ys, ps = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            us.append(int(r["unit_id"]))
            ys.append(float(r["y_true"]))
            ps.append(float(r["y_pred"]))
    o = np.argsort(us)
    return np.array(us)[o], np.array(ys)[o], np.array(ps)[o]


def build_audit(out):
    """Reference parametric layer for this backbone, refitted on its own residuals.

    Reads whatever (seed, fold) files are on disk, so it is idempotent and can be run
    after any subset of the training jobs has finished.
    """
    for sub_dir in sorted((out / "calx" / "fold_test").glob("FD*")):
        subset = sub_dir.name
        files = sorted(sub_dir.glob("B_s*_f*.csv"))
        if not files:
            continue
        members, y_ref = [], None
        for f in files:
            _, y, p = read_csv_round(f)
            if y_ref is None:
                y_ref = y
            members.append(p)
        mu_ref = np.mean(np.stack(members), axis=0)

        r_dev, p_dev = [], []
        for f in sorted((out / "cal" / "dev_preds" / subset).glob("B_s*_f*.csv")):
            _, y, p = read_csv_round(f)
            r_dev.append(y - p)
            p_dev.append(p)
        sig = fit_scale(np.concatenate(p_dev), np.concatenate(r_dev))
        d = out / "cal" / "audit"
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{subset}_B.npz", unit=np.arange(1, len(y_ref) + 1),
                 y=y_ref.astype(np.float32), yhat=mu_ref.astype(np.float32),
                 mu=mu_ref, sigma=sig(mu_ref),
                 p_cens=censored_pmf(mu_ref, sig(mu_ref)))
        print(f"[audit] {subset}: {len(files)} members, n={len(y_ref)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="2027,2028,2029")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "tim" / "bb"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arch", choices=("lstm", "stgmamba"), default="lstm")
    ap.add_argument("--mode", choices=("train", "audit"), default="train")
    a = ap.parse_args()

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if a.device == "auto" else a.device))
    out = Path(a.out)
    log = []
    seeds = [int(s) for s in a.seeds.split(",")]
    folds = [int(f) for f in a.folds.split(",")]

    if a.mode == "audit":
        build_audit(out)
        return

    for subset in a.subsets.split(","):
        for fold in folds:
            for seed in seeds:
                t0 = time.time()
                d, t = train_one(subset, fold, seed, a.epochs, a.patience, a.stride,
                                 device, log, a.arch)
                print(f"[{subset} f{fold} s{seed}] val RMSE {log[-1]['val_rmse']:5.2f} "
                      f"params {log[-1]['params']} ep {log[-1]['epochs_run']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                write_csv(out / "cal" / "dev_preds" / subset / f"B_s{seed}_f{fold}.csv", *d)
                write_csv(out / "calx" / "fold_test" / subset / f"B_s{seed}_f{fold}.csv", *t)

    (out / f"summary_s{seeds[0]}.json").write_text(
        json.dumps(dict(runs=log, device=str(device)), indent=1))
    print("wrote", out / f"summary_s{seeds[0]}.json")


if __name__ == "__main__":
    main()
