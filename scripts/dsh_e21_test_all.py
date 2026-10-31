"""E21 — unified official-test evaluation for FD001-FD004 (arms A, BE, PE).

Protocol frozen in paper/E21_TEST_ALL_PREREGISTRATION.md (sha 1579e0e0):
  one prediction per engine (last observable 50-cycle window), the fold's training
  scaler applied unchanged to the test trajectory, extras built exactly as WindowSet
  builds them, prediction clipped to [0, 125], truth = official RUL file.

Two modes:
  --mode verify  reconstruct each fold's *validation* features from the training scaler and
                 compare them with run_candidate.prepare()'s matched_set. Must pass on every
                 subset before any test file is read.
  --mode test    the evaluation itself.

Outputs: results/dsh/e21/test_all.json, results/dsh/e21/exclusions.json,
         results/dsh/e21/ensemble_predictions_<subset>.csv
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = "/home/yndx/zxk/bearing-rul-pi"
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")

import run_candidate as rc  # noqa: E402
from dataset import (load_cmapss_train, OperatingConditionScaler, SensorMinMaxScaler,  # noqa: E402
                     SELECTED_SENSOR_COLUMNS, CMAPSS_COLUMNS)  # noqa: E402
from candidates.model import CandidateModel  # noqa: E402

WINDOW, MAX_RUL, SPLIT_SEED, N_COND = 50, 125.0, 7, 6
ARMS = ("A", "BE", "PE")
SEEDS = (2027, 2028, 2029)
FOLDS = range(5)
RNG = 9000

CONFIG = {
    "FD001": dict(cond_aware=False, fold0="e3_fd001", later="e18"),
    "FD002": dict(cond_aware=True, fold0="e3", later="e10"),
    "FD003": dict(cond_aware=False, fold0="e20", later="e20"),
    "FD004": dict(cond_aware=True, fold0="e20", later="e20"),
}


def ckpt_path(subset, arm, seed, fold):
    cfg = CONFIG[subset]
    # e20 runs are organised as e20/<subset>/<arm>_s<seed>_f<fold>/<subset>/<arm>_s<seed>/
    # (subset level appears twice, and fold 0 also carries the _f0 suffix).
    if subset in ("FD003", "FD004"):
        return f"{ROOT}/experiments/e20/{subset}/{arm}_s{seed}_f{fold}/{subset}/{arm}_s{seed}/best_model.pt"
    if fold == 0:
        return f"{ROOT}/experiments/{cfg['fold0']}/{arm}_s{seed}/{subset}/{arm}_s{seed}/best_model.pt"
    exp = cfg["later"]
    if subset == "FD002":
        exp = {"A": "e10", "BE": "e10b", "PE": "e10"}[arm]
    return f"{ROOT}/experiments/{exp}/{arm}_s{seed}_f{fold}/{subset}/{arm}_s{seed}/best_model.pt"


def build_model():
    return CandidateModel(num_sensors=14, window_size=WINDOW, d_model=64, num_layers=1,
                          dropout=0.15, use_baseline=True, use_ema_features=True,
                          use_cycle=True, use_graph=False, use_dual_head=False)


def load_model(path):
    """Rebuild the architecture from the run's own summary.json (the training-time
    variant config), not from a one-size-fits-all spec — BE lacks the cycle channel
    and PE carries pe_proj, so a fixed spec silently corrupts both."""
    run_dir = os.path.dirname(path)
    with open(os.path.join(run_dir, "metrics.json")) as fh:
        meta = json.load(fh)
    cfg = meta["config"]
    m = CandidateModel(
        num_sensors=14, window_size=WINDOW, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=cfg["use_baseline"], use_ema_features=cfg["use_ema_features"],
        use_cycle=cfg["use_cycle"], use_pe=cfg.get("use_pe", False),
        use_pe_scramble=cfg.get("use_pe_scramble", False),
        use_graph=cfg["use_graph"], use_dual_head=cfg["use_dual_head"],
        use_sensor_id=cfg.get("use_sensor_id", False),
        use_shuffled_id=cfg.get("use_shuffled_id", False),
        use_fixed_id=cfg.get("use_fixed_id", False),
        capacity_pad=cfg.get("capacity_pad", False),
        use_independent_channels=cfg.get("use_independent_channels", False),
        use_reliability=cfg.get("use_reliability", False),
        gate_mode=cfg.get("gate_mode", "evidence"),
    )
    obj = torch.load(path, map_location="cpu")
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys {unexpected[:5]}"
    assert not missing, f"missing keys {missing[:5]}"
    m.eval()
    return m


def split_frames(subset, fold):
    raw = load_cmapss_train(os.path.join(ROOT, "data", "CMAPSS"), subset)
    units = np.sort(raw["unit"].unique())
    rng = np.random.default_rng(SPLIT_SEED + 20261020)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(units.size * 0.2)))
    start = fold * n_val
    val_units = sorted(int(x) for x in shuffled[start:start + n_val])
    train_units = sorted(int(x) for x in units if int(x) not in set(val_units))
    tr = raw[raw.unit.isin(train_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    va = raw[raw.unit.isin(val_units)].sort_values(["unit", "cycle"]).reset_index(drop=True)
    return raw, tr, va, val_units


def fit_scaler(subset, train_frame):
    """Mirror prepare() exactly: OperatingConditionScaler for the six-condition subsets,
    SensorMinMaxScaler for the single-condition ones. Getting this wrong is exactly what
    the verify mode exists to catch."""
    cols = list(SELECTED_SENSOR_COLUMNS)
    if CONFIG[subset]["cond_aware"]:
        scaler = OperatingConditionScaler(cols, n_clusters=N_COND, random_state=SPLIT_SEED)
        scaler.fit_transform(train_frame)
        return scaler
    scaler = SensorMinMaxScaler(cols)
    scaler.fit_transform(train_frame[cols].to_numpy(np.float64))
    return scaler


def scale_frame(subset, scaler, frame):
    cols = list(SELECTED_SENSOR_COLUMNS)
    if CONFIG[subset]["cond_aware"]:
        return scaler.transform(frame)
    return scaler.transform(frame[cols].to_numpy(np.float64))


def build_test_features(scaled_traj, raw_traj):
    """One window per engine, extras exactly as WindowSet builds them."""
    x, extras, units, lengths = [], [], [], []
    for u, g in raw_traj.groupby("unit", sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(scaled_traj[pos], dtype=np.float32)
        lengths.append((int(u), int(seq.shape[0])))
        if seq.shape[0] < WINDOW:
            continue
        w = seq[-WINDOW:]
        early = seq[:min(5, seq.shape[0])]
        baseline = early.mean(axis=0)
        ref_std = float(early.std(axis=0).mean()) if early.shape[0] > 1 else 0.0
        ref_count = min(1.0, early.shape[0] / 5.0)
        dev = float(np.abs(w - baseline).mean())
        ema = seq[0].copy()
        for r in seq:
            ema = 0.5 * ema + 0.5 * r
        jitter = float(np.abs(seq[-1] - ema).mean())
        mean_age = float(np.mean(np.arange(1, min(early.shape[0], 5) + 1)))
        ev = [ref_count, 1.0, float(np.exp(-mean_age / 30.0)), float(np.exp(-dev)),
              1.0 / (1.0 + ref_std), jitter]
        x.append(np.ascontiguousarray(w))
        extras.append(np.concatenate([baseline, [min(seq.shape[0] / rc.CYCLE_SCALE, 1.0)], ev]))
        units.append(int(u))
    return (np.stack(x).astype(np.float32), np.stack(extras).astype(np.float32),
            units, lengths)


def nasa(err):
    return np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)


def metric_block(err):
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()),
                nasa=float(nasa(err).mean()), late=float((err > 0).mean()),
                severe=float((err > 10).mean()), n=int(err.size))


def verify_subset(subset):
    """Reconstruct fold-0 validation features and compare with prepare()'s matched_set."""
    cond_aware = CONFIG[subset]["cond_aware"]
    _, tr, va, _ = split_frames(subset, 0)
    scaler = fit_scaler(subset, tr)
    va_s = scale_frame(subset, scaler, va)
    x, y, extras, us = rc.build_windows_matched(va, va_s, WINDOW, MAX_RUL, cutoff_seed=2026)
    p = rc.prepare(subset, SPLIT_SEED, WINDOW, MAX_RUL, condition_aware=cond_aware,
                   n_cond=N_COND, ref_mode="global", fold=0)
    ms = p[7]
    dx = float(np.abs(ms.x.numpy() - x).max())
    de = float(np.abs(ms.extras.numpy() - extras).max())
    ok = dx < 1e-6 and de < 1e-6
    print(f"verify {subset}: max|x diff| {dx:.3e}  max|extras diff| {de:.3e}  "
          f"extras dim {extras.shape[1]}  -> {'PASS' if ok else 'FAIL'}")
    return dict(subset=subset, dx=dx, de=de, extras_dim=int(extras.shape[1]), ok=bool(ok))


def evaluate_subset(subset):
    cond_aware = CONFIG[subset]["cond_aware"]
    raw = load_cmapss_train(os.path.join(ROOT, "data", "CMAPSS"), subset)  # for column order only
    del raw
    traj = pd.read_csv(f"{ROOT}/data/CMAPSS/test_{subset}.txt", sep=r"\s+", header=None)
    assert traj.shape[1] == len(CMAPSS_COLUMNS), \
        f"test_{subset}.txt has {traj.shape[1]} cols, expected {len(CMAPSS_COLUMNS)}"
    traj.columns = list(CMAPSS_COLUMNS)
    rul = pd.read_csv(f"{ROOT}/data/CMAPSS/RUL_{subset}.txt", sep=r"\s+", header=None)[0].to_numpy(float)

    lengths = traj.groupby("unit").cycle.max()
    keep_units = sorted(int(u) for u, L in lengths.items() if L >= WINDOW)
    excl = dict(n_excluded=int((lengths < WINDOW).sum()),
                excluded_lengths=[int(L) for L in lengths if L < WINDOW],
                excluded_rul=[float(rul[int(u) - 1]) for u, L in lengths.items() if L < WINDOW])

    per_model, preds = [], {arm: {} for arm in ARMS}
    for fold in FOLDS:
        _, tr, _, _ = split_frames(subset, fold)
        scaler = fit_scaler(subset, tr)
        ts = scale_frame(subset, scaler, traj)
        x, extras, units, _ = build_test_features(ts, traj)
        # Truth is capped at MAX_RUL to match the training target definition and the
        # standard C-MAPSS evaluation convention: the official RUL files report the
        # uncapped remaining life, and comparing a 125-capped prediction against an
        # uncapped truth manufactures an error of (true - 125) on every early-life
        # engine. Amendment 1 to paper/E21_TEST_ALL_PREREGISTRATION.md.
        y_true = np.clip(np.array([rul[u - 1] for u in units], dtype=float), None, MAX_RUL)
        for arm in ARMS:
            for seed in SEEDS:
                path = ckpt_path(subset, arm, seed, fold)
                if not os.path.exists(path):
                    print(f"  MISSING {path}")
                    continue
                m = load_model(path)
                cond = torch.zeros(len(x), WINDOW, 3)
                with torch.no_grad():
                    p = m(torch.from_numpy(x), cond=cond, extras=torch.from_numpy(extras))
                p = p["prediction"].reshape(-1).numpy().astype(float)
                p = np.clip(p, 0.0, MAX_RUL)
                err = p - y_true
                per_model.append(dict(arm=arm, fold=fold, seed=seed, **metric_block(err)))
                for u, pi in zip(units, p):
                    preds[arm].setdefault(u, []).append(pi)

    # ensemble per engine, then an engine-level bootstrap interval
    ens = {}
    for arm in ARMS:
        rows = []
        for u in sorted(preds[arm]):
            # capped truth for metrics (Amendment 1); raw truth kept for provenance
            rows.append((u, float(np.mean(preds[arm][u])),
                         float(min(rul[u - 1], MAX_RUL)), float(rul[u - 1])))
        d = pd.DataFrame(rows, columns=["unit", "y_pred", "y_true", "y_true_raw"])
        d["err"] = d.y_pred - d.y_true
        ens[arm] = d
        rng = np.random.default_rng(RNG)
        acc = np.empty(20000)
        errs = d.err.to_numpy(float)
        for r in range(20000):
            idx = rng.integers(0, len(errs), len(errs))
            acc[r] = np.sqrt((errs[idx] ** 2).mean())
        lo, hi = np.percentile(acc, [2.5, 97.5])
        ens[arm + "_ci"] = [float(lo), float(hi)]
        d.to_csv(f"{ROOT}/results/dsh/e21/ensemble_predictions_{subset}_{arm}.csv", index=False)

    arms_out = {}
    pm = pd.DataFrame(per_model)
    for arm in ARMS:
        g = pm[pm.arm == arm]
        arms_out[arm] = dict(mean_over_models={k: float(g[k].mean()) for k in
                                               ("rmse", "mae", "nasa", "late", "severe")},
                             n_models=int(len(g)),
                             ensemble=metric_block(ens[arm].err.to_numpy(float)),
                             ensemble_rmse_ci95=ens[arm + "_ci"])
    return dict(subset=subset, protocol="last 50-cycle window, cap 125, official RUL truth",
                n_engines_total=int(len(lengths)), n_engines_evaluated=int(len(keep_units)),
                exclusions=excl, arms=arms_out)


def main() -> None:
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "verify"
    os.makedirs(f"{ROOT}/results/dsh/e21", exist_ok=True)
    if mode == "verify":
        res = [verify_subset(s) for s in CONFIG]
        ok = all(r["ok"] for r in res)
        json.dump(res, open(f"{ROOT}/results/dsh/e21/feature_verification.json", "w"), indent=2)
        print("ALL PASS" if ok else "SOME FAILED")
        sys.exit(0 if ok else 1)

    out = {s: evaluate_subset(s) for s in CONFIG}
    json.dump(out, open(f"{ROOT}/results/dsh/e21/test_all.json", "w"), indent=2)
    print("\n=== ensemble RMSE per subset (95% bootstrap CI) ===")
    for s, r in out.items():
        for arm in ARMS:
            a = r["arms"][arm]
            print(f"{s} {arm:3s} {a['ensemble']['rmse']:7.3f} "
                  f"[{a['ensemble_rmse_ci95'][0]:.3f},{a['ensemble_rmse_ci95'][1]:.3f}]  "
                  f"n={a['ensemble']['n']}")


if __name__ == "__main__":
    main()
