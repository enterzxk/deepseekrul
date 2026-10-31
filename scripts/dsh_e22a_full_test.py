"""E22a — full official-test evaluation (all 707 engines) with the frozen padding rule.

Protocol frozen in paper/E22A_FULL_TEST_PREREGISTRATION.md (sha recorded in the E22
ledger): short trajectories (<50 cycles) are left-padded by repeating the first frame,
exactly as src/dataset.py:_last_window_with_edge_padding does for the training-side
test-window builder. Metrics are reported under BOTH truth conventions:

  capped   truth = min(RUL, 125)  — agreement with the training target definition,
                                   the convention for cross-paper RMSE comparison
  uncapped truth = RUL as given   — error against the real remaining life, the quantity
                                   a maintenance decision depends on; the convention
                                   under which the NASA late-penalty is operational

Outputs: results/dsh/e22a/full_test.json,
         results/dsh/e22a/preds_<subset>_<arm>.csv (unit, y_pred, y_true_capped, y_true_raw)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = "/home/yndx/zxk/bearing-rul-pi"
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")

import dsh_e21_test_all as e21  # frozen helper functions (ckpt_path, load_model, ...)
from dataset import CMAPSS_COLUMNS, _last_window_with_edge_padding  # noqa: E402

OUT_DIR = f"{ROOT}/results/dsh/e22a"
WINDOW, MAX_RUL = e21.WINDOW, e21.MAX_RUL
ARMS, SEEDS, FOLDS = e21.ARMS, e21.SEEDS, e21.FOLDS
RNG = 9000
BOOT = 20_000


def pad_window(seq: np.ndarray) -> np.ndarray:
    return _last_window_with_edge_padding(seq, WINDOW)


def build_full_features(scaled_traj, raw_traj):
    """One window per engine, padding short trajectories; extras as WindowSet builds them."""
    x, extras, units, lengths = [], [], [], []
    for u, g in raw_traj.groupby("unit", sort=True):
        pos = g.index.to_numpy(np.int64)
        seq = np.asarray(scaled_traj[pos], dtype=np.float32)
        lengths.append((int(u), int(seq.shape[0])))
        w = pad_window(seq)
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
        extras.append(np.concatenate([baseline,
                                      [min(seq.shape[0] / e21.rc.CYCLE_SCALE, 1.0)], ev]))
        units.append(int(u))
    return (np.stack(x).astype(np.float32), np.stack(extras).astype(np.float32),
            units, lengths)


def metrics_block(err):
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()),
                nasa_mean=float(e21.nasa(err).mean()),
                nasa_sum=float(e21.nasa(err).sum()),
                late=float((err > 0).mean()), severe=float((err > 10).mean()),
                n=int(err.size))


def evaluate(subset):
    cond_aware = e21.CONFIG[subset]["cond_aware"]
    traj = pd.read_csv(f"{ROOT}/data/CMAPSS/test_{subset}.txt", sep=r"\s+", header=None)
    assert traj.shape[1] == len(CMAPSS_COLUMNS)
    traj.columns = list(CMAPSS_COLUMNS)
    rul = pd.read_csv(f"{ROOT}/data/CMAPSS/RUL_{subset}.txt", sep=r"\s+",
                      header=None)[0].to_numpy(float)

    per_model, preds = [], {arm: {} for arm in ARMS}
    for fold in FOLDS:
        _, tr, _, _ = e21.split_frames(subset, fold)
        scaler = e21.fit_scaler(subset, tr)
        ts = e21.scale_frame(subset, scaler, traj)
        x, extras, units, _ = build_full_features(ts, traj)
        raw_true = np.array([rul[u - 1] for u in units], dtype=float)
        for arm in ARMS:
            for seed in SEEDS:
                m = e21.load_model(e21.ckpt_path(subset, arm, seed, fold))
                cond = torch.zeros(len(x), WINDOW, 3)
                with torch.no_grad():
                    p = m(torch.from_numpy(x), cond=cond, extras=torch.from_numpy(extras))
                p = np.clip(p["prediction"].reshape(-1).numpy().astype(float), 0.0, MAX_RUL)
                per_model.append(dict(arm=arm, fold=fold, seed=seed,
                                      **{f"cap_{k}": v for k, v in
                                         metrics_block(p - np.minimum(raw_true, MAX_RUL)).items()},
                                      **{f"raw_{k}": v for k, v in
                                         metrics_block(p - raw_true).items()}))
                for u, pi in zip(units, p):
                    preds[arm].setdefault(u, []).append(pi)

    arms_out = {}
    pm = pd.DataFrame(per_model)
    for arm in ARMS:
        rows = []
        for u in sorted(preds[arm]):
            rows.append((u, float(np.mean(preds[arm][u])),
                         float(min(rul[u - 1], MAX_RUL)), float(rul[u - 1])))
        d = pd.DataFrame(rows, columns=["unit", "y_pred", "y_true_capped", "y_true_raw"])
        err_cap = d.y_pred.to_numpy() - d.y_true_capped.to_numpy()
        err_raw = d.y_pred.to_numpy() - d.y_true_raw.to_numpy()
        rng = np.random.default_rng(RNG)
        acc_cap, acc_raw = np.empty(BOOT), np.empty(BOOT)
        for r in range(BOOT):
            idx = rng.integers(0, len(err_cap), len(err_cap))
            acc_cap[r] = np.sqrt((err_cap[idx] ** 2).mean())
            acc_raw[r] = np.sqrt((err_raw[idx] ** 2).mean())
        lo_c, hi_c = np.percentile(acc_cap, [2.5, 97.5])
        lo_r, hi_r = np.percentile(acc_raw, [2.5, 97.5])
        d.to_csv(f"{OUT_DIR}/preds_{subset}_{arm}.csv", index=False)
        g = pm[pm.arm == arm]
        arms_out[arm] = dict(
            n_models=int(len(g)),
            per_model_mean={k: float(g[k].mean()) for k in g if k.startswith(("cap_", "raw_"))},
            ensemble_capped=dict(metrics_block(err_cap), rmse_ci95=[float(lo_c), float(hi_c)]),
            ensemble_uncapped=dict(metrics_block(err_raw),
                                   rmse_ci95=[float(lo_r), float(hi_r)]),
        )
    return dict(subset=subset, protocol="all engines, frozen first-frame pad, last window",
                n_engines=int(len(rul)), arms=arms_out)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {s: evaluate(s) for s in ("FD001", "FD002", "FD003", "FD004")}
    json.dump(out, open(f"{OUT_DIR}/full_test.json", "w"), indent=2)
    print("\n=== full-set ensemble RMSE: capped | uncapped (95% CI capped) ===")
    for s, r in out.items():
        for arm in ARMS:
            a = r["arms"][arm]
            c, u = a["ensemble_capped"], a["ensemble_uncapped"]
            print(f"{s} {arm:3s} capped {c['rmse']:7.3f} [{c['rmse_ci95'][0]:.3f},"
                  f"{c['rmse_ci95'][1]:.3f}]  uncapped {u['rmse']:7.3f}"
                  f"  nasa_mean cap {c['nasa_mean']:6.1f} raw {u['nasa_mean']:6.1f}  n={c['n']}")
    print(f"wrote {OUT_DIR}/full_test.json")


if __name__ == "__main__":
    main()
