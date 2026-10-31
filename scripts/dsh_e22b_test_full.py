"""E22b — test evaluation: Dual-Mixer vs paired A-zscore under the E22a protocol.

Both arms see the same z-score inputs (per-fold training statistics), the same fold
partition, and the same 707-engine test protocol (last observable window, first-frame
pad, capped truth headline + uncapped alongside).

Outputs: results/dsh/e22b/test_full.json, preds_<subset>_<arm>.csv
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = "/home/yndx/zxk/bearing-rul-pi"
REPO = f"{ROOT}/external/dualmixer_ref"
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{REPO}/models/RULPrediction")
sys.path.insert(0, REPO)

import dsh_e21_test_all as e21  # noqa: E402
import dsh_e22a_full_test as e22a  # noqa: E402  (frozen padding + extras builder)
from dataset import CMAPSS_COLUMNS, SELECTED_SENSOR_COLUMNS  # noqa: E402
from MLPMixer import DualMLPMixer  # noqa: E402
from candidates.model import CandidateModel  # noqa: E402

WINDOW, MAX_RUL = e21.WINDOW, e21.MAX_RUL
ARMS = ("A_zscore", "DualMixer")
SUBSET = sys.argv[sys.argv.index("--subset") + 1] if "--subset" in sys.argv else "FD001"
OUT = f"{ROOT}/results/dsh/e22b"
os.makedirs(OUT, exist_ok=True)


def build_candidate(path):
    cfg = json.load(open(os.path.join(os.path.dirname(path), "metrics.json")))["config"]
    m = CandidateModel(
        num_sensors=14, window_size=WINDOW, d_model=64, num_layers=1, dropout=0.15,
        use_baseline=cfg["use_baseline"], use_ema_features=cfg["use_ema_features"],
        use_cycle=cfg["use_cycle"], use_pe=cfg.get("use_pe", False),
        use_pe_scramble=cfg.get("use_pe_scramble", False),
        use_graph=cfg["use_graph"], use_dual_head=cfg["use_dual_head"],
    )
    obj = torch.load(path, map_location="cpu")
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected and not missing, (missing[:3], unexpected[:3])
    m.eval()
    return m


def build_mixer(path):
    m = DualMLPMixer(window_size=WINDOW, in_features=len(SELECTED_SENSOR_COLUMNS),
                     hidden_dim=32, num_layers=6, dropout=0.0, device="cpu",
                     model_flag="dm", label_norm=False, filter_size=0)
    state = torch.load(path, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def candidate_ckpt(fold, seed):
    return f"{ROOT}/experiments/e22b/A_zscore/{SUBSET}_f{fold}_s{seed}/{SUBSET}/A_s{seed}/best_model.pt"


def mixer_ckpt(fold, seed):
    return f"{ROOT}/experiments/e22b/{SUBSET}/f{fold}_s{seed}/best_model.pt"


def metrics_block(err):
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mae=float(np.abs(err).mean()),
                nasa_mean=float(e21.nasa(err).mean()), nasa_sum=float(e21.nasa(err).sum()),
                late=float((err > 0).mean()), severe=float((err > 10).mean()), n=int(err.size))


def main():
    traj = pd.read_csv(f"{ROOT}/data/CMAPSS/test_{SUBSET}.txt", sep=r"\s+", header=None)
    assert traj.shape[1] == len(CMAPSS_COLUMNS)
    traj.columns = list(CMAPSS_COLUMNS)
    rul = pd.read_csv(f"{ROOT}/data/CMAPSS/RUL_{SUBSET}.txt", sep=r"\s+",
                      header=None)[0].to_numpy(float)
    cols = list(SELECTED_SENSOR_COLUMNS)
    preds = {arm: {} for arm in ARMS}
    per_model = []
    for fold in e21.FOLDS:
        _, tr, _, _ = e21.split_frames(SUBSET, fold)
        mu = tr[cols].mean().to_numpy()
        sd = tr[cols].std().to_numpy().clip(1e-6)
        ts = ((traj[cols].to_numpy(np.float64) - mu) / sd).astype(np.float32)
        x, extras, units, _ = e22a.build_full_features(ts, traj)
        raw_true = np.array([rul[u - 1] for u in units], dtype=float)
        cond = torch.zeros(len(x), WINDOW, 3)
        for seed in e21.SEEDS:
            mc = build_candidate(candidate_ckpt(fold, seed))
            with torch.no_grad():
                p = mc(torch.from_numpy(x), cond=cond,
                       extras=torch.from_numpy(extras))["prediction"].reshape(-1).numpy()
            p = np.clip(p.astype(float), 0.0, MAX_RUL)
            per_model.append(dict(arm="A_zscore", fold=fold, seed=seed,
                                  **{f"cap_{k}": v for k, v in
                                     metrics_block(p - np.minimum(raw_true, MAX_RUL)).items()}))
            for u, pi in zip(units, p):
                preds["A_zscore"].setdefault(u, []).append(pi)

            dm = build_mixer(mixer_ckpt(fold, seed))
            with torch.no_grad():
                q = dm(torch.from_numpy(x)).reshape(-1).numpy() * MAX_RUL
            q = np.clip(q.astype(float), 0.0, MAX_RUL)
            per_model.append(dict(arm="DualMixer", fold=fold, seed=seed,
                                  **{f"cap_{k}": v for k, v in
                                     metrics_block(q - np.minimum(raw_true, MAX_RUL)).items()}))
            for u, qi in zip(units, q):
                preds["DualMixer"].setdefault(u, []).append(qi)

    out = dict(subset=SUBSET, protocol="E22a full-set, z-score inputs, both arms")
    pm = pd.DataFrame(per_model)
    for arm in ARMS:
        rows = [(u, float(np.mean(preds[arm][u])), float(min(rul[u - 1], MAX_RUL)),
                 float(rul[u - 1])) for u in sorted(preds[arm])]
        d = pd.DataFrame(rows, columns=["unit", "y_pred", "y_true_capped", "y_true_raw"])
        err_c = d.y_pred.to_numpy() - d.y_true_capped.to_numpy()
        err_r = d.y_pred.to_numpy() - d.y_true_raw.to_numpy()
        d.to_csv(f"{OUT}/preds_{SUBSET}_{arm}.csv", index=False)
        g = pm[pm.arm == arm]
        out[arm] = dict(n_models=int(len(g)),
                        per_model_capped_rmse_mean=float(g.cap_rmse.mean()),
                        per_model_capped_rmse_sd=float(g.cap_rmse.std(ddof=1)),
                        ensemble_capped=metrics_block(err_c),
                        ensemble_uncapped=metrics_block(err_r))
    json.dump(out, open(f"{OUT}/test_full.json", "w"), indent=2)
    for arm in ARMS:
        c = out[arm]["ensemble_capped"]
        u = out[arm]["ensemble_uncapped"]
        print(f"{SUBSET} {arm:9s} capped {c['rmse']:6.3f}  uncapped {u['rmse']:6.3f}  "
              f"(per-model mean {out[arm]['per_model_capped_rmse_mean']:.3f} "
              f"± {out[arm]['per_model_capped_rmse_sd']:.3f})  n={c['n']}")
    print(f"wrote {OUT}/test_full.json")


if __name__ == "__main__":
    main()
