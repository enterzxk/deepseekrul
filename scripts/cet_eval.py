"""Evaluate a trained CET model under the project's truncation protocol.

The training loop reports `rmse_last`, which is the error at each engine's TRUE final
cycle - where the remaining life is zero by construction, so a constant-0 predictor
"scores" 0 and the number means nothing. This script instead truncates every engine at a
random observable cutoff (the `build_windows_matched` protocol, seed 2026), asks for the
RUL distribution at that point, and scores it against the true remaining life.

Reports capped-125 RMSE and NASA score (the benchmark convention), plus distributional
metrics on the true remaining life: CRPS, NLL, 50/90% interval coverage and width.

Usage:
  python cet_eval.py --ckpt experiments/cet/FD001/f0_s2027/model.pt --subset FD001 --fold 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("RUL_ROOT", "/home/yndx/zxk/bearing-rul-pi")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")

import dsh_e21_test_all as e21
from dataset import CMAPSS_COLUMNS
from cet.model import CETRUL, N_SENSORS, K_BINS, RUL_MAX
import pandas as pd

CYCLE_SCALE = 250.0
CUTOFF_SEED = 2026


def sequences(frame, scaled):
    out = []
    for unit, g in frame.groupby("unit", sort=True):
        idx = g.index.to_numpy()
        L = len(idx)
        if L < 20:
            continue
        out.append(dict(unit=int(unit),
                        x=np.asarray(scaled[idx], dtype=np.float32),
                        op=g[["op_1", "op_2", "op_3"]].to_numpy(np.float32),
                        age=g["cycle"].to_numpy(np.float32), L=L))
    return out


def collate(seqs, device):
    T = max(s["L"] for s in seqs)
    B = len(seqs)
    x = np.zeros((B, T, N_SENSORS), np.float32)
    op = np.zeros((B, T, 3), np.float32)
    age = np.zeros((B, T), np.float32)
    mask = np.zeros((B, T, N_SENSORS), np.float32)
    valid = np.zeros((B, T), np.float32)
    for i, s in enumerate(seqs):
        L = s["L"]
        x[i, :L] = s["x"]; op[i, :L] = s["op"]; age[i, :L] = s["age"]
        mask[i, :L] = 1.0; valid[i, :L] = 1.0
    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return t(x), t(op), t(mask), t(age), t(valid)


def nasa_score(err):
    """Asymmetric PHM08 score (late predictions penalised harder), summed per engine."""
    d = err / 13.0
    return float(np.sum(np.where(d < 0, np.exp(-d) - 1.0, np.exp(d) - 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    device = a.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck.get("config", {})
    model = CETRUL(k_bins=cfg.get("k_bins", K_BINS), ablation=cfg.get("ablation", "full"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    raw, tr, va, _ = e21.split_frames(a.subset, a.fold)
    scaler = e21.fit_scaler(a.subset, tr)
    scaled = e21.scale_frame(a.subset, scaler, va)
    seqs = sequences(va, scaled)
    x, op, mask, age, valid = collate(seqs, device)

    with torch.no_grad():
        out = model(x, op, mask, age, valid)
        q_all = out["rul"]                     # (B, T, RUL_MAX) remaining life
        pred_all = model.rul_mean(q_all)
        lo_all, hi_all = model.rul_interval(q_all)
        lo50_all, hi50_all = model.rul_interval(q_all, 0.25, 0.75)

    rng = np.random.default_rng(CUTOFF_SEED)
    rows = []
    for b, s in enumerate(seqs):
        L = s["L"]
        cut = int(round(rng.uniform(40, 0.95 * L)))
        cut = min(max(cut, 10), L - 1)
        true_rul = float(L - cut - 1)
        rows.append(dict(unit=s["unit"], L=L, cut=cut, true=true_rul,
                         pred=float(pred_all[b, cut]),
                         lo=float(lo_all[b, cut]), hi=float(hi_all[b, cut]),
                         lo50=float(lo50_all[b, cut]), hi50=float(hi50_all[b, cut]),
                         gate=float(out["gate"][b, cut]),
                         logE=float(out["logE"][b, cut])))

    true = np.array([r["true"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    lo = np.array([r["lo"] for r in rows])
    hi = np.array([r["hi"] for r in rows])

    cap_t = np.minimum(true, 125.0)
    cap_p = np.minimum(pred, 125.0)
    err = cap_p - cap_t
    # CRPS on the distribution at the cutoff, uncapped truth
    cs = []
    with torch.no_grad():
        for b, r in enumerate(rows):
            q = q_all[b, r["cut"]].unsqueeze(0)
            cs.append(float(model.crps(q, torch.tensor([r["true"]], device=device),
                                       torch.ones(1, device=device))))
    res = dict(
        subset=a.subset, fold=a.fold, n=len(rows),
        rmse_capped=float(np.sqrt(np.mean(err ** 2))),
        nasa_score=nasa_score(err),
        bias=float(np.mean(pred - true)),
        crps=float(np.mean(cs)),
        cover50=float(np.mean([(r["lo50"] <= r["true"] <= r["hi50"]) for r in rows])),
        width50=float(np.mean([r["hi50"] - r["lo50"] for r in rows])),
        cover90=float(np.mean((lo <= true) & (true <= hi))),
        width90=float(np.mean(hi - lo)),
        mean_gate=float(np.mean([r["gate"] for r in rows])),
        per_engine=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                    for r in rows])
    print(json.dumps({k: v for k, v in res.items() if k != "per_engine"}, indent=2))
    out_path = os.path.join(os.path.dirname(a.ckpt), "eval_cutoff.json")
    json.dump(res, open(out_path, "w"), indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
