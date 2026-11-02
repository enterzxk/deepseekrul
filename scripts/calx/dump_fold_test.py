"""Dump each fold model's prediction on the official test set.

The only artefact the E21 pipeline never wrote is the per-(fold, seed) test prediction:
`evaluate_subset` averages all 15 fold models before it touches disk, so the deployed
number is a 15-ensemble whose members cannot be recovered. Every cross-fitted method
needs those members -- a CV+ interval is built from the *spread* of fold models on the
test point, not from their mean.

This script re-runs inference with the fold's own scaler (the same object
`evaluate_subset` builds from that fold's training frame) and writes one CSV per
(subset, seed, fold) with the same schema as `experiments/cal/dev_preds`.

Protocol: paper/CVPLUS_PROTOCOL.md, section 2. Nothing here changes a reported number;
it only materialises a prediction the pipeline computed and discarded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts"))

import dsh_e21_test_all as e21  # noqa: E402
from dataset import CMAPSS_COLUMNS  # noqa: E402


def load_test_frame(subset):
    traj = pd.read_csv(f"{ROOT}/data/CMAPSS/test_{subset}.txt", sep=r"\s+", header=None)
    assert traj.shape[1] == len(CMAPSS_COLUMNS), f"test_{subset}.txt: {traj.shape[1]} cols"
    traj.columns = list(CMAPSS_COLUMNS)
    rul = pd.read_csv(f"{ROOT}/data/CMAPSS/RUL_{subset}.txt", sep=r"\s+", header=None)[0]
    return traj, rul.to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--arms", default="A")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="2027,2028,2029")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "calx" / "fold_test"))
    a = ap.parse_args()
    out = Path(a.out)

    for subset in a.subsets.split(","):
        traj, rul = load_test_frame(subset)
        for fold in [int(x) for x in a.folds.split(",")]:
            _, tr, _, _ = e21.split_frames(subset, fold)
            scaler = e21.fit_scaler(subset, tr)
            ts = e21.scale_frame(subset, scaler, traj)
            x, extras, units, _ = e21.build_test_features(ts, traj)
            y_true = np.clip(np.array([rul[u - 1] for u in units], dtype=float), None, e21.MAX_RUL)
            for arm in a.arms.split(","):
                for seed in [int(s) for s in a.seeds.split(",")]:
                    path = e21.ckpt_path(subset, arm, seed, fold)
                    if not os.path.exists(path):
                        print(f"MISSING {path}", flush=True)
                        continue
                    model = e21.load_model(path)
                    cond = torch.zeros(len(x), e21.WINDOW, 3)
                    with torch.no_grad():
                        p = model(torch.from_numpy(x), cond=cond, extras=torch.from_numpy(extras))
                    p = np.clip(p["prediction"].reshape(-1).numpy().astype(float), 0.0, e21.MAX_RUL)
                    d = out / subset
                    d.mkdir(parents=True, exist_ok=True)
                    f = d / f"{arm}_s{seed}_f{fold}.csv"
                    with open(f, "w") as fh:
                        fh.write("unit_id,y_true,y_pred\n")
                        for u, t, pi in zip(units, y_true, p):
                            fh.write(f"{int(u)},{float(t):.4f},{float(pi):.6f}\n")
                    print(subset, f"fold{fold} {arm} s{seed}", "n", len(units), flush=True)


if __name__ == "__main__":
    main()
