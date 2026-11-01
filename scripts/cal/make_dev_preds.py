"""Rebuild out-of-fold development predictions for every subset from stored checkpoints.

Mirrors E21's own feature construction (split_frames -> fold scaler -> build_windows_matched)
so a calibration fitted on these rows is fitted on the same preprocessing the backbone
was trained and evaluated with. One row per (engine, fold, seed): the engine's
deployment-style matched window.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import dsh_e21_test_all as e21  # noqa: E402
import run_candidate as rc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="FD001,FD002,FD003,FD004")
    ap.add_argument("--arms", default="A")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="2027,2028,2029")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "cal" / "dev_preds"))
    a = ap.parse_args()

    out = Path(a.out)
    for subset in a.subsets.split(","):
        cond_aware = e21.CONFIG[subset]["cond_aware"]
        for fold in [int(x) for x in a.folds.split(",")]:
            _, tr, va, val_units = e21.split_frames(subset, fold)
            scaler = e21.fit_scaler(subset, tr)
            va_s = e21.scale_frame(subset, scaler, va)
            x, y, extras, units = rc.build_windows_matched(va, va_s, e21.WINDOW, e21.MAX_RUL,
                                                           cutoff_seed=2026)
            for arm in a.arms.split(","):
                for seed in [int(s) for s in a.seeds.split(",")]:
                    path = e21.ckpt_path(subset, arm, seed, fold)
                    if not os.path.exists(path):
                        print(f"MISSING {path}")
                        continue
                    model = e21.load_model(path)
                    cond = torch.zeros(len(x), e21.WINDOW, 3)
                    with torch.no_grad():
                        p = model(torch.from_numpy(x), cond=cond, extras=torch.from_numpy(extras))
                    p = np.clip(p["prediction"].reshape(-1).numpy().astype(float), 0, e21.MAX_RUL)
                    d = out / subset
                    d.mkdir(parents=True, exist_ok=True)
                    f = d / f"{arm}_s{seed}_f{fold}.csv"
                    with open(f, "w") as fh:
                        fh.write("unit_id,y_true,y_pred\n")
                        for u, t, pi in zip(units, y, p):
                            fh.write(f"{int(u)},{float(t):.4f},{float(pi):.6f}\n")
                    print(subset, f"fold{fold} {arm} s{seed}", "n", len(units), flush=True)


if __name__ == "__main__":
    main()
