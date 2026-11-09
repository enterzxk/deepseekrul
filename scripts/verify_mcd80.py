"""Integrity check for the MC-Dropout 80-cell backfill.

Two questions, both about whether the new cells are comparable to the frozen
baseline in experiments/punet_matrix:

1. For the 40 cells that had NO checkpoint (seeds 2024/13), the point model was
   retrained with --save-ckpt. cuDNN determinism + a fixed seed should reproduce
   the stored preds_point.csv. If it does not, the retrained model is a
   different model and pairing it against the stored CQR run is invalid.
2. For the 10 cells that already had a mcdropout.json, rerunning the same code
   must reproduce the stored numbers.

Writes experiments/punet_mcd80/verify_mcd80.csv and prints a summary.
Exit code 1 if any integrity check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAT = Path("experiments/punet_matrix")
B = Path("experiments/punet_mcd80")
TOL = 1e-3


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main() -> int:
    rows = []
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        for seed in [42, 7, 2024, 13]:
            for fold in range(5):
                d, b = MAT / sub / f"f{fold}_s{seed}", B / sub / f"f{fold}_s{seed}"
                rec = dict(subset=sub, seed=seed, fold=fold,
                           had_ckpt=(d / "model.pt").exists(),
                           has_mcd=(b / "mcdropout.json").exists(),
                           retrained=(b / "model.pt").exists())

                # (1) retrained point model vs frozen point predictions
                if rec["retrained"] and (b / "preds_point.csv").exists() and (d / "preds_point.csv").exists():
                    a = np.genfromtxt(d / "preds_point.csv", delimiter=",", names=True)
                    c = np.genfromtxt(b / "preds_point.csv", delimiter=",", names=True)
                    rec["max_d_y_lo"] = max_abs_diff(a["y_lo"], c["y_lo"])
                    rec["max_d_y_true"] = max_abs_diff(a["y_true"], c["y_true"])
                    rec["n_row_match"] = int(a.shape == c.shape)

                # (2) rerun of an existing MC-Dropout cell
                old = d / "mcdropout.json"
                if old.exists() and (b / "mcdropout.json").exists():
                    o, n = load_json(old), load_json(b / "mcdropout.json")
                    rec["d_picp"] = abs(o["picp"] - n["picp"])
                    rec["d_winkler"] = abs(o["winkler"] - n["winkler"])
                    rec["d_qhat"] = abs(o["q_hat"] - n["q_hat"])

                # (3) new cell summary numbers
                if (b / "mcdropout.json").exists():
                    n = load_json(b / "mcdropout.json")
                    rec["picp"] = n["picp"]
                    rec["pinaw"] = n["pinaw"]
                    rec["winkler"] = n["winkler"]
                    rec["engine_min"] = n.get("engine_min")
                    rec["n_test"] = n.get("n_test")
                rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(B / "verify_mcd80.csv", index=False)

    print(f"cells with mcdropout.json: {int(df.has_mcd.sum())}/80")
    print(f"cells retrained:           {int(df.retrained.sum())}/40")
    if "max_d_y_lo" in df:
        v = df["max_d_y_lo"].dropna()
        print(f"retrained vs frozen y_lo: n={len(v)} max|d|={v.max():.3e} "
              f"median={v.median():.3e} >{TOL}: {int((v > TOL).sum())}")
    for col in ["d_picp", "d_winkler", "d_qhat"]:
        if col in df:
            v = df[col].dropna()
            print(f"rerun vs stored {col}: n={len(v)} max={v.max():.3e}")

    ok = True
    if "max_d_y_lo" in df and (df["max_d_y_lo"] > TOL).any():
        print("INTEGRITY FAIL: retrained point models differ from frozen preds")
        ok = False
    if "d_picp" in df and (df["d_picp"] > 1e-9).any():
        print("INTEGRITY WARN: rerun does not reproduce stored MC-Dropout picp")
        ok = False
    if int(df.has_mcd.sum()) != 80:
        print("INCOMPLETE: fewer than 80 MC-Dropout cells")
        ok = False
    print("VERIFY_OK" if ok else "VERIFY_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
