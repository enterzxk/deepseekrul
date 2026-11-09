"""Check that prep_fold's validation windows match preds_cqr.csv row by row.

Needed before any same-baseline comparison: E's scale must be applied to the same
windows the frozen CQR scored.

Run on a40: python scripts/check_alignment.py FD001 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_ida as R  # noqa: E402


def main() -> int:
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    preds = pd.read_csv(ROOT / "experiments/punet_matrix" / subset / f"f{fold}_s42" / "preds_cqr.csv")
    splits, frame = R.outer_folds(subset, 5)
    tr_u, va_u = splits[fold]
    (_, _, _, _, _), (xva, yva, eva, mva, _) = R.prep_fold(
        frame, subset, fold, tr_u, va_u, window=50, max_rul=125.0)
    m = np.asarray(mva)
    print("preds rows", len(preds), " prep rows", len(yva))
    if len(preds) != len(yva):
        print("LENGTH MISMATCH")
        return 1
    ok_y = np.allclose(preds.y_true.to_numpy(), np.asarray(yva, float), atol=1e-6)
    ok_u = np.array_equal(preds.unit.to_numpy(), m[:, 0].astype(int))
    ok_c = np.allclose(preds.cutoff.to_numpy(), m[:, 1], atol=1e-6)
    print("y_true match:", ok_y)
    print("unit match:", ok_u)
    print("cutoff match:", ok_c)
    print("RESULT:", "ALIGNED" if (ok_y and ok_u and ok_c) else "NOT ALIGNED")
    return 0 if (ok_y and ok_u and ok_c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
