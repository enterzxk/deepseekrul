"""Check whether the A2 switch-event penalty can fire.

Prints the OC-switch threshold (90th percentile of the L2 norm of consecutive
condition-plane differences on TRAINING windows) and the fraction of switch
steps in both training and validation windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ida as R  # noqa: E402


def main() -> int:
    for subset in ["FD002", "FD004"]:
        folds, df = R.outer_folds(subset)
        tr_u, va_u = folds[0]
        (_, _, _, mtr, ctr), (_, _, _, mva, cva) = R.prep_fold(df, subset, 0, tr_u, va_u, window=50)
        for name, c in [("train", ctr), ("val", cva)]:
            if c is None:
                continue
            d = np.sqrt((np.diff(c, axis=1) ** 2).sum(axis=-1))
            tau = float(np.quantile(d.reshape(-1), 0.90))
            frac = float((d > tau).mean())
            print(f"{subset} {name}: tau={tau:.4f} switch_frac={frac:.4f} "
                  f"zero_frac={(d == 0).mean():.4f} max={d.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
