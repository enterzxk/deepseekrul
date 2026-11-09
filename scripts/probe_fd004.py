import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_ida as R

print("start", flush=True)
splits, frame = R.outer_folds("FD004", 5)
print("folds ok", flush=True)
tr_u, va_u = splits[0]
print("units", len(tr_u), len(va_u), flush=True)
(xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(
    frame, "FD004", 0, tr_u, va_u, window=50, max_rul=125.0)
print("xtr", np.shape(xtr), "xva", np.shape(xva), "eva", np.shape(eva), flush=True)
print("DONE", flush=True)
