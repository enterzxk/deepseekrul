"""Extract frozen backbone features for the two-level scale experiment.

Loads the frozen E2ECQR body from experiments/punet_matrix, runs ONE forward pass
over the validation windows of a fold, and caches (h, y, unit, cutoff).

No training, no gradient, no weight updates.  The backbone is frozen and eval().

Run on a40:
    python scripts/extract_features.py FD001 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_ida as R  # noqa: E402
from src.candidates.model import CandidateModel  # noqa: E402


def build_body(ck_path: Path) -> CandidateModel:
    sd = torch.load(ck_path, map_location="cpu", weights_only=True)
    body_sd = {k[len("body."):]: v for k, v in sd.items() if k.startswith("body.")}
    body = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                          dropout=0.15, use_baseline=True, use_ema_features=True,
                          use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
    body.load_state_dict(body_sd, strict=True)
    body.eval()
    return body


def main() -> int:
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ck = ROOT / "experiments/punet_matrix" / subset / f"f{fold}_s42" / "model.pt"
    body = build_body(ck)
    print("body loaded")

    splits, frame = R.outer_folds(subset, 5)
    tr_u, va_u = splits[fold]
    (xtr, ytr, etr, mtr, ctr), (xva, yva, eva, mva, cva) = R.prep_fold(
        frame, subset, fold, tr_u, va_u, window=50, max_rul=125.0)
    print(f"xtr {np.shape(xtr)}  xva {np.shape(xva)}  yva {np.shape(yva)}  eva {np.shape(eva)}")
    print("meta type:", type(mva).__name__,
          getattr(mva, "columns", None) if hasattr(mva, "columns") else np.shape(mva))

    x = torch.as_tensor(np.asarray(xva), dtype=torch.float32)
    e = torch.as_tensor(np.asarray(eva), dtype=torch.float32)
    c = None if cva is None else torch.as_tensor(np.asarray(cva), dtype=torch.float32)
    with torch.no_grad():
        try:
            out = body.encode(x, c, e)
        except TypeError:
            try:
                out = body.encode(x, e)
            except TypeError:
                out = body.encode(x)
    print("encode ->", type(out).__name__, len(out) if hasattr(out, "__len__") else "n/a")
    if isinstance(out, (tuple, list)):
        for i, part in enumerate(out):
            print(f"  [{i}] {type(part).__name__} {tuple(part.shape) if hasattr(part, 'shape') else part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
