"""Probe: reconstruct the frozen backbone and inspect the data-prep signature.

Step 1 of the feature-extraction pipeline.  It does NOT train anything; it only
loads the frozen weights and prints what prep_fold returns so the extraction script
can be written against the real shapes.

Run on a40:  python scripts/probe_frozen_backbone.py FD001 0
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


def main() -> int:
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    ck_path = ROOT / "experiments/punet_matrix" / subset / f"f{fold}_s42" / "model.pt"
    sd = torch.load(ck_path, map_location="cpu", weights_only=True)
    body_sd = {k[len("body."):]: v for k, v in sd.items() if k.startswith("body.")}
    print(f"checkpoint: {ck_path}  tensors={len(sd)}  body tensors={len(body_sd)}")

    body = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                          dropout=0.15, use_baseline=True, use_ema_features=True,
                          use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
    missing, unexpected = body.load_state_dict(body_sd, strict=False)
    print("missing:", list(missing)[:6])
    print("unexpected:", list(unexpected)[:6])
    body.eval()

    # data prep: inspect what prep_fold hands back
    df = R.load_cmapss_train(str(ROOT / "data/CMAPSS"), subset)
    a, b = R.outer_folds(subset, 5)
    print("elem0:", type(a).__name__, len(a))
    print("  elem0[0]:", type(a[fold]).__name__,
          [getattr(x, "shape", type(x).__name__) for x in a[fold]])
    print("elem1:", type(b).__name__, getattr(b, "shape", len(b)))
    if hasattr(b, "columns"):
        print("  columns:", list(b.columns)[:6])
        print("  head:", b.head(2).to_dict("records"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
