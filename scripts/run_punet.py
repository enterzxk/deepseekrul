"""CLI for PU-Net: train / calibrate / evaluate / diagnose.

Examples
--------
python scripts/run_punet.py --subset FD001 --stage smoke
python scripts/run_punet.py --subset FD001 --stage train --epochs 30 --lam 0.5
python scripts/run_punet.py --subset FD001 --stage evaluate --delta 0.1
python scripts/run_punet.py --subset FD001 --stage train --ablate A4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ABLATIONS = {
    "none": dict(use_scale_head=True, use_bias_head=True, use_sigma=True),
    "A1": dict(use_scale_head=False, use_bias_head=True, use_sigma=True),
    "A2": dict(use_scale_head=True, use_bias_head=True, use_sigma=False),
    "A4": dict(use_scale_head=True, use_bias_head=True, use_sigma=True),  # sigma shuffled at load time
    "A6": dict(use_scale_head=True, use_bias_head=False, use_sigma=True),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PU-Net runner")
    p.add_argument("--subset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--data-dir", default=str(ROOT / "data" / "CMAPSS"))
    p.add_argument("--stage", default="smoke", choices=["smoke", "train", "calibrate", "evaluate", "diagnose"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1e-4)
    p.add_argument("--delta", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablate", default="none", choices=sorted(ABLATIONS))
    p.add_argument("--out-dir", default=str(ROOT / "results" / "punet"))
    return p


def smoke(args) -> int:
    """Shape-level smoke test that does not need the real backbone."""
    import torch
    from src.punet import PUNetModel, PriorFeatureBuilder, block_split
    from src.punet.conformal import BlockConformalCalibrator

    class TinyBackbone(torch.nn.Module):
        def __init__(self, d=16):
            super().__init__()
            self.net = torch.nn.Linear(3, 1)
            self.enc = torch.nn.Linear(3, d)

        def encode(self, x):
            pooled = x.mean(dim=1)
            return self.enc(pooled), self.net(pooled).squeeze(-1)

    class FakePrior:
        def lookup(self, fold, unit, t):
            return 100.0 + unit, 5.0 + 0.1 * unit, 0, t

    torch.manual_seed(args.seed)
    model = PUNetModel(TinyBackbone(), d_model=16, **ABLATIONS[args.ablate])
    feats = PriorFeatureBuilder(FakePrior()).build([1, 2, 3, 4], [10, 20, 30, 40])
    x = torch.randn(4, 30, 3)
    out = model(x, None, feats)
    assert out.y_hat.shape == (4,), out.y_hat.shape
    assert out.w.shape == (4,), out.w.shape
    assert bool((out.w > 0).all())
    assert torch.allclose(out.y_hat, out.y_base), "bias head must start at zero"

    scores = BlockConformalCalibrator.scores(
        torch.tensor([10.0, 20, 30, 40]).numpy(), out.y_hat.detach().numpy(), out.w.detach().numpy()
    )
    cal = BlockConformalCalibrator(delta=args.delta).fit(scores)
    lo, hi = cal.interval(out.y_hat.detach().numpy(), out.w.detach().numpy())
    assert (hi > lo).all()

    split = block_split([1, 1, 2, 2, 3, 3, 4, 4], [10, 11, 20, 21, 30, 31, 40, 41], seed=args.seed)
    print(json.dumps({"stage": "smoke", "split": split.describe(), "q_hat": cal.q_hat}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "smoke":
        return smoke(args)

    out_dir = Path(args.out_dir) / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "stage {} needs the real backbone and prior table; wire them in before running".format(args.stage)
    )


if __name__ == "__main__":
    raise SystemExit(main())
