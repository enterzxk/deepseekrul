import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from src.candidates.model import CandidateModel  # noqa: E402

sd = torch.load(ROOT / "experiments/punet_matrix/FD001/f0_s42/model.pt",
                map_location="cpu", weights_only=True)
body_sd = {k[len("body."):]: v for k, v in sd.items() if k.startswith("body.")}
body = CandidateModel(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                      dropout=0.15, use_baseline=True, use_ema_features=True,
                      use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)
body.load_state_dict(body_sd, strict=True)
body.eval()

for name, mod in body.named_modules():
    if name and name.count(".") <= 2:
        print(f"{name}: {type(mod).__name__}")
