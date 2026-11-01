"""Band-wise error and calibration of a trained R3 filter on the official test set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("RUL_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "r3"))

from r3_train import BiasFilter, build_test_sequences, collate, MAX_RUL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--scaler", default="zscore")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    cfg = ck["config"]
    model = BiasFilter(d_state=cfg["d_state"], use_bias=cfg["bias"],
                       bias_specific=cfg.get("bias_specific", True),
                       bias_step=cfg.get("bias_step", 0.05),
                       bias_decay=cfg.get("bias_decay", 0.995)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    seqs = build_test_sequences(a.subset, a.fold, a.scaler)
    x, age, valid, _ = collate(seqs, device)
    with torch.no_grad():
        logits = model(x, age, valid)["logits"]
        p = torch.softmax(logits, dim=-1)
        mean = (p * torch.arange(p.shape[-1], device=device, dtype=p.dtype)).sum(-1)[:, -1].cpu().numpy()
        cdf = torch.cumsum(p, dim=-1)
        lo_q = torch.searchsorted(cdf[:, -1], torch.full((len(seqs), 1), 0.05, device=device)).squeeze(-1)
        hi_q = torch.searchsorted(cdf[:, -1], torch.full((len(seqs), 1), 0.95, device=device)).squeeze(-1)
        lo, hi = lo_q.cpu().numpy(), hi_q.cpu().numpy()
    truth = np.array([min(s["rul_end"], MAX_RUL) for s in seqs], dtype=np.float32)

    rows = []
    bands = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 126)]
    for lo_b, hi_b in bands:
        m = (truth >= lo_b) & (truth < hi_b)
        if m.sum() == 0:
            continue
        err = mean[m] - truth[m]
        rows.append({"band": f"[{lo_b},{hi_b})", "n": int(m.sum()),
                     "rmse": float(np.sqrt(np.mean(err ** 2))),
                     "bias": float(np.mean(err)),
                     "pred_mean": float(np.mean(mean[m])),
                     "cover90": float(np.mean((lo[m] <= truth[m]) & (truth[m] <= hi[m]))),
                     "width90": float(np.mean(hi[m] - lo[m]))})
    overall = {"n": int(len(truth)), "rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
               "bias": float(np.mean(mean - truth)),
               "cover90": float(np.mean((lo <= truth) & (truth <= hi))),
               "width90": float(np.mean(hi - lo))}
    print(json.dumps({"overall": overall, "bands": rows}, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({"overall": overall, "bands": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
