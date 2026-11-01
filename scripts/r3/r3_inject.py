"""Single-channel step injection: does the bias state absorb a sensor fault?

For each test engine, at 60% of its observed length, add a constant offset (+k sigma, in
z-score units) to ONE sensor channel and keep it on. Then compare

  * delta RUL   : |pred(injected) - pred(clean)| at the last cycle
  * attribution : mean |b_j| on the injected channel vs the other channels
  * health drift: change in the RUL prediction attributable to the health state

A working bias state must show a large |b_j| on the injected channel and a smaller delta
RUL than the no-bias control. If the two models move by the same amount, the operator is
decorative and the line is NO-GO on its own central claim.
"""

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

from r3_train import BiasFilter, build_test_sequences, MAX_RUL  # noqa: E402


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    cfg = ck["config"]
    m = BiasFilter(d_state=cfg["d_state"], use_bias=cfg["bias"],
                   bias_specific=cfg.get("bias_specific", True),
                   bias_step=cfg.get("bias_step", 0.05),
                   bias_decay=cfg.get("bias_decay", 0.995)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


@torch.no_grad()
def run_seq(model, seq, cutoff, device, inject_ch=None, inject_from=None, offset=0.0):
    x = seq["x"][:cutoff].copy()
    if inject_ch is not None:
        x[inject_from:, inject_ch] += offset
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    age = torch.from_numpy(seq["age"][:cutoff]).unsqueeze(0).to(device)
    valid = torch.ones(1, cutoff, device=device)
    out = model(xt, age, valid)
    p = torch.softmax(out["logits"][0, -1], dim=-1)
    mean = (p * torch.arange(p.shape[-1], device=device, dtype=p.dtype)).sum().item()
    b = out["b"][0, -1].cpu().numpy()
    return mean, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--scaler", default="zscore")
    ap.add_argument("--channel", type=int, default=11, help="0-based index into the 14 sensors")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load(a.ckpt, device)
    seqs = build_test_sequences(a.subset, a.fold, a.scaler)

    rows = []
    for s in seqs:
        c = int(0.6 * s["L"])
        if c < 30:
            continue
        p0, b0 = run_seq(model, s, s["L"], device)
        p1, b1 = run_seq(model, s, s["L"], device, inject_ch=a.channel,
                         inject_from=c, offset=a.sigma)
        rows.append(dict(unit=s["unit"], p0=p0, p1=p1, delta=abs(p1 - p0),
                         b_j=float(abs(b1[a.channel])), b_other=float(np.mean(np.delete(np.abs(b1), a.channel)))))
    deltas = np.array([r["delta"] for r in rows])
    bj = np.array([r["b_j"] for r in rows])
    bo = np.array([r["b_other"] for r in rows])
    res = {
        "ckpt": a.ckpt, "n": len(rows),
        "delta_rul_mean": float(deltas.mean()), "delta_rul_median": float(np.median(deltas)),
        "delta_rul_p90": float(np.percentile(deltas, 90)),
        "bias_injected_channel": float(bj.mean()),
        "bias_other_channels": float(bo.mean()),
        "attribution_ratio": float(bj.mean() / (bo.mean() + 1e-9)),
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
