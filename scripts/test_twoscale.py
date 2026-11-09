"""CPU unit test for the two-level scale module (no training, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.twoscale.model import TwoScaleModel  # noqa: E402


class StubBackbone(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.d = d
        self.lin = nn.Linear(4, d)

    def encode(self, x, adjacency=None):
        h = torch.tanh(self.lin(x))
        return h, h.mean(dim=1) * 3.0


def main() -> int:
    torch.manual_seed(0)
    n_eng, per, d = 5, 20, 16
    x = torch.randn(n_eng * per, 4)
    ids = torch.arange(n_eng).repeat_interleave(per)

    model = TwoScaleModel(StubBackbone(d), d)
    out = model(x, ids, n_engines=n_eng)

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
        ok = ok and bool(cond)

    check("shapes", out.scale.shape == (n_eng * per,), str(tuple(out.scale.shape)))
    check("scale == win * eng", torch.allclose(out.scale, out.sigma_win * out.sigma_eng))

    # within-engine mean of sigma_win is 1
    means = torch.stack([out.sigma_win[ids == e].mean() for e in range(n_eng)])
    check("sigma_win mean 1 per engine", torch.allclose(means, torch.ones_like(means), atol=1e-4),
          str([round(float(m), 5) for m in means]))

    # sigma_eng is constant within an engine and varies across engines
    const = all(torch.allclose(out.sigma_eng[ids == e], out.sigma_eng[ids == e][0]) for e in range(n_eng))
    check("sigma_eng constant within engine", const)
    eng_vals = torch.stack([out.sigma_eng[ids == e][0] for e in range(n_eng)])
    check("sigma_eng varies across engines", float(eng_vals.std()) > 1e-3,
          f"std={float(eng_vals.std()):.4f}")

    # gradients reach the heads but never the backbone
    out.scale.sum().backward()
    grad_heads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.win_head.parameters())
    grad_eng = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.eng_head.parameters())
    grad_backbone = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    check("grad flows to window head", grad_heads)
    check("grad flows to engine head", grad_eng)
    check("backbone frozen", not grad_backbone)

    # ablations
    m_noeng = TwoScaleModel(StubBackbone(d), d, use_engine=False)
    o = m_noeng(x, ids, n_engines=n_eng)
    check("ablation use_engine=False -> sigma_eng == 1",
          torch.allclose(o.sigma_eng, torch.ones_like(o.sigma_eng)))
    m_nowin = TwoScaleModel(StubBackbone(d), d, use_window=False)
    o = m_nowin(x, ids, n_engines=n_eng)
    check("ablation use_window=False -> sigma_win == 1",
          torch.allclose(o.sigma_win, torch.ones_like(o.sigma_win)))

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
