"""Module B/C equivalence tests.

T1: DegAttnEncoderLayer(bias=None) == nn.TransformerEncoderLayer numerically.
T2: DegAttentionBias at init produces an all-zero bias (zero-init gate).
T3: DegCandidateModel(M1p config + modules) at init is numerically identical
    to the stock M1p PriorWrapper (module capacity present, output unchanged).
T4: Non-zero condition moves the outputs (bias/gate actually consume deg);
    and B_deg all-zero tensor passed explicitly is still identity.
T5: Parameter budget: module B/C additions < 10% of M1p total.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidates.model import CandidateModel  # noqa: E402
from candidates.deg_modules import (  # noqa: E402
    DegAttnEncoderLayer, DegAttentionBias, DegCandidateModel, DegSensorMixer,
)
from run_ida import PriorWrapper  # noqa: E402


def _kwargs():
    return dict(num_sensors=14, window_size=50, d_model=64, num_layers=1,
                dropout=0.15, use_baseline=True, use_ema_features=True,
                use_cycle=True, use_pe=True, use_graph=False, use_dual_head=False)


def _inputs(batch=4, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 50, 14, generator=g)
    cond = torch.zeros(batch, 50, 3)
    extras = torch.randn(batch, 21, generator=g)
    deg = torch.randn(batch, 9, generator=g)
    return x, cond, extras, deg


def test_t1_layer_equivalence():
    torch.manual_seed(3)
    stock = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0,
                                       activation="gelu", batch_first=True,
                                       norm_first=True)
    torch.manual_seed(3)
    deg = DegAttnEncoderLayer(64, 4, 128, dropout=0.0)
    deg.load_state_dict(stock.state_dict())
    x = torch.randn(2, 50, 64)
    with torch.no_grad():
        d = float((stock(x) - deg(x, None)).abs().max())
    assert d < 1e-6, f"T1 layer mismatch {d}"
    print("T1 layer equivalence OK", flush=True)


def test_t2_bias_zero_init():
    b = DegAttentionBias()
    out = b(torch.randn(5, 9))
    assert float(out.abs().max()) == 0.0, "T2 bias not zero at init"
    print("T2 bias zero-init OK", flush=True)


def test_t3_init_equivalence_to_m1p():
    torch.manual_seed(7)
    body = CandidateModel(**_kwargs())
    wrapper = PriorWrapper(body, "M1p").eval()
    torch.manual_seed(7)
    deg_model = DegCandidateModel(module_b=True, module_c=True, **_kwargs()).eval()
    missing, unexpected = deg_model.load_state_dict(body.state_dict(), strict=False)
    assert not unexpected, f"T3 unexpected keys {unexpected[:5]}"
    assert all("layers" not in k for k in missing), \
        f"T3 body keys missing: {[k for k in missing if 'layers' in k][:5]}"
    x, cond, extras, deg = _inputs()
    prior = torch.randn(4, 9, generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        ref = wrapper(x, cond, extras, prior)["prediction"]
        got = deg_model(x, cond, extras, deg, prior)["prediction"]
    d = float((ref - got).abs().max())
    assert d < 1e-4, f"T3 init equivalence broken: {d}"
    print(f"T3 init equivalence OK (max delta {d:.2e})", flush=True)


def test_t4_condition_moves_output():
    torch.manual_seed(7)
    deg_model = DegCandidateModel(module_b=True, module_c=True, **_kwargs()).eval()
    x, cond, extras, deg = _inputs()
    with torch.no_grad():
        # break the zero-init deliberately: only then can the condition act
        deg_model.deg_bias.alpha.weight.fill_(0.05)
        deg_model.deg_bias.alpha.bias.fill_(0.05)
        deg_model.mixer.gate.weight.fill_(0.05)
        deg_model.mixer.gate.bias.fill_(0.05)
        pr = torch.randn(4, 9, generator=torch.Generator().manual_seed(13))
        base = deg_model(x, cond, extras, torch.zeros_like(deg), pr)["prediction"]
        moved = deg_model(x, cond, extras, deg, pr)["prediction"]
    d = float((base - moved).abs().max())
    assert d > 1e-6, f"T4 condition has no effect ({d})"
    big = deg_model(x, cond, extras, deg * 50, pr)["prediction"]
    assert torch.isfinite(big).all(), "T4 non-finite output for large cond"
    print(f"T4 condition sensitivity OK (delta {d:.3e})", flush=True)


def test_t5_param_budget():
    torch.manual_seed(7)
    body = CandidateModel(**_kwargs())
    n_body = sum(p.numel() for p in body.parameters())
    deg_model = DegCandidateModel(module_b=True, module_c=True, **_kwargs())
    n_deg = sum(p.numel() for p in deg_model.parameters())
    extra = n_deg - n_body
    frac = extra / (n_body + 640)  # vs M1p total (body + 640 context proj)
    assert frac < 0.10, f"T5 module params {extra} = {frac:.1%} of M1p"
    print(f"T5 param budget OK: +{extra} params ({frac:.2%} of M1p)", flush=True)


def test_t6_mixer_gate_behaviour():
    m = DegSensorMixer()
    h = torch.randn(2, 50, 14, 64)
    z = m(h, torch.zeros(2, 9))
    assert float((z - h).abs().max()) == 0.0, "T6 mixer not identity at init"
    with torch.no_grad():
        m.gate.weight.fill_(0.1)
        m.gate.bias.fill_(0.1)
        moved = m(h, torch.randn(2, 9))
    assert float((moved - h).abs().max()) > 1e-6, "T6 gate has no effect"
    print("T6 mixer gate behaviour OK", flush=True)


if __name__ == "__main__":
    test_t1_layer_equivalence()
    test_t2_bias_zero_init()
    test_t3_init_equivalence_to_m1p()
    test_t4_condition_moves_output()
    test_t5_param_budget()
    test_t6_mixer_gate_behaviour()
    print("ALL MODULE B/C TESTS PASS", flush=True)
