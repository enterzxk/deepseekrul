"""Boundary and parameter-independence tests (Stage 2 review round 2)."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidates.model import CandidateModel, SensorGraphAttention  # noqa: E402
from dataset import load_cmapss_train  # noqa: E402


def test_transformer_layers_are_independent() -> None:
    model = CandidateModel(num_layers=2, d_model=32)
    p0 = dict(model.layers[0].named_parameters())
    p1 = dict(model.layers[1].named_parameters())
    assert p0 and p1
    for name in p0:
        assert p0[name] is not p1[name], name
    # Zero/one-initialised tensors (biases, LayerNorm scales) may coincide;
    # the attention projection weights must not.
    probe = "self_attn.in_proj_weight"
    assert probe in p0 and not torch.equal(p0[probe], p1[probe]), probe


def test_graph_block_is_identity_at_init() -> None:
    torch.manual_seed(0)
    block = SensorGraphAttention(d_model=16, num_heads=2, cond_dim=3)
    h = torch.randn(5, 14, 16)
    cond = torch.zeros(5, 1, 3)
    with torch.no_grad():
        out = block(h, cond)
    assert torch.equal(out, h), (out - h).abs().max()


def test_train_loader_never_opens_test_files() -> None:
    real_open = builtins.open

    def guarded(path, *args, **kwargs):
        name = str(path)
        if "test_FD" in name or "RUL_FD" in name:
            raise AssertionError(f"development path opened {name}")
        return real_open(path, *args, **kwargs)

    builtins.open = guarded
    try:
        frame = load_cmapss_train(ROOT / "data" / "CMAPSS", "FD001")
    finally:
        builtins.open = real_open
    assert frame.shape[0] > 0 and "unit" in frame.columns


if __name__ == "__main__":
    test_transformer_layers_are_independent()
    test_graph_block_is_identity_at_init()
    test_train_loader_never_opens_test_files()
    print("boundary and independence tests passed")
