"""Tests for the scale/bias heads and the frozen-backbone wrapper (needs torch)."""

import unittest

import torch

from src.punet.heads import BiasHead, ScaleHead
from src.punet.model import PUNetModel
from src.punet.features import PriorFeatures


def make_feats(b=4, sigma=None, tau=None):
    sigma = torch.full((b,), 2.0) if sigma is None else sigma
    tau = torch.full((b,), 0.5) if tau is None else tau
    return PriorFeatures(mu=torch.full((b,), 100.0), sigma=sigma, tau=tau, valid=torch.ones(b, dtype=torch.bool))


class TestHeads(unittest.TestCase):
    def test_scale_is_monotone_in_sigma(self):
        torch.manual_seed(0)
        head = ScaleHead(d_model=8)
        h = torch.randn(5, 8)
        tau = torch.linspace(0.1, 0.9, 5)
        sigma = torch.linspace(1.0, 5.0, 5)
        w = head(h, sigma, tau)
        self.assertTrue(bool((w[1:] > w[:-1]).all()), "scale must increase with sigma")

    def test_scale_initialises_as_power_law(self):
        torch.manual_seed(0)
        head = ScaleHead(d_model=8, eps=0.0)
        h = torch.randn(3, 8)
        sigma = torch.tensor([1.0, 2.0, 4.0])
        tau = torch.full((3,), 0.5)
        w = head(h, sigma, tau)
        self.assertTrue(torch.allclose(w, sigma, atol=1e-5), w)

    def test_bias_starts_at_zero(self):
        torch.manual_seed(0)
        head = BiasHead(d_model=8)
        b = head(torch.randn(6, 8), torch.rand(6) + 1.0)
        self.assertTrue(torch.allclose(b, torch.zeros_like(b), atol=1e-8))


class _TinyBackbone(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.enc = torch.nn.Linear(3, d)
        self.out = torch.nn.Linear(d, 1)

    def encode(self, x):
        pooled = x.mean(dim=1)
        h = self.enc(pooled)
        return h, self.out(h).squeeze(-1)


class TestModel(unittest.TestCase):
    def test_backbone_is_frozen(self):
        model = PUNetModel(_TinyBackbone(), d_model=8)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.parameters()))

    def test_initial_equivalence_to_base(self):
        torch.manual_seed(0)
        model = PUNetModel(_TinyBackbone(), d_model=8)
        x = torch.randn(4, 30, 3)
        out = model(x, None, make_feats(4))
        self.assertTrue(torch.allclose(out.y_hat, out.y_base, atol=1e-6))

    def test_ablation_switches(self):
        torch.manual_seed(0)
        x = torch.randn(4, 30, 3)
        m_a1 = PUNetModel(_TinyBackbone(), d_model=8, use_scale_head=False)
        self.assertTrue(torch.allclose(m_a1(x, None, make_feats(4)).w, torch.ones(4)))
        m_a6 = PUNetModel(_TinyBackbone(), d_model=8, use_bias_head=False)
        self.assertTrue(torch.allclose(m_a6(x, None, make_feats(4)).y_hat, m_a6(x, None, make_feats(4)).y_base))
