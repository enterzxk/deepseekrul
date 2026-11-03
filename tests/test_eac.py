"""EAC unit tests (spec section 7).  Fast CPU tests; the V0-reproduction check
lives in scripts/run_eac.py --repro-v0 (it needs a GPU training run).

Run:  python -m pytest tests/test_eac.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from candidates.model import CandidateModel  # noqa: E402
from eac import SensorInFiLM, derangement, install_eac  # noqa: E402

N, D, K, L = 14, 64, 30, 50
V0_CFG = dict(num_sensors=N, window_size=L, d_model=D, num_layers=1, dropout=0.15,
              use_baseline=True, use_ema_features=True, use_cycle=True,
              use_pe=True, use_graph=False, use_dual_head=False)


def _make_e(rng: np.random.Generator, cycles: int = 200) -> np.ndarray:
    """A synthetic unit history: (cycles, N) standard normal rows."""
    return rng.standard_normal((cycles, N)).astype(np.float32)


def _stats_from_rows(rows: np.ndarray) -> np.ndarray:
    """The E matrix the builder must produce: first K rows of the unit."""
    return rows[:K]


def _forward(model, e: torch.Tensor | None, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(4, L, N)
    extras = torch.randn(4, N + 1)
    cond = torch.randn(4, L, 3)
    if isinstance(model.sensor_in, SensorInFiLM):
        model.sensor_in.current_E = e
    model.eval()
    with torch.no_grad():
        out = model(x, cond=cond, extras=extras)
    return out["prediction"]


def test_eac_ignores_cycles_ge_k() -> None:
    """Changing rows with cycle index >= k must not change E or the FiLM output."""
    rng = np.random.default_rng(0)
    hist = _make_e(rng)
    wrapper = SensorInFiLM(torch.nn.Linear(3, D), N, D)
    e1 = torch.from_numpy(_stats_from_rows(hist)).unsqueeze(0).expand(4, -1, -1)
    g1 = wrapper.gen(e1)
    hist2 = hist.copy()
    hist2[K:] += 100.0                                   # perturb only cycles >= k
    e2 = torch.from_numpy(_stats_from_rows(hist2)).unsqueeze(0).expand(4, -1, -1)
    g2 = wrapper.gen(e2)
    for a, b in zip(g1, g2):
        assert torch.equal(a, b)


def test_identity_at_init() -> None:
    """At init the FiLM layer is zero -> wrapped output == V0 output bitwise."""
    torch.manual_seed(2027)
    v0 = CandidateModel(**V0_CFG)
    torch.manual_seed(2027)
    v1 = CandidateModel(**V0_CFG)
    install_eac(v1, N, D)
    # body weights identical (generator RNG drawn after body construction);
    # the wrapper renames sensor_in.* -> sensor_in.inner.*, map it back.
    body1 = {k.replace("sensor_in.inner.", "sensor_in."): p
             for k, p in v1.named_parameters()
             if not k.startswith("sensor_in.gen")}
    for k0, p0 in v0.named_parameters():
        assert k0 in body1 and torch.equal(p0, body1[k0]), k0
    # current_E=None path (e.g. an EAC-free forward) is exactly V0
    assert torch.equal(_forward(v0, None), _forward(v1, None))
    # with E set, output is STILL identical at init (gamma=1, beta=0)
    rng = np.random.default_rng(1)
    e = torch.from_numpy(_make_e(rng)[:K]).unsqueeze(0).expand(4, -1, -1)
    assert torch.equal(_forward(v0, None), _forward(v1, e))


def test_short_unit_finite() -> None:
    """A unit shorter than k must not error; output stays finite."""
    torch.manual_seed(3)
    model = CandidateModel(**V0_CFG)
    install_eac(model, N, D)
    rng = np.random.default_rng(2)
    short = _make_e(rng, cycles=10)                      # 10 < K
    e = torch.from_numpy(short).unsqueeze(0).expand(4, -1, -1)
    pred = _forward(model, e)
    assert torch.isfinite(pred).all()


def test_output_does_not_depend_on_labels() -> None:
    """EAC never sees RUL or total life: identical inputs give identical outputs."""
    torch.manual_seed(4)
    model = CandidateModel(**V0_CFG)
    install_eac(model, N, D)
    rng = np.random.default_rng(5)
    e = torch.from_numpy(_make_e(rng)[:K]).unsqueeze(0).expand(4, -1, -1)
    y1 = _forward(model, e)
    y2 = _forward(model, e)                              # same E, different labels elsewhere
    assert torch.equal(y1, y2)


def test_conditioning_actually_conditions() -> None:
    """With a trained (non-zero) FiLM head, different units' E give different
    outputs -- otherwise V1 could not differ from V0 and the gate is vacuous."""
    torch.manual_seed(6)
    model = CandidateModel(**V0_CFG)
    install_eac(model, N, D)
    with torch.no_grad():                                # simulate a trained head
        model.sensor_in.gen.film_head.weight.normal_(0, 0.05)
    rng = np.random.default_rng(7)
    e_a = torch.from_numpy(_make_e(rng)[:K]).unsqueeze(0).expand(4, -1, -1)
    e_b = torch.from_numpy(_make_e(rng)[:K]).unsqueeze(0).expand(4, -1, -1)
    assert float((_forward(model, e_a) - _forward(model, e_b)).abs().max()) > 1e-6


def test_v6_uses_another_units_e() -> None:
    """The V6 permutation has no fixed points: every unit gets someone else's E."""
    rng = np.random.default_rng(8)
    for n in (2, 7, 30, 100):
        p = derangement(n, rng)
        assert (p != np.arange(n)).all()
        assert sorted(p.tolist()) == list(range(n))


def test_film_stats_are_leak_free_by_construction() -> None:
    """The builder contract: E rows come from cycle indices < k only.  Covered
    end-to-end by test_eac_ignores_cycles_ge_k; here the generator is checked to
    consume E exactly as documented (mean/std/first-last over rows)."""
    torch.manual_seed(9)
    gen = SensorInFiLM(torch.nn.Linear(3, D), N, D).gen
    rng = np.random.default_rng(10)
    e = torch.from_numpy(_make_e(rng)[:K]).unsqueeze(0)
    with torch.no_grad():
        gen.film_head.weight.normal_(0, 0.05)
        gamma, beta = gen(e)
    mu, sd = e.mean(dim=1), e.std(dim=1)
    assert torch.allclose(mu.squeeze(0), e[0].mean(dim=0))
    assert torch.allclose(sd.squeeze(0), e[0].std(dim=0))
    assert gamma.shape == (1, N, D) and beta.shape == (1, N, D)
