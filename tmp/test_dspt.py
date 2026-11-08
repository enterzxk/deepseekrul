"""D0 unit tests for DSPT-v2."""
import sys
sys.path.insert(0, 'src')
import torch
from candidates.dspt import build, masked_mean, MAX_SPAN, SHORT_SPAN

def test_params():
    for L, expect in [(1, 45506), (2, 78978)]:
        m = build(num_layers=L)
        n = sum(p.numel() for p in m.parameters())
        assert n == expect, f'{L}L got {n} expected {expect}'
        print(f'  params {L}L = {n} OK')

def test_masked_mean():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    out = masked_mean(h, mask)
    assert torch.allclose(out, torch.tensor([[4.0, 4.0]])), out
    print('  masked mean ignores padded step OK')

def test_short_equals_long_when_short_history():
    torch.manual_seed(0)
    m = build(num_layers=1).eval()
    B = 4
    x = torch.randn(B, MAX_SPAN, 14)
    mask = torch.zeros(B, MAX_SPAN)
    for i, mm in enumerate([30, 40, 50, 50]):
        mask[i, MAX_SPAN - mm:] = 1.0
    out = m(x, mask=mask)
    assert torch.allclose(out['y_A'], out['y_B'], atol=1e-5), (out['y_A'] - out['y_B']).abs().max()
    print('  m<=50: stream A identical to stream B OK')

def test_gate_monotone():
    m = build(num_layers=1).eval()
    x = torch.randn(5, MAX_SPAN, 14)
    mask = torch.zeros(5, MAX_SPAN)
    for i, mm in enumerate([50, 75, 100, 60, 100]):
        mask[i, MAX_SPAN - mm:] = 1.0
    g = m(x, mask=mask)['gate'].squeeze(-1)
    assert g[0].item() == 0.0 and g[2].item() == 1.0
    assert g[1].item() > 0.0 and g[1].item() < 1.0
    print('  gate ramp 0 at m=50, 1 at m=100 OK')

def test_padding_invariance():
    torch.manual_seed(1)
    m = build(num_layers=1).eval()
    x = torch.randn(2, MAX_SPAN, 14)
    mask = torch.zeros(2, MAX_SPAN)
    mask[:, MAX_SPAN - 100:] = 1.0
    a = m(x, mask=mask)['prediction']
    x2 = x.clone()
    x2[:, :MAX_SPAN - 100] = 999.0  # garbage where mask is 0
    b = m(x2, mask=mask)['prediction']
    assert torch.allclose(a, b, atol=1e-4), 'padding leaked into the prediction'
    print('  padded positions do not affect prediction OK')

if __name__ == '__main__':
    print('D0 unit tests')
    test_params()
    test_masked_mean()
    test_short_equals_long_when_short_history()
    test_gate_monotone()
    test_padding_invariance()
    print('ALL PASS')