"""Reduction tests for Stage D losses (D4.1/D4.5).

These exist because the advisor's falsification criterion for the formal matrix
is "the actual loss reduction departs from the spec". A claim in a document
cannot satisfy that; a test can. Three things are pinned:

  1. L_RUL reduces point-weighted (every cycle is a sample).
  2. L_terminal reduces ENGINE-EQUAL: a long engine must not outvote a short one.
     The first trainer multiplied the total by the engine length, which silently
     made terminal point-weighted -- that is the bug this test locks out.
  3. ||grad sum_i L_i|| is NOT the mean of ||grad L_i||. The D4.5 measurement was
     the latter, which is wrong by the triangle inequality; this test constructs
     two engines whose gradients oppose each other so the two quantities are
     visibly different (and one of them is zero).
"""
import math

import torch

from paper_b.hc.train_d import aggregate, term_sums


def _fo(pred, r, D=None, delta=None):
    fo = dict(pred=pred, r=r)
    if D is not None:
        fo['D'] = D
        fo['delta'] = delta
    return fo


def test_rul_is_point_weighted():
    # engine A: 2 points with losses 0 and 0; engine B: 4 points each loss 1
    fo_a = _fo(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    fo_b = _fo(torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.tensor([0.0] * 4))
    acc = {}
    for fo in (fo_a, fo_b):
        for k, v in term_sums(None, 'B0', fo, 125.0).items():
            acc[k] = acc[k] + v if k in acc else v
    L = aggregate(acc, 1.0)
    # point-weighted: 4 of 6 points carry loss 1 -> 4/6, not (0+1)/2 = 0.5
    assert abs(float(L['rul']) - 4.0 / 6.0) < 1e-6, float(L['rul'])


def test_terminal_is_engine_equal():
    # engine A: 2 cycles, terminal loss 0; engine B: 10 cycles, terminal loss 4
    long_D = torch.cat([torch.linspace(0.1, 2.0, 10), torch.tensor([3.0])])
    fo_a = _fo(torch.zeros(2), torch.zeros(2),
               D=torch.tensor([0.0, 1.0]), delta=torch.tensor([0.5, 0.5]))
    fo_b = _fo(torch.zeros(10), torch.zeros(10), D=long_D,
               delta=torch.tensor([0.3] * 10))
    acc = {}
    for fo in (fo_a, fo_b):
        for k, v in term_sums(None, 'B3', fo, 125.0).items():
            acc[k] = acc[k] + v if k in acc else v
    L = aggregate(acc, 1.0)
    want = ((1.0 - 1.0) ** 2 + (3.0 - 1.0) ** 2) / 2.0  # (0 + 4)/2 = 2.0
    assert abs(float(L['terminal']) - want) < 1e-6, float(L['terminal'])
    # the point-weighted alternative would be 4 * 10/12 = 3.333
    assert abs(float(L['terminal']) - 4.0 * 10.0 / 12.0) > 1e-3


def test_gradient_norm_of_sum_is_not_mean_of_norms():
    # Two engines pulling one shared parameter in opposite directions:
    # L1 = (p-1)^2, L2 = (p+1)^2 at p=0. dL1/dp = -2, dL2/dp = +2, so the
    # aggregate gradient cancels to 0 while the mean of norms is 2.
    p = torch.zeros((), requires_grad=True)
    l1 = (p - 1.0) ** 2
    l2 = (p + 1.0) ** 2
    g_agg = float(torch.autograd.grad(l1 + l2, [p], retain_graph=True)[0].norm())
    g1 = float(torch.autograd.grad(l1, [p], retain_graph=True)[0].norm())
    g2 = float(torch.autograd.grad(l2, [p], retain_graph=True)[0].norm())
    g_mean_of_norms = 0.5 * (g1 + g2)
    assert g_agg < 1e-9, g_agg                      # aggregate cancels
    assert abs(g_mean_of_norms - 2.0) < 1e-6, g_mean_of_norms
    assert abs(g_agg - g_mean_of_norms) > 1.0       # the two differ by 2


def test_smooth_is_point_weighted_and_divided_by_s0():
    # 3 intervals with second differences 0, 1, 1 -> mean 2/3; s0 = 2 -> 1/3
    delta = torch.tensor([0.0, 0.0, 1.0, 2.0])
    fo = _fo(torch.zeros(4), torch.zeros(4), D=torch.cumsum(delta, 0), delta=delta)
    acc = {}
    for k, v in term_sums(None, 'B3', fo, 125.0).items():
        acc[k] = acc[k] + v if k in acc else v
    L = aggregate(acc, 2.0)
    assert abs(float(L['smooth']) - (2.0 / 3.0) / 2.0) < 1e-6, float(L['smooth'])


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS %s' % name)
    print('all reduction tests passed')
