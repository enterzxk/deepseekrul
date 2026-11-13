import torch
from paper_b.hc import train_d as T

fo = dict(pred=torch.tensor([10., 20., 30.]),
          r=torch.tensor([5., 3., 0.]),
          delta=torch.tensor([0.5, 0.6, 0.7]),
          D=torch.tensor([0.5, 1.1, 1.8]))

acc0 = T.term_sums(None, 'B1', fo, 125.0, scr=False)
acc1 = T.term_sums(None, 'B1', fo, 125.0, scr=True)
L0, L1 = T.aggregate(acc0, 1.0), T.aggregate(acc1, 1.0)

alive = float(L1['alive'])
term = float(L1['terminal'])
# alive mask is r>0 -> cycles 0 and 1; overshoot^2 = 0 and 0.01 -> mean 0.005
expect_alive = (0.0 + (1.1 - 1.0) ** 2) / 2.0
print('keys scr=False:', sorted(acc0))
print('keys scr=True :', sorted(acc1))
print('alive = %.6f  expect %.6f' % (alive, expect_alive))
print('terminal = %.6f  expect %.6f' % (term, (1.8 - 1.0) ** 2))
lam_a = 125.0 ** 2
print('total delta = %.6f  expect %.6f' % (lam_a * alive, lam_a * expect_alive))
assert abs(alive - expect_alive) < 1e-6
assert abs(term - 0.64) < 1e-6
assert 'alive' not in L0
# gradient reaches the damage path
D = fo['D'].clone().requires_grad_(True)
fo2 = dict(fo, D=D)
acc2 = T.term_sums(None, 'B1', fo2, 125.0, scr=True)
L2 = T.aggregate(acc2, 1.0)
(L2['alive']).backward()
print('dL_alive/dD =', D.grad.tolist())
assert D.grad[0].item() == 0.0 and D.grad[1].item() > 0.0
print('SCR UNIT OK')
