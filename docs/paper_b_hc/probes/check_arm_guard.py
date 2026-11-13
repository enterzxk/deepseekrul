"""Verify the arm-dispatch guard: unknown names must raise, known ones must build.

Regression for the failure 燕子's build gate found: build_d had no final else, so
an unregistered or misspelled arm silently built B0 and produced a genuine B0
result under another arm's run_id.
"""
import torch

from paper_b.hc.train_d import ALL_ARMS, build_d

INIT = 'experiments/hc/tmp/wave1_init_A1_s101.pt'
dev = torch.device('cpu')

print('ALL_ARMS =', list(ALL_ARMS))
try:
    build_d('NOPE', 'FD001', 600, None, 101, dev, INIT)
except ValueError as e:
    print('PASS unknown arm raises: %s' % str(e)[:70])
else:
    raise SystemExit('FAIL: build_d(NOPE) did not raise')

for arm in ('B0', 'M2'):
    net, missing = build_d(arm, 'FD001', 600, None, 101, dev, INIT)
    n = sum(p.numel() for p in net.parameters())
    print('PASS %-3s builds: params=%d new_tensors=%d' % (arm, n, len(missing)))
print('GUARD OK')
