"""Is B7's failure a saturated clamp, or the zero-init output layer?

A zero-initialised LAST layer makes dL/dh = W^T dL/dy = 0 at step 0, so every
earlier layer sees exactly zero gradient on the first backward pass. That is the
same construction AAR/M2/M4/M6 use, and it is the reason their step-0 prediction
equals the parent's. If that is the cause, no clamp change can fix B7 -- the test
itself asks for something impossible at step 0.

Run on the 3090 (CPU is enough).
"""
import json

import torch

from paper_b.hc import train_d as T

SH = '/home/user/sprint24/experiments/hc/d/FD001_shared.json'
INIT = '/home/user/sprint24/experiments/hc/tmp/wave1_init_A1_s101.pt'

sh = json.load(open(SH))
dev = torch.device('cpu')

# Which arms carry an anchor head? M3B and (if registered) M3C.
arm = 'M3C' if 'M3C' in T.ALL_ARMS else 'M3B'
net, _ = T.build_d(arm, subset='FD001', K=int(sh['K']), l_mean=float(sh['l_mean']),
                   head_seed=int(sh['head_seed']), dev=dev, init_from=INIT, seed=101)
anchor = net.head.anchor
print('arm=%s anchor=%s' % (arm, anchor))

w_last = anchor[2].weight.detach()
print('last layer zero-init: max|w| = %.3e' % float(w_last.abs().max()))


def grad_probe(tag):
    z = torch.randn(64, net.d_model, generator=torch.Generator().manual_seed(0))
    z.requires_grad_(True)
    u = anchor(z).squeeze(-1)
    loss = u.sum()   # nonzero dL/du at u=0, like relu(margin - D_A + D_B) at the tie
    loss.backward()
    g0 = anchor[0].weight.grad
    g2 = anchor[2].weight.grad
    print('%s  |dL/dW0| = %.3e   |dL/dW2| = %.3e   |dL/dz| = %.3e'
          % (tag, float(g0.abs().max()), float(g2.abs().max()), float(z.grad.abs().max())))
    return float(g0.abs().max()), float(g2.abs().max())


g0, g2 = grad_probe('step 0 (zero-init last layer)')

# Same net, last layer made non-zero: the earlier layers must now receive gradient.
with torch.no_grad():
    anchor[2].weight.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(1))
g0b, g2b = grad_probe('after perturbing W2')

print('')
print('verdict: at step 0 gradient reaches the LAST layer (%s) but not layer 0 (%s)'
      % ('yes' if g2 > 0 else 'no', 'yes' if g0 > 0 else 'no'))
print('after W2 perturbation layer 0 receives gradient: %s' % ('yes' if g0b > 0 else 'no'))
