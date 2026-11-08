import sys; sys.path.insert(0,'src')
import torch
from candidates.dspt import build
for L in (1,2):
    m = build(num_layers=L)
    print('layers', L, 'params', sum(p.numel() for p in m.parameters()))
m = build(num_layers=1).eval()
x = torch.randn(4, 100, 14)
mask = torch.zeros(4, 100)
for i, mm in enumerate([30, 60, 100, 100]):
    mask[i, 100-mm:] = 1.0
out = m(x, mask=mask, mode='slot')
print('pred shape', out['prediction'].shape)
print('slot weights', out['slot_w'].detach().numpy().round(3))
print('slot rows sum to 1:', out['slot_w'].sum(dim=1).detach().numpy().round(4))