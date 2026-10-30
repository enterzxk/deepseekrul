
import sys, torch
sys.path.insert(0, "src")
from model import CRGraphTransformer

for dm in (64, 72, 80, 88):
    for nl in (2, 3):
        for gl in (1, 2):
            m = CRGraphTransformer(num_sensors=14, window_size=50, d_model=dm,
                                   num_transformer_layers=nl, num_gcn_layers=gl, dropout=0.15)
            print(f"d_model={dm} tf={nl} gcn={gl} params={sum(p.numel() for p in m.parameters()):,}")
    print()

m = CRGraphTransformer(num_sensors=14, window_size=50, d_model=64, dropout=0.15)
x = torch.randn(64, 50, 14); cond = torch.randn(64, 50, 3)
m.train(); p, t = m(x, cond=cond, return_trajectory=True)
m.eval()
with torch.no_grad(): p2, t2 = m(x, cond=cond, return_trajectory=True)
print("train traj", tuple(t.shape), "eval traj", tuple(t2.shape))
print("mono violation:", float((torch.diff(t2, dim=1) > 0).float().mean()))
print("terminal identity:", bool(torch.allclose(p2.view(-1), t2[:, -1], atol=1e-6)))
print("range:", float(t2.min()), float(t2.max()))
print("alpha init:", float(m.spatial.max_alpha * torch.sigmoid(torch.zeros(1) - 1.5)))
print("tail weights:", [round(float(v), 3) for v in torch.softmax(m.tail_logits, 0)])

for dm in (76, 78, 80):
    m = CRGraphTransformer(num_sensors=14, window_size=50, d_model=dm,
                           num_transformer_layers=2, num_gcn_layers=1, dropout=0.15)
    print(f"d_model={dm} params={sum(p.numel() for p in m.parameters()):,}")
