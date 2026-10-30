
import sys, torch
sys.path.insert(0, "src")
from model import CRGraphTransformer
base = dict(num_sensors=14, window_size=50, dropout=0.15, num_heads=4,
            num_transformer_layers=2, use_synergy=True, tail_window=5,
            ff_mult=1.8, tail_mode="adaptive")
for dm in (80, 88, 96):
    g = CRGraphTransformer(d_model=dm, use_crgcn=True, native_graph=True, **base)
    n = CRGraphTransformer(d_model=dm, use_crgcn=False, use_spatial=False, native_graph=False, **base)
    print(f"d_model={dm}: full={sum(p.numel() for p in g.parameters()):,}  wo_graph={sum(p.numel() for p in n.parameters()):,}")

g = CRGraphTransformer(d_model=88, use_crgcn=True, native_graph=True, **base)
x = torch.randn(8, 50, 14); cond = torch.randn(8, 50, 3)
filtered, A = g.spatial(x, cond)
print("graph is identity at init:", bool(torch.allclose(filtered, x, atol=0.0)))
print("adjacency shape:", tuple(A.shape), "row sums:", [round(float(v), 3) for v in A.sum(-1)[:3]])
g.eval()
with torch.no_grad(): p, t = g(x, cond=cond, return_trajectory=True)
print("mono:", float((torch.diff(t, dim=1) > 0).float().mean()), "terminal:", bool(torch.allclose(p.view(-1), t[:, -1], atol=1e-6)))
