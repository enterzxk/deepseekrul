
import sys, torch
sys.path.insert(0, "src")
from model import CRGraphTransformer
base = dict(num_sensors=14, window_size=50, d_model=80, dropout=0.15,
            num_heads=4, num_transformer_layers=2, num_gcn_layers=1,
            use_synergy=True, tail_window=5, ff_mult=1.8, tail_mode="adaptive")
g = CRGraphTransformer(use_crgcn=True, native_graph=True, **base)
n = CRGraphTransformer(use_crgcn=False, native_graph=False, **base)
print("p10_full params:", sum(p.numel() for p in g.parameters()))
print("p10_wo_graph params:", sum(p.numel() for p in n.parameters()))
x = torch.randn(8, 50, 14); cond = torch.randn(8, 50, 3)
g.eval(); n.eval()
with torch.no_grad():
    pg, tg = g(x, cond=cond, return_trajectory=True)
    pn, tn = n(x, cond=cond, return_trajectory=True)
print("init identity (graph == no graph):", bool(torch.allclose(pg, pn, atol=1e-6)))
print("mono violation:", float((torch.diff(tg, dim=1) > 0).float().mean()))
print("terminal identity:", bool(torch.allclose(pg.view(-1), tg[:, -1], atol=1e-6)))
print("alpha at init:", float(torch.tanh(g.spatial.alpha_base).abs().max()))
print("tail weights:", [round(float(v), 3) for v in torch.softmax(g.tail_logits, 0)])
print("beta:", [round(float(v), 3) for v in torch.sigmoid(g.tail_beta)])
