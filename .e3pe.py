import sys, torch
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from candidates.model import CandidateModel
base=dict(num_sensors=14, window_size=50, d_model=64, num_layers=1, dropout=0.15,
          use_baseline=True, use_ema_features=True, use_cycle=True,
          use_graph=False, use_dual_head=False)
a=CandidateModel(**base); p=CandidateModel(**base, use_pe=True)
print('A params', sum(x.numel() for x in a.parameters()))
print('PE params', sum(x.numel() for x in p.parameters()))
x=torch.randn(4,50,14); ex=torch.rand(4,21); ex[:,14]=torch.rand(4)
a.eval(); p.eval()
with torch.no_grad():
    ya=a(x,extras=ex)['prediction']; yp=p(x,extras=ex)['prediction']
print('A finite', torch.isfinite(ya).all().item(), 'PE finite', torch.isfinite(yp).all().item())
print('outputs differ:', float((ya-yp).abs().max())>1e-6)
