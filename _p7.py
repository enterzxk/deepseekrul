import torch
from model import STG_MambaNet
for h in (36,40,44,48):
    m=STG_MambaNet(num_sensors=14,window_size=50,d_model=64,graph_mode="hollow",hollow_alpha=0.10,monotonic_head=True,use_delta_features=False,condition_film=True,temporal_mode="bilstm_attn",lstm_hidden_size=h)
    x=torch.randn(4,50,14); c=torch.randn(4,50,3)
    y,t=m(x,cond=c,return_trajectory=True)
    print(h,m.num_parameters,y.shape,t.shape,float((t[:,1:]-t[:,:-1]).max()),float(m.spatial_branch.graph_alpha))
