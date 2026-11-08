import sys, torch
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb","/home/user/rulb/tim_20261105/full_20261105"]
import run_ida as ida
from run_full_fold3 import build_net
def count(net):
    return sum(p.numel() for p in net.parameters())
for arm,graph in [('M0',False),('M0',True)]:
    n=count(build_net(arm,50,graph))
    print('%-6s graph=%-5s params %d'%(arm,graph,n))
body=ida.CandidateModel(num_sensors=14,window_size=50,d_model=64,num_layers=1,dropout=0.15,
    use_baseline=True,use_ema_features=True,use_cycle=True,use_pe=True,use_graph=False,use_dual_head=False)
print('M1 wrapper +', count(ida.PriorWrapper(body,'M1'))-count(body))
