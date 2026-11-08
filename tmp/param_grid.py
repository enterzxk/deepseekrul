import sys
sys.path[:0]=["/home/user/rulb/tim_20261105","/home/user/rulb/scripts","/home/user/rulb/src","/home/user/rulb","/home/user/rulb/tim_20261105/full_20261105"]
import run_ida as ida
def count(net): return sum(p.numel() for p in net.parameters())
for d in [64,72,76,80,84]:
    for layers in [1,2]:
        body=ida.CandidateModel(num_sensors=14,window_size=50,d_model=d,num_layers=layers,dropout=0.15,
            use_baseline=True,use_ema_features=True,use_cycle=True,use_pe=True,use_graph=False,use_dual_head=False)
        print('d_model=%d layers=%d params %d'%(d,layers,count(body)))
