import sys
sys.path[:0]=['/home/user/rulb/tim_20261105','/home/user/rulb/scripts','/home/user/rulb/src','/home/user/rulb']
import run_ida as ida
for d,l in [(64,1),(128,2),(128,4),(192,2)]:
    m=ida.CandidateModel(num_sensors=14, window_size=50, d_model=d, num_layers=l, dropout=0.15,
                         use_baseline=True, use_ema_features=True, use_cycle=True, use_pe=True,
                         use_graph=False, use_dual_head=False)
    print('d_model=%d layers=%d -> %d params'%(d,l,sum(p.numel() for p in m.parameters())))